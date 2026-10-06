from dataclasses import dataclass
from momentum_scanner import ranking, config


@dataclass
class C:
    ticker: str = "AAA"
    gap_pct: float = 0.20
    rel_volume: float = 4.0
    float_shares: float | None = 5_000_000
    edgar_signal: float = 0.0
    score: float = 0.0
    catalyst_score: float = 0.0


@dataclass
class V:
    confidence: float | None = None
    content_source: str | None = None
    dilution_flag: str | None = None
    label: str = "real"   # label-aware catalyst_bonus needs a label on the verdict fixture


def test_base_score_adds_net_edgar_signal():
    c = C(gap_pct=0.20, rel_volume=4.0, edgar_signal=0.5)
    # base = 0.20*4.0 + W_EDGAR(0.3)*0.5 = 0.80 + 0.15 = 0.95
    assert ranking.score(c) == round(0.20 * 4.0 + config.W_EDGAR * 0.5, 6)


def test_base_score_unchanged_when_no_edgar_signal():
    c = C(gap_pct=0.20, rel_volume=4.0, edgar_signal=0.0)
    assert ranking.score(c) == round(0.20 * 4.0, 6)   # regression: identical to legacy


def test_catalyst_bonus_confidence_and_source_weighted():
    c = C()
    v = V(confidence=0.9, content_source="finnhub")
    # W_CATALYST(0.2) * 0.9 * SOURCE_WEIGHTS["finnhub"](0.7)
    assert ranking.catalyst_bonus(c, v) == round(config.W_CATALYST * 0.9 * 0.7, 6)


def test_catalyst_bonus_zero_for_pump_label():
    # A high-confidence pump/no-data verdict must NOT be rewarded.
    c = C()
    v = V(confidence=0.95, content_source="finnhub", label="pump")
    assert ranking.catalyst_bonus(c, v) == 0.0


def test_effective_score_dilution_wraps_everything():
    c = C(gap_pct=0.20, rel_volume=4.0, float_shares=5_000_000)   # float_factor 1.0
    v = V(confidence=0.9, content_source="finnhub", dilution_flag="active_offering")   # 0.3x
    base = ranking.score(c)
    bonus = ranking.catalyst_bonus(c, v)
    expected = round((base * 1.0 + bonus) * config.DILUTION_FACTORS["active_offering"], 6)
    assert ranking.effective_score(c, {"AAA": v}) == expected


def test_effective_score_unknown_dilution_penalizes():
    c = C(float_shares=5_000_000)
    v = V(confidence=None, content_source=None, dilution_flag="unknown")
    # no verdict confidence -> bonus 0; unknown -> 0.85x
    assert ranking.effective_score(c, {"AAA": v}) == round(ranking.score(c) * 1.0 * 0.85, 6)


def test_weaker_move_with_catalyst_outscores_stronger_move_without():
    # THE behavioral proof. Pinned fixtures.
    strong_no_cat = C(ticker="STRONG", gap_pct=0.30, rel_volume=4.0, edgar_signal=0.0)   # base 1.20
    weak_cat = C(ticker="WEAK", gap_pct=0.25, rel_volume=4.0, edgar_signal=0.6)           # base 1.00 + 0.18
    v_strong = V(confidence=None, content_source=None)
    v_weak = V(confidence=0.95, content_source="edgar")
    verdicts = {"STRONG": v_strong, "WEAK": v_weak}
    assert ranking.effective_score(weak_cat, verdicts) > ranking.effective_score(strong_no_cat, verdicts)


def test_rank_candidates_stamps_catalyst_score():
    c = C(ticker="AAA")
    v = V(confidence=0.8, content_source="finnhub")
    ranking.rank_candidates([c], {"AAA": v})
    assert c.catalyst_score == ranking.catalyst_bonus(c, v)
    assert c.score == ranking.effective_score(c, {"AAA": v})


def test_effective_score_float_placement_pinned():
    # Every OTHER fixture in this file uses float_shares=5_000_000
    # (float_factor==1.0), which makes the correct grouping (base*float_factor + bonus)*dil
    # arithmetically IDENTICAL to the WRONG grouping (base+bonus)*float_factor*dil. Use a high
    # float (float_factor < 1.0) + a non-zero catalyst_bonus so the two groupings actually diverge,
    # so this test truly pins the placement of float_factor in effective_score's arithmetic.
    c = C(gap_pct=0.20, rel_volume=4.0, edgar_signal=0.0, float_shares=30_000_000)
    v = V(confidence=0.9, content_source="finnhub", dilution_flag=None, label="real")
    base = ranking.score(c)                        # 0.20*4.0 = 0.80
    ff = ranking.float_factor(c.float_shares)       # 1 - 1.5*((30M-20M)/20M)^2 = 0.625
    bonus = ranking.catalyst_bonus(c, v)            # round(0.2*0.9*0.7*1.0, 6) = 0.126
    dil = config.DILUTION_FACTORS[None]             # 1.0
    correct = round((base * ff + bonus) * dil, 6)          # ~0.626
    wrong = round((base + bonus) * ff * dil, 6)             # ~0.579
    assert correct != wrong          # sanity: fixtures actually distinguish the two groupings
    got = ranking.effective_score(c, {"AAA": v})
    assert got == correct
    assert got != wrong

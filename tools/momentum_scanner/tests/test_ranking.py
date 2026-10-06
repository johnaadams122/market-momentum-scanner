from momentum_scanner import config
from momentum_scanner import ranking
from momentum_scanner.movers_scan import MoverCandidate
from momentum_scanner.catalyst_evaluator import CatalystVerdict


def _c(ticker, gap, relvol, float_shares):
    return MoverCandidate(ticker=ticker, company_name=ticker, premarket_price=5.0, prev_close=4.0,
                          gap_pct=gap, float_shares=float_shares, premarket_volume=1_000_000,
                          baseline_volume=500_000, rel_volume=relvol, day_high=5.0, new_hod=True,
                          scan_mode="intraday", last_seen_at=None)


def test_score_orders_by_gap_times_relvol():
    assert ranking.score(_c("HI", 0.20, 5.0, 4_000_000)) > ranking.score(_c("LO", 0.10, 5.0, 4_000_000))


def test_none_relvol_uses_neutral_multiplier():
    assert ranking.score(_c("X", 0.30, None, 4_000_000)) == round(0.30 * 1.0, 6)


def test_known_float_ranked_above_unknown():
    a = _c("A", 0.30, 5.0, 4_000_000)       # known, score 1.5
    b = _c("B", 0.10, 5.0, 4_000_000)       # known, score 0.5
    u = _c("U", 0.90, 9.0, None)            # unknown, big score but bucketed below knowns
    ranked = ranking.rank_candidates([b, u, a])
    assert [c.ticker for c in ranked] == ["A", "B", "U"]
    assert ranked[0].score == round(0.30 * 5.0, 6)


def test_confidence_breaks_ties_when_verdicts_given():
    # two equal numeric scores (gap*relvol = 1.0); higher catalyst confidence ranks first.
    x = _c("X", 0.20, 5.0, 4_000_000)
    y = _c("Y", 0.20, 5.0, 4_000_000)
    verdicts = {"X": CatalystVerdict("PROCEED", "real", "r", 0.6, "llm"),
                "Y": CatalystVerdict("PROCEED", "real", "r", 0.9, "llm")}
    ranked = ranking.rank_candidates([x, y], verdicts=verdicts)
    assert [c.ticker for c in ranked] == ["Y", "X"]


def test_unknown_float_bucket_capped_with_scrambled_input():
    knowns = [_c(f"K{i}", 0.10, 2.0, 4_000_000) for i in range(2)]
    # scrambled scores so a naive slice (no sort) would keep the wrong ones.
    unknowns = [_c("U_lo", 0.10, 1.0, None), _c("U_hi", 0.90, 9.0, None),
                _c("U_mid", 0.50, 5.0, None)]
    unknowns = unknowns * 3                                  # > UNKNOWN_FLOAT_CAP copies
    ranked = ranking.rank_candidates(knowns + unknowns)
    kept = [c for c in ranked if c.float_shares is None]
    assert len(kept) == config.UNKNOWN_FLOAT_CAP
    assert kept[0].ticker == "U_hi"                          # highest-scored unknown kept first


def test_empty_and_all_unknown():
    assert ranking.rank_candidates([]) == []
    allu = [_c(f"U{i}", 0.50 - i * 0.01, 5.0, None) for i in range(config.UNKNOWN_FLOAT_CAP + 2)]
    assert len(ranking.rank_candidates(allu)) == config.UNKNOWN_FLOAT_CAP


from types import SimpleNamespace


def _cand(ticker, gap, relvol, fs):
    return SimpleNamespace(ticker=ticker, gap_pct=gap, rel_volume=relvol, float_shares=fs, score=0.0)


def test_float_factor_curve():
    assert ranking.float_factor(None) == 1.0
    assert ranking.float_factor(10_000_000) == 1.0
    assert ranking.float_factor(20_000_000) == 1.0
    assert round(ranking.float_factor(22_000_000), 2) == 0.98
    assert round(ranking.float_factor(30_000_000), 3) == 0.625   # exact 0.625; round(_,2) would banker's-round to 0.62
    assert round(ranking.float_factor(35_000_000), 2) == 0.16


def test_dilution_factor_lookup():
    assert ranking.dilution_factor("active_offering") == 0.3
    assert ranking.dilution_factor("reverse_split_history") == 1.0
    assert ranking.dilution_factor(None) == 1.0
    assert ranking.dilution_factor("bogus") == 1.0


def test_effective_score_composition_written_to_score():
    c = _cand("AAA", 0.5, 4.0, 30_000_000)  # base = 2.0
    v = {"AAA": SimpleNamespace(dilution_flag="active_offering", confidence=0.9)}
    ranked = ranking.rank_candidates([c], verdicts=v)
    # base 2.0 * float_factor(30M)=0.625 * dilution 0.3 = 0.375
    assert round(ranked[0].score, 3) == 0.375


def test_reverse_split_outranks_active_offering_same_base():
    a = _cand("RS", 0.5, 4.0, 5_000_000)     # base 2.0, reverse_split -> 2.0
    b = _cand("AO", 0.5, 4.0, 5_000_000)     # base 2.0, active_offering -> 0.6
    v = {"RS": SimpleNamespace(dilution_flag="reverse_split_history", confidence=0.5),
         "AO": SimpleNamespace(dilution_flag="active_offering", confidence=0.5)}
    ranked = ranking.rank_candidates([b, a], verdicts=v)
    assert [c.ticker for c in ranked] == ["RS", "AO"]


def test_unknown_float_appended_and_capped():
    known = _cand("K", 0.5, 4.0, 5_000_000)
    unknowns = [_cand(f"U{i}", 0.9, 9.0, None) for i in range(10)]
    ranked = ranking.rank_candidates([known] + unknowns, verdicts={})
    assert ranked[0].ticker == "K"
    assert sum(1 for c in ranked if c.float_shares is None) == 5  # UNKNOWN_FLOAT_CAP


def test_effective_score_with_real_catalyst_verdict():
    # Thread a REAL CatalystVerdict (with the new dilution_flag field) end-to-end so a producer/consumer
    # field-name mismatch with ranking._dilution_flag would surface here (SimpleNamespace can't catch it).
    from momentum_scanner.catalyst_evaluator import CatalystVerdict
    c = _cand("AAA", 0.5, 4.0, 5_000_000)     # base 2.0, float_factor(5M)=1.0
    v = {"AAA": CatalystVerdict("PROCEED", "real", "ok", 0.9, "llm", dilution_flag="active_offering")}
    ranked = ranking.rank_candidates([c], verdicts=v)
    assert round(ranked[0].score, 3) == 0.6   # 2.0 * 1.0 * 0.3

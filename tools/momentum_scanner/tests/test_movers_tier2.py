from datetime import datetime, timezone
from momentum_scanner import config, movers_scan
from momentum_scanner.movers_scan import Tier1Hit

NOW = datetime(2026, 6, 29, 14, 0, tzinfo=timezone.utc)
QT = datetime(2026, 6, 29, 14, 0, tzinfo=timezone.utc)


def _hit(ticker, last=5.0, prev=4.0, gap=0.25, vol=1_500_000, high=5.0, avg=500_000, relvol=5.0):
    return Tier1Hit(ticker=ticker, last=last, prev_close=prev, gap_pct=gap, total_volume=vol,
                    day_high=high, avg_daily_volume=avg, rel_volume=relvol, quote_time=QT, source="regular")


def _floats(mapping):
    return lambda t: mapping.get(t)


def test_known_low_float_kept_intraday_at_hod():
    cands = movers_scan.tier2_enrich([_hit("AAA", last=5.0, high=5.0)], float_cache={},
                                     float_fetch_fn=_floats({"AAA": 4_000_000}), now_utc=NOW, premarket=False)
    assert len(cands) == 1
    c = cands[0]
    assert c.ticker == "AAA" and c.float_shares == 4_000_000 and c.new_hod is True
    assert c.premarket_price == 5.0 and c.scan_mode == "intraday" and c.last_seen_at == QT.isoformat()


def test_known_high_float_dropped():
    cands = movers_scan.tier2_enrich([_hit("BIG")], float_cache={},
                                     float_fetch_fn=_floats({"BIG": 50_000_000}), now_utc=NOW, premarket=False)
    assert cands == []


def test_unknown_float_kept_flagged():
    cands = movers_scan.tier2_enrich([_hit("UNK")], float_cache={},
                                     float_fetch_fn=_floats({}), now_utc=NOW, premarket=False)
    assert len(cands) == 1 and cands[0].float_shares is None


def _hod_threshold(high):
    """The predicate under test is `last >= day_high * (1 - NEW_HOD_TOLERANCE_PCT)`.

    Derived from config rather than pinned to a literal: the tolerance is a tuned knob, so
    hardcoded prices would break whenever it is retuned. What is worth pinning is the BOUNDARY
    SEMANTICS -- inclusive at the threshold, dropped below it -- not the knob's current value.
    That is guarded separately, as a literal, by test_config_constants.
    """
    return high * (1 - config.NEW_HOD_TOLERANCE_PCT)


def test_new_hod_tolerance_boundary():
    high, eps = 5.0, 0.001
    threshold = _hod_threshold(high)
    # Guard against a vacuous pass: if the tolerance is ever widened far enough that the "drop"
    # price falls out of the price band, this test would still go green while proving nothing
    # about the HOD gate. Fail loudly instead.
    assert config.PRICE_MIN < threshold - eps < config.PRICE_MAX, (
        "fixture no longer isolates the HOD gate at the configured tolerance")

    keep = movers_scan.tier2_enrich([_hit("K", last=threshold + eps, high=high)], float_cache={},
                                    float_fetch_fn=_floats({"K": 4_000_000}), now_utc=NOW, premarket=False)
    edge = movers_scan.tier2_enrich([_hit("E", last=threshold, high=high)], float_cache={},
                                    float_fetch_fn=_floats({"E": 4_000_000}), now_utc=NOW, premarket=False)
    drop = movers_scan.tier2_enrich([_hit("D", last=threshold - eps, high=high)], float_cache={},
                                    float_fetch_fn=_floats({"D": 4_000_000}), now_utc=NOW, premarket=False)
    assert len(keep) == 1 and keep[0].new_hod is True
    assert len(edge) == 1 and edge[0].new_hod is True      # boundary is INCLUSIVE (>=, not >)
    assert drop == []


def test_intraday_missing_dayhigh_dropped():
    cands = movers_scan.tier2_enrich([_hit("NOHI", high=None)], float_cache={},
                                     float_fetch_fn=_floats({"NOHI": 4_000_000}), now_utc=NOW, premarket=False)
    assert cands == []


def test_unknown_float_still_requires_hod_intraday():
    high = 5.0
    below = _hod_threshold(high) - 0.01                    # clearly outside the tolerance band
    assert config.PRICE_MIN < below < config.PRICE_MAX, (
        "fixture no longer isolates the HOD gate at the configured tolerance")
    drop = movers_scan.tier2_enrich([_hit("U", last=below, high=high)], float_cache={},
                                    float_fetch_fn=_floats({}), now_utc=NOW, premarket=False)
    keep = movers_scan.tier2_enrich([_hit("U2", last=high, high=high)], float_cache={},
                                    float_fetch_fn=_floats({}), now_utc=NOW, premarket=False)
    assert drop == []
    assert len(keep) == 1 and keep[0].float_shares is None and keep[0].new_hod is True


def test_premarket_does_not_require_hod():
    cands = movers_scan.tier2_enrich([_hit("PM", last=4.9, high=5.0)], float_cache={},
                                     float_fetch_fn=_floats({"PM": 4_000_000}), now_utc=NOW, premarket=True)
    assert len(cands) == 1 and cands[0].new_hod is None and cands[0].scan_mode == "premarket"


def test_per_symbol_isolation():
    class BadHit:
        ticker = "BAD"
        @property
        def day_high(self):
            raise RuntimeError("boom")
    cands = movers_scan.tier2_enrich([BadHit(), _hit("OK")], float_cache={},
                                     float_fetch_fn=_floats({"OK": 4_000_000}), now_utc=NOW, premarket=False)
    assert [c.ticker for c in cands] == ["OK"]


def test_float_cache_reused_across_hits():
    calls = {"n": 0}
    def fetch(t):
        calls["n"] += 1
        return 4_000_000
    cache = {}
    movers_scan.tier2_enrich([_hit("SAME")], float_cache=cache, float_fetch_fn=fetch, now_utc=NOW, premarket=False)
    movers_scan.tier2_enrich([_hit("SAME")], float_cache=cache, float_fetch_fn=fetch, now_utc=NOW, premarket=False)
    assert calls["n"] == 1     # second pass within the day serves the cached float


def test_tier2_keeps_30m_float_drops_40m():
    keep = movers_scan.tier2_enrich([_hit("KEEP")], float_cache={},
                                    float_fetch_fn=_floats({"KEEP": 30_000_000}), now_utc=NOW, premarket=False)
    stats = {}
    drop = movers_scan.tier2_enrich([_hit("DROP")], float_cache={},
                                    float_fetch_fn=_floats({"DROP": 40_000_000}), now_utc=NOW, premarket=False,
                                    stats=stats)
    assert len(keep) == 1 and keep[0].ticker == "KEEP" and keep[0].float_shares == 30_000_000
    assert drop == [] and stats["tier2_dropped_highfloat"] >= 1


def test_tier2_float_hard_max_boundary():
    # Strict `> FLOAT_HARD_MAX`: exactly 36M is KEPT; 36M+1 is dropped (guards an operator flip to >=).
    keep = movers_scan.tier2_enrich([_hit("EQ")], float_cache={},
                                    float_fetch_fn=_floats({"EQ": 36_000_000}), now_utc=NOW, premarket=False)
    drop = movers_scan.tier2_enrich([_hit("OVER")], float_cache={},
                                    float_fetch_fn=_floats({"OVER": 36_000_001}), now_utc=NOW, premarket=False)
    assert len(keep) == 1 and keep[0].float_shares == 36_000_000
    assert drop == []

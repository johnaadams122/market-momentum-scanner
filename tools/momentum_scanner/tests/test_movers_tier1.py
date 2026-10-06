from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from momentum_scanner import movers_scan
from momentum_scanner.adapters.schwab_quotes import QuoteSnapshot

ET = ZoneInfo("America/New_York")
NOW_MID = datetime(2026, 6, 29, 13, 0, tzinfo=ET)        # midday RTH; = 17:00 UTC
NOW_MID_UTC = NOW_MID.astimezone(timezone.utc)
NOW_PM = datetime(2026, 6, 29, 8, 30, tzinfo=ET)         # premarket; = 12:30 UTC
NOW_PM_UTC = NOW_PM.astimezone(timezone.utc)


def _q(ticker, last, prev, vol, high, avg, qt=NOW_MID_UTC):
    return QuoteSnapshot(ticker=ticker, last=last, prev_close=prev, total_volume=vol,
                         day_high=high, avg_daily_volume=avg, quote_time=qt, source="regular")


def test_passing_candidate_intraday():
    # gap 0.25; vol 1.5M > 200k floor; relvol = 1.5M/(0.5M*0.5385) = 5.57 -> >= 2x; quote fresh.
    quotes = {"AAA": _q("AAA", 5.0, 4.0, 1_500_000, 5.1, 500_000)}
    hits = movers_scan.tier1_filter(quotes, NOW_MID, premarket=False)
    assert len(hits) == 1 and hits[0].ticker == "AAA" and hits[0].gap_pct == 0.25
    assert hits[0].rel_volume is not None and abs(hits[0].rel_volume - 5.57) < 0.05   # formula pinned


def test_price_band_rejects_out_of_band():
    quotes = {"HI": _q("HI", 25.0, 20.0, 1_500_000, 25.1, 500_000),
              "LO": _q("LO", 0.5, 0.4, 1_500_000, 0.6, 500_000)}
    assert movers_scan.tier1_filter(quotes, NOW_MID, premarket=False) == []


def test_gap_below_min_rejected():
    quotes = {"X": _q("X", 4.1, 4.0, 1_500_000, 4.2, 500_000)}        # gap 0.025 < 0.05
    assert movers_scan.tier1_filter(quotes, NOW_MID, premarket=False) == []


def test_volume_floor_rejected_intraday():
    quotes = {"X": _q("X", 5.0, 4.0, 100_000, 5.1, 500_000)}         # 100k < 200k intraday floor
    assert movers_scan.tier1_filter(quotes, NOW_MID, premarket=False) == []


def test_known_low_relvol_rejected():
    # relvol = 300k/(5M*0.5385) = 0.11 -> below 2x and NOT None -> dropped.
    quotes = {"X": _q("X", 5.0, 4.0, 300_000, 5.1, 5_000_000)}
    assert movers_scan.tier1_filter(quotes, NOW_MID, premarket=False) == []


def test_relvol_just_below_and_above_intraday_floor():
    # floor 2.0. just below: relvol=1.9 -> 2.0*0.5385*0.95... pick vol so relvol ~1.9 vs ~2.1.
    below = {"B": _q("B", 5.0, 4.0, int(2.0 * 500_000 * 0.5385 * 0.95), 5.1, 500_000)}  # ~1.9x
    above = {"A": _q("A", 5.0, 4.0, int(2.0 * 500_000 * 0.5385 * 1.05), 5.1, 500_000)}  # ~2.1x
    assert movers_scan.tier1_filter(below, NOW_MID, premarket=False) == []
    assert len(movers_scan.tier1_filter(above, NOW_MID, premarket=False)) == 1


def test_none_relvol_kept_raw_fallback():
    # avg_daily_volume None -> relvol_proxy returns None -> candidate KEPT on raw volume+gap.
    quotes = {"X": _q("X", 5.0, 4.0, 1_500_000, 5.1, None)}
    hits = movers_scan.tier1_filter(quotes, NOW_MID, premarket=False)
    assert len(hits) == 1 and hits[0].rel_volume is None


def test_opening_guard_keeps_on_raw_fallback():
    # 09:31 ET is inside the 2-min opening guard -> relvol_proxy None even with a real avg -> kept raw.
    now = datetime(2026, 6, 29, 9, 31, tzinfo=ET)
    quotes = {"X": _q("X", 5.0, 4.0, 1_500_000, 5.1, 500_000, qt=now.astimezone(timezone.utc))}
    hits = movers_scan.tier1_filter(quotes, now, premarket=False)
    assert len(hits) == 1 and hits[0].rel_volume is None


def test_premarket_relvol_floor():
    # premarket floor 5.0. window 04:00-09:30 (19800s); 08:30 elapsed 16200s -> frac 0.818.
    rej = {"R": _q("R", 5.0, 4.0, 500_000, 5.1, 200_000, qt=NOW_PM_UTC)}    # ~3.06x rejected
    acc = {"A": _q("A", 5.0, 4.0, 1_000_000, 5.1, 200_000, qt=NOW_PM_UTC)}  # ~6.11x accepted
    assert movers_scan.tier1_filter(rej, NOW_PM, premarket=True) == []
    assert len(movers_scan.tier1_filter(acc, NOW_PM, premarket=True)) == 1


def test_premarket_missing_last_skipped():
    quotes = {"X": QuoteSnapshot("X", None, 4.0, None, 5.1, 500_000, NOW_PM_UTC, "extended")}
    assert movers_scan.tier1_filter(quotes, NOW_PM, premarket=True) == []


def test_stale_quote_skipped():
    stale = NOW_MID_UTC - timedelta(seconds=120)           # > QUOTE_MAX_AGE_SEC (30)
    quotes = {"X": _q("X", 5.0, 4.0, 1_500_000, 5.1, 500_000, qt=stale)}
    assert movers_scan.tier1_filter(quotes, NOW_MID, premarket=False) == []


def test_missing_quote_time_skipped():
    quotes = {"X": _q("X", 5.0, 4.0, 1_500_000, 5.1, 500_000, qt=None)}
    assert movers_scan.tier1_filter(quotes, NOW_MID, premarket=False) == []


def test_per_symbol_failure_isolated():
    # A quote whose attribute access RAISES must not abort the batch.
    class Boom:
        ticker = "BAD"
        quote_time = NOW_MID_UTC
        prev_close = 4.0
        total_volume = 1_500_000
        day_high = 5.1
        avg_daily_volume = 500_000
        source = "regular"
        @property
        def last(self):
            raise RuntimeError("boom")
    good = _q("GOOD", 5.0, 4.0, 1_500_000, 5.1, 500_000)
    hits = movers_scan.tier1_filter({"BAD": Boom(), "GOOD": good}, NOW_MID, premarket=False)
    assert [h.ticker for h in hits] == ["GOOD"]

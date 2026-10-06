"""tier1_filter/tier2_enrich/discover_movers populate an optional stats dict
(premarket-skip + cycle visibility). With stats=None behavior is unchanged (the tier1/tier2 suites cover it)."""
from datetime import datetime, timezone
from types import SimpleNamespace

from momentum_scanner.movers_scan import tier1_filter


def _q(**kw):
    base = dict(last=5.0, prev_close=4.0, total_volume=500_000, day_high=5.0,
                avg_daily_volume=50_000, quote_time=None, source="regular")
    base.update(kw)
    return SimpleNamespace(**base)


def test_tier1_stats_counts_stale_quote_skips():
    now = datetime(2026, 6, 30, 13, 0, tzinfo=timezone.utc)
    quotes = {"STALE": _q(quote_time=None), "GOOD": _q(quote_time=now)}
    stats = {}
    hits = tier1_filter(quotes, now, premarket=True, stats=stats)
    assert stats["quotes_seen"] == 2
    assert stats["skip_stale_quote"] == 1
    assert stats["tier1_hits"] == len(hits)
    assert "skip_price_band" in stats and "skip_low_relvol" in stats


def test_tier1_stats_none_is_noop_and_returns_same_hits():
    now = datetime(2026, 6, 30, 13, 0, tzinfo=timezone.utc)
    assert len(tier1_filter({"GOOD": _q(quote_time=now)}, now, premarket=True, stats=None)) == 1

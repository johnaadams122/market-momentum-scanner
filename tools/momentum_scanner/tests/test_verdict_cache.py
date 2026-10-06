from datetime import datetime, timezone, timedelta
from dataclasses import replace
from zoneinfo import ZoneInfo
from momentum_scanner.verdict_cache import VerdictCache
from momentum_scanner.catalyst_evaluator import CatalystVerdict

ET = ZoneInfo("America/New_York")
NOW = datetime(2026, 6, 29, 10, 0, tzinfo=ET)            # intraday (TTL 15 min)
PROCEED = CatalystVerdict("PROCEED", "real", "FDA", 0.9, "llm")


def _judged(now_et, v=PROCEED):
    return replace(v, judged_at=now_et.astimezone(timezone.utc))


def test_catalystverdict_new_fields_default_none():
    v = CatalystVerdict("REJECT", "no-data", "x", None, "rule")
    assert v.judged_at is None and v.headline is None


def test_put_then_get_fresh_returns_verdict():
    c = VerdictCache()
    c.put("ABCD", NOW, _judged(NOW))
    got = c.get_fresh("ABCD", NOW + timedelta(minutes=5), premarket=False)
    assert got is not None and got.decision == "PROCEED"


def test_get_fresh_missing_returns_none():
    assert VerdictCache().get_fresh("ZZZZ", NOW, premarket=False) is None


def test_stale_intraday_verdict_returns_none():
    c = VerdictCache()
    c.put("ABCD", NOW, _judged(NOW))
    assert c.get_fresh("ABCD", NOW + timedelta(minutes=16), premarket=False) is None   # > 15 TTL


def test_premarket_ttl_longer_than_intraday():
    pm = datetime(2026, 6, 29, 8, 0, tzinfo=ET)
    c = VerdictCache()
    c.put("ABCD", pm, _judged(pm))
    later = pm + timedelta(minutes=20)
    assert c.get_fresh("ABCD", later, premarket=True) is not None      # 20 < 30 premarket TTL
    assert c.get_fresh("ABCD", later, premarket=False) is None         # 20 > 15 intraday TTL (routing proof)


def test_date_scoping_is_real_not_ttl_artifact():
    # A verdict judged on the NEXT day must be found on that day and missed on NOW -- a date-less key
    # (only TTL) would FAIL the "found next day" assertion, so this distinguishes the two.
    c = VerdictCache()
    nd = NOW + timedelta(days=1)
    c.put("ABCD", nd, _judged(nd))
    assert c.get_fresh("ABCD", nd + timedelta(minutes=5), premarket=False) is not None
    assert c.get_fresh("ABCD", NOW, premarket=False) is None


def test_date_scoping_across_midnight_boundary():
    # put at 23:58 ET day 1; query at 00:03 ET day 2 (only 5 min later -> within the 15-min intraday
    # TTL). A TTL-ONLY cache would HIT; the date-scoped key MISSES (different ET trading date).
    d1_late = datetime(2026, 6, 29, 23, 58, tzinfo=ET)
    d2_early = datetime(2026, 6, 30, 0, 3, tzinfo=ET)
    c = VerdictCache()
    c.put("ABCD", d1_late, _judged(d1_late))
    assert c.get_fresh("ABCD", d2_early, premarket=False) is None      # date key differs -> miss
    # sanity: still found within its own date + TTL
    assert c.get_fresh("ABCD", d1_late + timedelta(minutes=1), premarket=False) is not None


def test_key_normalizes_ticker():
    c = VerdictCache()
    c.put("brk.b", NOW, _judged(NOW))
    assert c.get_fresh("BRK-B", NOW, premarket=False) is not None


def test_naive_clock_or_judged_at_fail_closed():
    c = VerdictCache()
    c.put("ABCD", NOW, _judged(NOW))
    naive = datetime(2026, 6, 29, 10, 5)                               # no tzinfo
    assert c.get_fresh("ABCD", naive, premarket=False) is None        # naive now_et -> None
    c2 = VerdictCache()
    c2.put("X", NOW, replace(PROCEED, judged_at=datetime(2026, 6, 29, 14, 0)))   # naive judged_at
    assert c2.get_fresh("X", NOW + timedelta(minutes=1), premarket=False) is None


def test_future_judged_at_returns_none():
    c = VerdictCache()
    future = NOW + timedelta(minutes=10)
    c.put("ABCD", NOW, _judged(future))                               # judged_at in the future (skew)
    assert c.get_fresh("ABCD", NOW, premarket=False) is None


def test_get_max_item_ts():
    c = VerdictCache()
    ts = NOW.astimezone(timezone.utc)
    c.put("ABCD", NOW, _judged(NOW), max_item_ts=ts)
    assert c.get_max_item_ts("ABCD", NOW) == ts

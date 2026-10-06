from datetime import datetime, timezone, timedelta
from momentum_scanner import fundamentals


def test_refresh_days_forces_daily_refetch():
    calls = {"n": 0}
    def fetch(t):
        calls["n"] += 1
        return 4_000_000.0
    cache = {}
    now = datetime(2026, 6, 29, tzinfo=timezone.utc)
    fundamentals.get_float("X", cache=cache, fetch_fn=fetch, now_utc=now, refresh_days=1)
    # 2 days later with refresh_days=1 -> stale -> refetch (weekly default would NOT refetch).
    fundamentals.get_float("X", cache=cache, fetch_fn=fetch, now_utc=now + timedelta(days=2), refresh_days=1)
    assert calls["n"] == 2


def test_default_refresh_days_is_backward_compatible_weekly():
    calls = {"n": 0}
    def fetch(t):
        calls["n"] += 1
        return 4_000_000.0
    cache = {}
    now = datetime(2026, 6, 29, tzinfo=timezone.utc)
    fundamentals.get_float("X", cache=cache, fetch_fn=fetch, now_utc=now)                       # weekly
    fundamentals.get_float("X", cache=cache, fetch_fn=fetch, now_utc=now + timedelta(days=2))   # within week
    assert calls["n"] == 1

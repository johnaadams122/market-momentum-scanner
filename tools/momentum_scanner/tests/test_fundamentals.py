from datetime import datetime, timezone, timedelta
from momentum_scanner import fundamentals

def test_get_float_caches_then_serves():
    cache, calls = {}, {"n": 0}
    def fetch_fn(t):
        calls["n"] += 1
        return 4_000_000.0
    now = datetime(2026, 6, 29, tzinfo=timezone.utc)
    assert fundamentals.get_float("ABCD", cache=cache, fetch_fn=fetch_fn, now_utc=now) == 4_000_000.0
    fundamentals.get_float("ABCD", cache=cache, fetch_fn=fetch_fn, now_utc=now + timedelta(days=1))
    assert calls["n"] == 1

def test_get_float_missing_or_bad_returns_none():
    now = datetime(2026, 6, 29, tzinfo=timezone.utc)
    assert fundamentals.get_float("Z", cache={}, fetch_fn=lambda t: None, now_utc=now) is None
    assert fundamentals.get_float("Z", cache={}, fetch_fn=lambda t: float("inf"), now_utc=now) is None
    assert fundamentals.get_float("Z", cache={}, fetch_fn=lambda t: 0, now_utc=now) is None

def test_get_float_fetch_error_returns_none():
    def boom(t):
        raise RuntimeError("yf down")
    now = datetime(2026, 6, 29, tzinfo=timezone.utc)
    assert fundamentals.get_float("Z", cache={}, fetch_fn=boom, now_utc=now) is None


def test_get_float_transient_error_not_cached_retries():
    """A fetch EXCEPTION must NOT be cached (transient blip would freeze float for a week)."""
    cache, calls = {}, {"n": 0}
    def boom(t):
        calls["n"] += 1
        raise RuntimeError("transient yf error")
    now = datetime(2026, 6, 29, tzinfo=timezone.utc)
    # First call: error -> None, NOT written to cache
    result1 = fundamentals.get_float("Z", cache=cache, fetch_fn=boom, now_utc=now)
    assert result1 is None
    assert "Z" not in cache, "transient error must NOT populate the cache"
    # Second call: must re-invoke fetch_fn (no cache entry to serve)
    result2 = fundamentals.get_float("Z", cache=cache, fetch_fn=boom, now_utc=now)
    assert result2 is None
    assert calls["n"] == 2, f"expected 2 fetch attempts, got {calls['n']}"

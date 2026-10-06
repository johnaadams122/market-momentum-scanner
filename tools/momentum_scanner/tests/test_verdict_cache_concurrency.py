"""VerdictCache must be safe under a background judge worker thread.
The spy-lock test proves the lock is actually ENTERED by each
method (a bare unused _lock attribute would fail it); the thread test is a GIL
smoke test for the single-writer/single-reader model."""
import threading
from datetime import datetime, timezone
from types import SimpleNamespace

from momentum_scanner.verdict_cache import VerdictCache


def _verdict(judged_at):
    return SimpleNamespace(decision="PROCEED", judged_at=judged_at)


class _SpyLock:
    """Wraps a real lock and counts context-manager entries."""
    def __init__(self):
        self._lock = threading.Lock()
        self.enters = 0

    def __enter__(self):
        self.enters += 1
        return self._lock.__enter__()

    def __exit__(self, *a):
        return self._lock.__exit__(*a)

    def acquire(self, *a, **k):
        return self._lock.acquire(*a, **k)

    def release(self):
        return self._lock.release()


def test_each_method_enters_the_lock():
    cache = VerdictCache()
    assert hasattr(cache, "_lock"), "VerdictCache must hold a lock"
    spy = _SpyLock()
    cache._lock = spy
    now = datetime.now(timezone.utc)
    cache.put("AAA", now, _verdict(now), max_item_ts="2026-06-30T13:00:00+00:00")
    cache.get_fresh("AAA", now, premarket=True)
    cache.get_max_item_ts("AAA", now)
    assert spy.enters >= 3, f"put/get_fresh/get_max_item_ts must each enter the lock; saw {spy.enters}"


def test_stored_max_item_ts_round_trips_under_lock():
    cache = VerdictCache()
    now = datetime.now(timezone.utc)
    cache.put("BBB", now, _verdict(now), max_item_ts="2026-06-30T14:30:00+00:00")
    assert cache.get_max_item_ts("BBB", now) == "2026-06-30T14:30:00+00:00"


def test_concurrent_put_get_fresh_smoke():
    cache = VerdictCache()
    now = datetime.now(timezone.utc)
    tickers = [f"T{i}" for i in range(64)]
    errors = []
    start = threading.Barrier(len(tickers))

    def worker(t):
        try:
            start.wait()
            cache.put(t, now, _verdict(now))
            assert cache.get_fresh(t, now, premarket=True) is not None
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in tickers]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=30)
    assert errors == []
    for t in tickers:
        assert cache.get_fresh(t, now, premarket=True) is not None

"""Per-(ticker, date) catalyst verdict cache for the decoupled movers judge. A verdict is
'fresh' while its judged_at age is within the mode TTL (premarket 30 / intraday 15 min); missing,
stale, or a naive/future clock returns None so the judge batch (re-)judges it. The key normalizes the
ticker (BRK-B == brk.b) and is scoped to the ET trading date so verdicts never leak across days.
Fail-closed: a naive now_et or judged_at never silently miscomputes (host-local tz) -- it returns None.
Per-method locking is NOT compound-op atomic; JUDGE_CONCURRENCY must stay 1 until a compound put_if_newer exists."""
import threading
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from momentum_scanner import config
from momentum_scanner.edgar_rss import normalize_ticker

ET = ZoneInfo("America/New_York")


def _aware(dt):
    return isinstance(dt, datetime) and dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None


class VerdictCache:
    def __init__(self):
        self._d = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(ticker, now_et):
        return (normalize_ticker(ticker), now_et.astimezone(ET).date().isoformat())

    def get_fresh(self, ticker, now_et, *, premarket):
        if not _aware(now_et):
            return None                                  # fail-closed: naive clock
        with self._lock:
            entry = self._d.get(self._key(ticker, now_et))
        if not entry:
            return None
        verdict = entry["verdict"]
        judged_at = getattr(verdict, "judged_at", None)
        if not _aware(judged_at):
            return None                                  # fail-closed: naive/missing judged_at
        ttl_min = config.VERDICT_TTL_MIN_PREMARKET if premarket else config.VERDICT_TTL_MIN_INTRADAY
        age_min = (now_et.astimezone(timezone.utc) - judged_at).total_seconds() / 60.0
        if age_min < 0 or age_min > ttl_min:
            return None                                  # future (skew) or stale -> re-judge
        return verdict

    def put(self, ticker, now_et, verdict, *, max_item_ts=None):
        with self._lock:
            self._d[self._key(ticker, now_et)] = {"verdict": verdict, "max_item_ts": max_item_ts}

    def get_max_item_ts(self, ticker, now_et):
        with self._lock:
            entry = self._d.get(self._key(ticker, now_et))
        return entry["max_item_ts"] if entry else None

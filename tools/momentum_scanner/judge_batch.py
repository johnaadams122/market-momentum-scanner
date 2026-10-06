# judge_batch.py
"""Priority-ordered, bounded, fail-closed judge batch. Judges only candidates lacking a
fresh cached verdict, in descending ranking.score order, serial (Gemma JUDGE_CONCURRENCY=1). At most
MAX_JUDGE_PER_CYCLE are judged per cycle; the lower-priority remainder is LEFT UNJUDGED and the drop
count is LOGGED (no silent cap). A judge_fn returning (None, _) (systemic) is NOT cached;
a per-candidate exception is isolated. `edgar` is the per-cycle EdgarContext built by the
daemon's run_once; it is forwarded verbatim to every judge_fn call so a ticker resolved during pre-judge
reuses the same EDGAR cache instead of re-fetching (None when EDGAR_PREJUDGE_ENABLED is False)."""
import logging

from momentum_scanner import config
from momentum_scanner.ranking import score

_log = logging.getLogger(__name__)


def process_judge_batch(candidates, cache, *, now_et, premarket, judge_fn, max_items=None, edgar=None):
    cap = config.MAX_JUDGE_PER_CYCLE if max_items is None else max_items
    stats = {"judged": 0, "cached_hits": 0, "dropped": 0, "failed": 0}
    pending = []
    for c in candidates:
        if cache.get_fresh(c.ticker, now_et, premarket=premarket) is not None:
            stats["cached_hits"] += 1
        else:
            pending.append(c)
    pending.sort(key=score, reverse=True)
    to_judge, dropped = pending[:cap], pending[cap:]
    if dropped:
        stats["dropped"] = len(dropped)
        _log.warning("judge batch cap %d: leaving %d lower-priority ticker(s) unjudged this cycle: %s",
                     cap, len(dropped), [c.ticker for c in dropped])
    for c in to_judge:
        try:
            verdict, max_item_ts = judge_fn(c, now_et=now_et, premarket=premarket, edgar=edgar)
        except Exception as exc:
            _log.debug("judge batch: judge_fn raised for %s: %s", c.ticker, exc)
            stats["failed"] += 1
            continue
        if verdict is None:
            stats["failed"] += 1                          # systemic -> unjudged (not cached)
            continue
        cache.put(c.ticker, now_et, verdict, max_item_ts=max_item_ts)
        stats["judged"] += 1
    return stats

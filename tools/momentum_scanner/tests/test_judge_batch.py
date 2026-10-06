from datetime import datetime, timezone
from dataclasses import dataclass
from zoneinfo import ZoneInfo
from momentum_scanner import judge_batch, config
from momentum_scanner.verdict_cache import VerdictCache
from momentum_scanner.catalyst_evaluator import CatalystVerdict

ET = ZoneInfo("America/New_York")
NOW = datetime(2026, 6, 29, 10, 0, tzinfo=ET)


@dataclass
class _Cand:
    ticker: str
    gap_pct: float = 0.20
    rel_volume: float = 5.0
    float_shares: float = 4_000_000


def _v(decision="PROCEED"):
    return CatalystVerdict(decision, "real", "r", 0.8, "llm", judged_at=NOW.astimezone(timezone.utc))


def test_judges_uncached_and_caches():
    cache = VerdictCache()
    seen = []
    def jf(c, *, now_et, premarket, edgar=None):
        seen.append((c.ticker, premarket)); return _v(), None
    stats = judge_batch.process_judge_batch([_Cand("AAA")], cache, now_et=NOW, premarket=False, judge_fn=jf)
    assert stats["judged"] == 1 and seen == [("AAA", False)]
    assert cache.get_fresh("AAA", NOW, premarket=False) is not None


def test_cached_fresh_not_rejudged():
    cache = VerdictCache(); cache.put("AAA", NOW, _v())
    calls = {"n": 0}
    def jf(c, *, now_et, premarket, edgar=None):
        calls["n"] += 1; return _v(), None
    stats = judge_batch.process_judge_batch([_Cand("AAA")], cache, now_et=NOW, premarket=False, judge_fn=jf)
    assert calls["n"] == 0 and stats["cached_hits"] == 1 and stats["judged"] == 0


def test_priority_order_and_cap_drops_lowest():
    cache = VerdictCache(); order = []
    def jf(c, *, now_et, premarket, edgar=None):
        order.append(c.ticker); return _v(), None
    cands = [_Cand("LO", gap_pct=0.10), _Cand("HI", gap_pct=0.90), _Cand("MID", gap_pct=0.50)]
    stats = judge_batch.process_judge_batch(cands, cache, now_et=NOW, premarket=False, judge_fn=jf, max_items=2)
    assert order == ["HI", "MID"] and stats["judged"] == 2 and stats["dropped"] == 1
    assert cache.get_fresh("LO", NOW, premarket=False) is None


def test_reject_verdict_is_cached():
    cache = VerdictCache()
    def jf(c, *, now_et, premarket, edgar=None):
        return _v("REJECT"), None
    stats = judge_batch.process_judge_batch([_Cand("AAA")], cache, now_et=NOW, premarket=False, judge_fn=jf)
    assert stats["judged"] == 1 and cache.get_fresh("AAA", NOW, premarket=False).decision == "REJECT"


def test_systemic_none_not_cached():
    cache = VerdictCache()
    def jf(c, *, now_et, premarket, edgar=None):
        return None, None
    stats = judge_batch.process_judge_batch([_Cand("AAA")], cache, now_et=NOW, premarket=False, judge_fn=jf)
    assert stats["failed"] == 1 and cache.get_fresh("AAA", NOW, premarket=False) is None


def test_per_candidate_exception_isolated():
    cache = VerdictCache()
    def jf(c, *, now_et, premarket, edgar=None):
        if c.ticker == "BAD":
            raise RuntimeError("judge boom")
        return _v(), None
    stats = judge_batch.process_judge_batch([_Cand("BAD", gap_pct=0.9), _Cand("OK", gap_pct=0.1)],
                                            cache, now_et=NOW, premarket=False, judge_fn=jf)
    assert stats["failed"] == 1 and stats["judged"] == 1
    assert cache.get_fresh("OK", NOW, premarket=False) is not None
    assert cache.get_fresh("BAD", NOW, premarket=False) is None


def test_edgar_ctx_threaded_to_judge_fn():
    """process_judge_batch forwards its `edgar` kwarg verbatim to every judge_fn
    call, so the daemon's per-cycle EdgarContext rides through to judge_candidate/evaluate_news."""
    cache = VerdictCache()
    sentinel = object()
    seen = []
    def jf(c, *, now_et, premarket, edgar=None):
        seen.append(edgar); return _v(), None
    stats = judge_batch.process_judge_batch([_Cand("AAA")], cache, now_et=NOW, premarket=False,
                                            judge_fn=jf, edgar=sentinel)
    assert stats["judged"] == 1
    assert seen == [sentinel]


def test_edgar_defaults_to_none_when_not_passed():
    """Backward compat: callers that don't pass `edgar` (e.g. judge_runtime.make_judge_batch) still
    work -- judge_fn receives edgar=None."""
    cache = VerdictCache()
    seen = []
    def jf(c, *, now_et, premarket, edgar=None):
        seen.append(edgar); return _v(), None
    judge_batch.process_judge_batch([_Cand("AAA")], cache, now_et=NOW, premarket=False, judge_fn=jf)
    assert seen == [None]

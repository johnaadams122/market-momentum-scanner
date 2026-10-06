# judge_runtime.py
"""Wiring for the DECOUPLED movers judge -- the helpers the daemon binds per cycle.
Discovery (discover_movers / run_movers_scan) only READS the verdict cache via make_verdict_lookup; a
SEPARATE judge pass (make_judge_batch -> process_judge_batch) fills it, so the slow Gemma judge never
blocks the fast discovery cadence. The daemon runs ONE discovery per cycle and writes from the same
candidates via write_from_candidates. Gemma-only."""
from momentum_scanner.judge import judge_candidate
from momentum_scanner.judge_batch import process_judge_batch


def make_verdict_lookup(cache, *, now_et, premarket):
    def lookup(candidate):
        return cache.get_fresh(candidate.ticker, now_et, premarket=premarket)
    return lookup


def make_judge_batch(cache, news_client, *, now_et, premarket, judge_fn=None):
    if judge_fn is None:
        def judge_fn(candidate, *, now_et, premarket, edgar=None):    # noqa: F811 -- default real judge
            return judge_candidate(candidate, news_client=news_client, now_et=now_et, premarket=premarket,
                                   edgar=edgar)

    def run_batch(candidates):
        return process_judge_batch(candidates, cache, now_et=now_et, premarket=premarket,
                                   judge_fn=judge_fn)
    return run_batch

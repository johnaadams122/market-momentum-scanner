"""Discovery daemon. run_once is driven with injected deps; the worker thread, the
dead-worker path, the rate-backoff/limiter-absent skips, the self-sufficient starvation status:error,
the outage/empty-universe status:error, the coalescing queue, the per-cycle crash path, and the
running two-cycle async seam are all exercised offline + deterministically."""
import json
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from momentum_scanner import scanner_movers_daemon as d
from momentum_scanner import movers_scan, config, judge, shadow_judge
from momentum_scanner.verdict_cache import VerdictCache
from momentum_scanner.errors import ScannerNetworkError

ET = ZoneInfo("America/New_York")


class _Verdict:
    def __init__(self):
        self.decision, self.label, self.reason = "PROCEED", "real", "test catalyst"
        self.confidence, self.source, self.headline = 0.9, "rule", "Test Co beats and raises"
        self.judged_at = datetime.now(timezone.utc)


def _cand(ticker="AAA"):
    return movers_scan.MoverCandidate(
        ticker=ticker, company_name=ticker, premarket_price=5.0, prev_close=4.0, gap_pct=0.25,
        float_shares=5_000_000, premarket_volume=500_000, baseline_volume=50_000, rel_volume=10.0,
        day_high=5.0, new_hod=True, scan_mode="intraday", last_seen_at=None)


def _judge_fn_proceed(candidate, *, now_et, premarket, edgar=None):
    v = _Verdict(); v.judged_at = now_et.astimezone(timezone.utc)
    return v, None


class _FakeLimiter:
    def __init__(self, grant=True):
        self.grant, self.calls = grant, []
    def acquire(self, n=1, tier="discovery", now=None):
        self.calls.append((n, tier)); return self.grant


def _sync_enqueue(cache, judge_fn):
    from momentum_scanner.judge_batch import process_judge_batch
    def enqueue(batch):
        cands, now_et, premarket, edgar_ctx = batch
        process_judge_batch(cands, cache, now_et=now_et, premarket=premarket, judge_fn=judge_fn,
                            edgar=edgar_ctx)
    return enqueue


def _now():
    return datetime(2026, 6, 30, 14, 0, tzinfo=timezone.utc)        # Tue 10:00 ET intraday


@pytest.fixture(autouse=True)
def _fail_closed_env(monkeypatch):
    monkeypatch.setattr(config, "DAEMON_ALLOW_UNGATED", False)      # never depend on ambient env


def _kw(tmp_path, **over):
    cache = over.pop("cache", VerdictCache())
    base = dict(
        symbols=["AAA", "BBB"], token_fn=lambda: "tok", session=object(),
        limiter=_FakeLimiter(True), cache=cache, enqueue_fn=_sync_enqueue(cache, _judge_fn_proceed),
        float_cache={}, float_fetch_fn=lambda *a, **k: None, now_utc=_now(), now_et=_now().astimezone(ET),
        mode="intraday", out_path=str(tmp_path / "watchlist.json"), heartbeat_path=str(tmp_path / "hb.json"),
        started_at_iso=_now().isoformat(), pid=4321, worker_alive=True, queue_depth=0, drops=0, rate_backoff=0)
    base.update(over)
    return base


def _wl(tmp_path):
    return json.loads((tmp_path / "watchlist.json").read_text())


def _hb(tmp_path):
    return json.loads((tmp_path / "hb.json").read_text())


def test_closed_mode_no_scan(tmp_path, monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(movers_scan, "discover_movers", lambda *a, **k: called.__setitem__("n", 1) or [])
    res = d.run_once(**_kw(tmp_path, mode="closed"))
    assert res["status"] == "closed" and called["n"] == 0 and _hb(tmp_path)["last_status"] == "closed"


def test_dead_worker_writes_status_error_even_when_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(movers_scan, "discover_movers", lambda *a, **k: [_cand()])
    res = d.run_once(**_kw(tmp_path, worker_alive=False, mode="closed"))
    assert res["status"] == "error"                                # worker-alive checked BEFORE closed
    assert _wl(tmp_path)["status"] == "error" and _hb(tmp_path)["last_status"] == "judge_dead"


def test_rate_backoff_skips_scan_and_leaves_prior_watchlist(tmp_path, monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(movers_scan, "discover_movers", lambda *a, **k: called.__setitem__("n", 1) or [_cand()])
    res = d.run_once(**_kw(tmp_path, limiter=_FakeLimiter(False)))
    assert res["status"] == "rate_backoff" and called["n"] == 0 and res["rate_backoff"] == 1
    assert _hb(tmp_path)["last_status"] == "rate_backoff"
    assert not (tmp_path / "watchlist.json").exists()              # within window -> no write (stale lapses)


def test_sustained_backoff_writes_status_error(tmp_path, monkeypatch):
    monkeypatch.setattr(movers_scan, "discover_movers", lambda *a, **k: [_cand()])
    n = config.WATCHLIST_MAX_AGE_SEC // config.DISCOVERY_INTERVAL_SEC      # backoff outlasts the window
    res = d.run_once(**_kw(tmp_path, limiter=_FakeLimiter(False), rate_backoff=n - 1))
    assert res["status"] == "rate_backoff" and res["rate_backoff"] == n
    assert _wl(tmp_path)["status"] == "error"                     # self-sufficient: proactive fail-closed
    assert _hb(tmp_path)["last_status"] == "rate_backoff"


def test_limiter_absent_fails_closed(tmp_path, monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(movers_scan, "discover_movers", lambda *a, **k: called.__setitem__("n", 1) or [_cand()])
    res = d.run_once(**_kw(tmp_path, limiter=None))
    assert res["status"] == "limiter_absent" and called["n"] == 0
    assert _hb(tmp_path)["last_status"] == "limiter_absent"


def test_limiter_absent_allows_with_opt_in(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DAEMON_ALLOW_UNGATED", True)
    monkeypatch.setattr(movers_scan, "discover_movers", lambda *a, **k: [_cand("AAA")])
    res = d.run_once(**_kw(tmp_path, limiter=None))
    assert res["status"] == "ok" and _wl(tmp_path)["status"] == "ok"


def test_empty_universe_status_error(tmp_path, monkeypatch):
    monkeypatch.setattr(movers_scan, "discover_movers", lambda *a, **k: [_cand()])
    res = d.run_once(**_kw(tmp_path, symbols=[]))
    assert res["status"] == "error" and _wl(tmp_path)["status"] == "error"


def test_outage_writes_status_error(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise ScannerNetworkError("schwab down")
    monkeypatch.setattr(movers_scan, "discover_movers", _boom)
    res = d.run_once(**_kw(tmp_path))
    assert res["status"] == "error" and _wl(tmp_path)["status"] == "error"
    assert _hb(tmp_path)["last_status"] == "error"


def test_ok_writes_judged_watchlist_with_drops_in_heartbeat(tmp_path, monkeypatch):
    monkeypatch.setattr(movers_scan, "discover_movers", lambda *a, **k: [_cand("AAA")])
    res = d.run_once(**_kw(tmp_path, drops=2))
    assert res["status"] == "ok"
    wl = _wl(tmp_path)
    assert wl["status"] == "ok" and wl["schema_version"] == 2
    assert [c["ticker"] for c in wl["candidates"]] == ["AAA"] and wl["candidates"][0]["gate_decision"] == "PROCEED"
    hb = _hb(tmp_path)
    assert hb["last_status"] == "ok" and hb["drops"] == 2 and hb["rate_backoff"] == 0


def test_heartbeat_carries_judge_systemic_failures(tmp_path, monkeypatch):
    # The counter exists so a dead news feed can be told apart from a quiet tape. It is
    # only useful if the watchdog/operator can SEE it, so it rides the heartbeat next to drops.
    from momentum_scanner import judge
    judge.reset_systemic_failures()
    monkeypatch.setattr(movers_scan, "discover_movers", lambda *a, **k: [_cand("AAA")])
    assert _hb_after(tmp_path, monkeypatch)["judge_systemic_failures"] == 0
    monkeypatch.setattr(judge, "_systemic_failures", 4)
    assert _hb_after(tmp_path, monkeypatch)["judge_systemic_failures"] == 4


def _hb_after(tmp_path, monkeypatch):
    d.run_once(**_kw(tmp_path))
    return _hb(tmp_path)


def test_first_sighting_absent_with_async_noop_enqueue(tmp_path, monkeypatch):
    monkeypatch.setattr(movers_scan, "discover_movers", lambda *a, **k: [_cand("AAA")])
    res = d.run_once(**_kw(tmp_path, enqueue_fn=lambda batch: None))
    assert res["status"] == "ok" and _wl(tmp_path)["candidates"] == []


def test_run_once_builds_and_forwards_edgar_context_when_enabled(tmp_path, monkeypatch):
    """run_once must build an EdgarContext per cycle and forward it to
    discover_movers as a non-None `edgar` kwarg when config.EDGAR_PREJUDGE_ENABLED is True. The
    judge-side threading of this same context is covered separately below."""
    monkeypatch.setattr(config, "EDGAR_PREJUDGE_ENABLED", True)
    captured = {}

    def _spy(*a, **kw):
        captured.update(kw)
        return [_cand("AAA")]

    monkeypatch.setattr(movers_scan, "discover_movers", _spy)
    res = d.run_once(**_kw(tmp_path))
    assert res["status"] == "ok"
    assert captured.get("edgar") is not None
    assert isinstance(captured["edgar"], d.EdgarContext)


def test_run_once_edgar_context_none_when_prejudge_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EDGAR_PREJUDGE_ENABLED", False)
    captured = {}

    def _spy(*a, **kw):
        captured.update(kw)
        return [_cand("AAA")]

    monkeypatch.setattr(movers_scan, "discover_movers", _spy)
    res = d.run_once(**_kw(tmp_path))
    assert res["status"] == "ok"
    assert captured.get("edgar") is None


def test_run_once_threads_same_edgar_ctx_from_discovery_to_judge_fn(tmp_path, monkeypatch):
    """End-to-end: the SAME EdgarContext instance run_once builds and hands to
    discover_movers also reaches the judge_fn via enqueue_fn -> process_judge_batch, proving the
    per-cycle shared-cache dedup (a ticker pre-judged and judged in one cycle hits ONE EdgarContext,
    not two)."""
    monkeypatch.setattr(config, "EDGAR_PREJUDGE_ENABLED", True)
    discover_captured = {}

    def _spy_discover(*a, **kw):
        discover_captured.update(kw)
        return [_cand("AAA")]

    monkeypatch.setattr(movers_scan, "discover_movers", _spy_discover)

    judge_captured = {}
    def jf(c, *, now_et, premarket, edgar=None):
        judge_captured["edgar"] = edgar
        v = _Verdict(); v.judged_at = now_et.astimezone(timezone.utc)
        return v, None

    cache = VerdictCache()
    res = d.run_once(**_kw(tmp_path, cache=cache, enqueue_fn=_sync_enqueue(cache, jf)))
    assert res["status"] == "ok"
    assert discover_captured.get("edgar") is not None
    assert judge_captured.get("edgar") is not None
    assert judge_captured["edgar"] is discover_captured["edgar"]      # identical shared ctx, no 2nd build


def test_judge_worker_judges_with_travelling_clock_both_windows(tmp_path):
    cache = VerdictCache()
    seen = []
    def jf(c, *, now_et, premarket, edgar=None):
        seen.append(premarket)
        v = _Verdict(); v.judged_at = now_et.astimezone(timezone.utc)
        return v, None
    w = d.JudgeWorker(cache, jf); w.start()
    try:
        now_et = _now().astimezone(ET)
        # Use distinct tickers per window: VerdictCache keys (ticker, date) not premarket, so the same
        # ticker judged in pm=True would be found cached for pm=False and skip re-judging.
        for pm, ticker in zip((True, False), ("AAA", "BBB")):
            w.enqueue(([_cand(ticker)], now_et, pm, None))
            dl = time.time() + 3.0
            while cache.get_fresh(ticker, now_et, premarket=pm) is None and time.time() < dl:
                time.sleep(0.02)
            assert cache.get_fresh(ticker, now_et, premarket=pm) is not None
        assert True in seen and False in seen                      # the flag travels with each batch
        assert w.is_alive()
    finally:
        w.stop(join_timeout=2.0)
    assert not w.is_alive()


def test_judge_worker_coalesces_to_latest(tmp_path):
    cache = VerdictCache()
    entered = threading.Event(); gate = threading.Event(); judged = []
    def jf(c, *, now_et, premarket, edgar=None):
        entered.set(); gate.wait(2.0); judged.append(c.ticker)
        v = _Verdict(); v.judged_at = now_et.astimezone(timezone.utc)
        return v, None
    w = d.JudgeWorker(cache, jf); w.start()
    try:
        now_et = _now().astimezone(ET)
        w.enqueue(([_cand("OLD")], now_et, False, None))
        assert entered.wait(2.0)                                   # deterministic: worker has OLD, parked in jf
        w.enqueue(([_cand("MID")], now_et, False, None))
        w.enqueue(([_cand("NEW")], now_et, False, None))
        assert w.queue_depth() == 1 and w.drops() >= 1
        gate.set()
        dl = time.time() + 3.0
        while cache.get_fresh("NEW", now_et, premarket=False) is None and time.time() < dl:
            time.sleep(0.02)
        assert "NEW" in judged and "MID" not in judged
    finally:
        w.stop(join_timeout=2.0)


def test_worker_isolates_normal_exception_and_stays_alive(tmp_path):
    cache = VerdictCache(); calls = []
    def jf(c, *, now_et, premarket, edgar=None):
        calls.append(c.ticker)
        if c.ticker == "BAD":
            raise ValueError("transient judge error")
        v = _Verdict(); v.judged_at = now_et.astimezone(timezone.utc)
        return v, None
    w = d.JudgeWorker(cache, jf); w.start()
    try:
        now_et = _now().astimezone(ET)
        w.enqueue(([_cand("BAD")], now_et, False, None))
        dl = time.time() + 3.0
        while "BAD" not in calls and time.time() < dl:
            time.sleep(0.02)
        assert w.is_alive()                                        # a normal Exception does NOT kill it
        w.enqueue(([_cand("GOOD")], now_et, False, None))
        dl = time.time() + 3.0
        while cache.get_fresh("GOOD", now_et, premarket=False) is None and time.time() < dl:
            time.sleep(0.02)
        assert w.is_alive()
    finally:
        w.stop(join_timeout=2.0)


# The worker is killed on purpose; pytest's unhandled-thread-exception warning is expected here.
@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_dead_worker_is_detected(tmp_path):
    cache = VerdictCache()
    def jf(c, *, now_et, premarket, edgar=None):
        raise BaseException("worker died")                        # noqa: TRY002
    w = d.JudgeWorker(cache, jf); w.start()
    try:
        w.enqueue(([_cand("AAA")], _now().astimezone(ET), False, None))
        dl = time.time() + 3.0
        while w.is_alive() and time.time() < dl:
            time.sleep(0.02)
        assert not w.is_alive()
    finally:
        w.stop(join_timeout=2.0)


def test_judge_worker_threads_edgar_ctx_to_judge_fn(tmp_path):
    """End-to-end (JudgeWorker leg): the 4th batch element (edgar_ctx) reaches judge_fn's `edgar`
    kwarg via process_judge_batch."""
    cache = VerdictCache()
    sentinel = object()
    seen = []
    def jf(c, *, now_et, premarket, edgar=None):
        seen.append(edgar)
        v = _Verdict(); v.judged_at = now_et.astimezone(timezone.utc)
        return v, None
    w = d.JudgeWorker(cache, jf); w.start()
    try:
        now_et = _now().astimezone(ET)
        w.enqueue(([_cand("AAA")], now_et, False, sentinel))
        dl = time.time() + 3.0
        while cache.get_fresh("AAA", now_et, premarket=False) is None and time.time() < dl:
            time.sleep(0.02)
        assert cache.get_fresh("AAA", now_et, premarket=False) is not None
        assert seen == [sentinel]
    finally:
        w.stop(join_timeout=2.0)


def test_run_daemon_smoke_closed_then_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(movers_scan, "discover_movers", lambda *a, **k: [])
    clocks = iter([datetime(2026, 6, 30, 2, 0, tzinfo=timezone.utc)] * 4)   # 22:00 ET prior day -> closed
    d.run_daemon(symbols=["AAA"], token_fn=lambda: "t", session=object(), limiter=_FakeLimiter(True),
                 cache=VerdictCache(), judge_fn=_judge_fn_proceed, float_cache={}, float_fetch_fn=lambda *a, **k: None,
                 out_path=str(tmp_path / "wl.json"), heartbeat_path=str(tmp_path / "hb.json"),
                 lock_path=str(tmp_path / "daemon.lock"), clock=lambda: next(clocks), max_cycles=1)
    assert _hb(tmp_path)["last_status"] == "closed"
    with d.single_instance_lock(str(tmp_path / "daemon.lock")):    # lock released on teardown
        pass


def test_run_daemon_teardown_when_setup_raises(tmp_path):
    """The worker is constructed+started INSIDE the try; a clock raising AFTER start still tears the
    worker down and releases the lock (no leaked non-daemon thread, no orphaned lock)."""
    calls = {"n": 0}
    def clock():
        calls["n"] += 1
        if calls["n"] == 1:
            return datetime(2026, 6, 30, 2, 0, tzinfo=timezone.utc)   # started_at_iso
        raise RuntimeError("clock exhausted after start")
    with pytest.raises(RuntimeError):
        d.run_daemon(symbols=["AAA"], token_fn=lambda: "t", session=object(), limiter=_FakeLimiter(True),
                     cache=VerdictCache(), judge_fn=_judge_fn_proceed, float_cache={}, float_fetch_fn=lambda *a, **k: None,
                     out_path=str(tmp_path / "wl.json"), heartbeat_path=str(tmp_path / "hb.json"),
                     lock_path=str(tmp_path / "daemon.lock"), clock=clock, max_cycles=3)
    with d.single_instance_lock(str(tmp_path / "daemon.lock")):    # lock was released despite the raise
        pass


def test_run_daemon_cycle_crash_is_fail_closed(tmp_path, monkeypatch):
    """An unexpected raise inside a cycle -> status:error + last_status='crash' + the loop continues
    and tears down cleanly."""
    monkeypatch.setattr(d, "session_mode", lambda now_et: (_ for _ in ()).throw(RuntimeError("boom")))
    clocks = iter([datetime(2026, 6, 30, 14, 0, tzinfo=timezone.utc)] * 6)
    d.run_daemon(symbols=["AAA"], token_fn=lambda: "t", session=object(), limiter=_FakeLimiter(True),
                 cache=VerdictCache(), judge_fn=_judge_fn_proceed, float_cache={}, float_fetch_fn=lambda *a, **k: None,
                 out_path=str(tmp_path / "wl.json"), heartbeat_path=str(tmp_path / "hb.json"),
                 lock_path=str(tmp_path / "daemon.lock"), clock=lambda: next(clocks), max_cycles=1)
    assert json.loads((tmp_path / "wl.json").read_text())["status"] == "error"
    assert _hb(tmp_path)["last_status"] == "crash"


def test_run_daemon_async_seam_absent_then_present(tmp_path, monkeypatch):
    """End-to-end across two running daemon cycles: cycle 1 absent, cycle 2 present once the
    real background worker has judged the first-sighting ticker."""
    monkeypatch.setattr(movers_scan, "discover_movers", lambda *a, **k: [_cand("AAA")])
    monkeypatch.setattr(d, "session_mode", lambda now_et: "intraday")
    out = str(tmp_path / "wl.json")
    cache = VerdictCache()
    captured = []
    real_run_once = d.run_once
    def spy(**kw):
        r = real_run_once(**kw)
        captured.append(json.loads((tmp_path / "wl.json").read_text())["candidates"])
        return r
    monkeypatch.setattr(d, "run_once", spy)
    clocks = iter([datetime(2026, 6, 30, 14, 0, tzinfo=timezone.utc),
                   datetime(2026, 6, 30, 14, 0, 1, tzinfo=timezone.utc),
                   datetime(2026, 6, 30, 14, 0, 2, tzinfo=timezone.utc)])
    # A tiny sleep callback is injected as the clock to let the worker drain between cycles.
    def clock():
        time.sleep(0.15)
        return next(clocks)
    d.run_daemon(symbols=["AAA"], token_fn=lambda: "t", session=object(), limiter=_FakeLimiter(True),
                 cache=cache, judge_fn=_judge_fn_proceed, float_cache={}, float_fetch_fn=lambda *a, **k: None,
                 out_path=out, heartbeat_path=str(tmp_path / "hb.json"),
                 lock_path=str(tmp_path / "daemon.lock"), clock=clock, max_cycles=2)
    assert captured[0] == []                                       # cycle 1: first-sighting absent
    assert [c["ticker"] for c in captured[1]] == ["AAA"]           # cycle 2: judged by the worker -> present


def test_concurrent_verdict_cache_get_put_is_safe():
    cache = VerdictCache(); now_et = _now().astimezone(ET); errors = []
    def writer():
        try:
            for i in range(200):
                v = _Verdict(); v.judged_at = now_et.astimezone(timezone.utc)
                cache.put(f"T{i % 5}", now_et, v)
        except Exception as exc:                                   # pragma: no cover
            errors.append(exc)
    def reader():
        try:
            for i in range(200):
                cache.get_fresh(f"T{i % 5}", now_et, premarket=False)
        except Exception as exc:                                   # pragma: no cover
            errors.append(exc)
    threads = [threading.Thread(target=writer) for _ in range(2)] + [threading.Thread(target=reader) for _ in range(2)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert not errors


def test_accumulate_funnel_counters(tmp_path):
    from datetime import datetime as _dt
    p = tmp_path / "discovery_funnel.json"
    now_et = _dt(2026, 7, 6, 10, 0, tzinfo=ET)
    d._accumulate_funnel(
        p, {"status": "ok", "stats": {"quotes_seen": 40, "tier1_hits": 3}}, now_et)
    d._accumulate_funnel(p, {"status": "rate_backoff", "stats": {}}, now_et)
    data = json.loads(p.read_text())
    day = data["2026-07-06"]
    assert day["cycles"] == 2
    assert day["by_status"] == {"ok": 1, "rate_backoff": 1}
    assert day["stats"]["quotes_seen"] == 40 and day["stats"]["tier1_hits"] == 3


def test_run_daemon_wires_funnel_accumulation():
    import inspect
    src = inspect.getsource(d.run_daemon)
    assert "_accumulate_funnel(" in src


def test_build_real_judge_resolves_shadow_once_and_reuses_callable(monkeypatch):
    """Catches recreating a shadow runtime for every candidate in a daemon."""
    calls = {"resolve": 0, "judge": []}
    deciding = object()

    def resolve():
        calls["resolve"] += 1
        return SimpleNamespace(
            callable=deciding,
            status="active",
            metrics_fn=lambda: {"shadow_status": "active"},
            close_fn=lambda: None,
        )

    def candidate(candidate, **kwargs):
        calls["judge"].append(kwargs["news_judge"])
        return None, None

    monkeypatch.setattr(judge, "resolve_news_judge_runtime", resolve)
    monkeypatch.setattr(judge, "judge_candidate", candidate)

    fn = d._build_real_judge_fn(object())
    assert fn.shadow_status == "active"
    fn(_cand("AAA"), now_et=_now(), premarket=False)
    fn(_cand("BBB"), now_et=_now(), premarket=False)

    assert calls == {"resolve": 1, "judge": [deciding, deciding]}


def test_shadow_metrics_exception_degrades_both_lanes():
    # A total metrics_fn() failure means BOTH the haiku and qwen3 lanes are unreadable,
    # not just the haiku one -- without the qwen3_* keys here, _shadow_metrics_projection
    # would fall back to _DEFAULT_QWEN3_SHADOW_METRICS ("off"), which reads as
    # "deliberately disabled" rather than "broken".
    def boom():
        raise RuntimeError("metrics collection failed")

    fn = d.DaemonNewsJudge(object(), SimpleNamespace(
        callable=lambda h, s: None, status="active", metrics_fn=boom, close_fn=lambda: None))
    metrics = fn.shadow_metrics()
    assert metrics["shadow_status"] == "degraded"
    assert metrics["qwen3_shadow_status"] == "degraded"
    assert metrics["shadow_last_error_type"] == "RuntimeError"
    assert metrics["qwen3_shadow_last_error_type"] == "RuntimeError"


def test_primary_cache_publication_precedes_blocked_shadow_release(tmp_path):
    """Catches a shadow call that delays putting the primary verdict in the cache."""
    entered = threading.Event()
    release = threading.Event()
    cache = VerdictCache()
    now = _now().astimezone(ET)
    primary_verdict = SimpleNamespace(label="real", judged_at=now)
    runtime = shadow_judge.make_shadow_news_judge(
        lambda headline, summary: primary_verdict,
        lambda headline, summary: (entered.set(), release.wait(10), {"label": "real"})[2],
        log_path=tmp_path / "shadow.jsonl",
    )

    def judge_fn(candidate, *, now_et, premarket, edgar=None):
        return runtime(candidate.ticker, "synthetic summary"), None

    try:
        from momentum_scanner.judge_batch import process_judge_batch

        process_judge_batch(
            [_cand("AAA")], cache, now_et=now, premarket=False,
            judge_fn=judge_fn, edgar=None,
        )
        assert cache.get_fresh("AAA", now, premarket=False) is primary_verdict
        assert entered.wait(1.0)
    finally:
        release.set()
        runtime.close()


def test_run_daemon_stops_primary_worker_before_closing_shadow(tmp_path, monkeypatch):
    """Catches releasing the daemon-owned shadow before primary judging has stopped."""
    events = []

    class FakeWorker:
        def __init__(self, cache, judge_fn):
            pass

        def start(self):
            pass

        def stop(self, *, join_timeout):
            events.append("primary_stopped")
            return True

    class FakeJudge:
        def close_shadow(self):
            events.append("shadow_closed")

    monkeypatch.setattr(d, "JudgeWorker", FakeWorker)
    d.run_daemon(
        symbols=[], token_fn=lambda: "t", session=object(), limiter=_FakeLimiter(True),
        cache=VerdictCache(), judge_fn=FakeJudge(), float_cache={},
        float_fetch_fn=lambda *args, **kwargs: None, out_path=str(tmp_path / "wl.json"),
        heartbeat_path=str(tmp_path / "hb.json"), lock_path=str(tmp_path / "daemon.lock"),
        max_cycles=0,
    )

    assert events == ["primary_stopped", "shadow_closed"]


def test_run_daemon_defers_shadow_close_until_real_primary_worker_terminates(tmp_path, monkeypatch):
    """Catches either an early close or a permanently abandoned shadow after a timed join."""
    entered = threading.Event()
    release = threading.Event()
    created = []

    class BlockingJudge:
        def __init__(self):
            self.close_calls = 0

        def __call__(self, candidate, *, now_et, premarket, edgar=None):
            entered.set()
            release.wait(5.0)
            return _judge_fn_proceed(candidate, now_et=now_et, premarket=premarket, edgar=edgar)

        def close_shadow(self):
            assert created and not created[0]._thread.is_alive()
            self.close_calls += 1

    real_worker = d.JudgeWorker

    class CapturingWorker(real_worker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    blocking_judge = BlockingJudge()
    now = _now().astimezone(ET)
    monkeypatch.setattr(config, "DAEMON_WORKER_JOIN_SEC", 0.01)
    monkeypatch.setattr(d, "JudgeWorker", CapturingWorker)

    def enqueue_blocked_batch(**kwargs):
        kwargs["enqueue_fn"](([_cand("AAA")], now, False, None))
        assert entered.wait(1.0)
        return {"status": "ok", "stats": {}, "rate_backoff": 0}

    monkeypatch.setattr(d, "run_once", enqueue_blocked_batch)
    try:
        started = time.monotonic()
        d.run_daemon(
            symbols=["AAA"], token_fn=lambda: "t", session=object(), limiter=_FakeLimiter(True),
            cache=VerdictCache(), judge_fn=blocking_judge, float_cache={},
            float_fetch_fn=lambda *args, **kwargs: None, out_path=str(tmp_path / "wl.json"),
            heartbeat_path=str(tmp_path / "hb.json"), lock_path=str(tmp_path / "daemon.lock"),
            clock=_now, max_cycles=1,
        )
        assert time.monotonic() - started < 1.0
        assert created[0]._thread.is_alive()
        assert blocking_judge.close_calls == 0
    finally:
        release.set()
        if created:
            created[0]._thread.join(1.0)
            assert not created[0]._thread.is_alive()
            deadline = time.monotonic() + 1.0
            while blocking_judge.close_calls == 0 and time.monotonic() < deadline:
                time.sleep(0.01)
            assert blocking_judge.close_calls == 1


class _BlockingShadowJudge:
    """A close operation that must not wait for a synthetic blocked shadow thread."""

    def __init__(self):
        self.blocked = threading.Event()
        self.release = threading.Event()
        self.close_calls = 0
        self.thread = threading.Thread(target=self._block, daemon=True)

    def _block(self):
        self.blocked.set()
        self.release.wait(10)

    def start(self):
        self.thread.start()
        assert self.blocked.wait(1.0)

    def close_shadow(self):
        assert self.blocked.is_set()
        self.close_calls += 1

    def finish(self):
        self.release.set()
        self.thread.join(1.0)


def _patch_once_main(monkeypatch, blocking_judge, *, raise_from_run_once=False):
    from momentum_scanner import fundamentals, universe

    monkeypatch.setattr(d, "_configure_daemon_logging", lambda state_dir: None)
    monkeypatch.setattr(d, "build_news_client", lambda source: object())
    monkeypatch.setattr(d, "_build_real_judge_fn", lambda client: blocking_judge)
    monkeypatch.setattr(d, "load_rate_limiter", lambda path: _FakeLimiter(True))
    monkeypatch.setattr(universe, "load_universe", lambda *args, **kwargs: [])
    monkeypatch.setattr(fundamentals, "load_float_cache", lambda state_dir: {})
    monkeypatch.setattr(fundamentals, "save_float_cache", lambda state_dir, cache: None)

    def run_once(**kwargs):
        if raise_from_run_once:
            raise RuntimeError("synthetic --once failure")
        return {"status": "closed"}

    monkeypatch.setattr(d, "run_once", run_once)


def test_once_closes_shadow_once_after_success_while_shadow_is_blocked(tmp_path, monkeypatch):
    """Catches a successful --once process leaking its daemon-owned shadow runtime."""
    blocking_judge = _BlockingShadowJudge()
    blocking_judge.start()
    _patch_once_main(monkeypatch, blocking_judge)
    try:
        d.main(["--state-dir", str(tmp_path), "--once"])
        assert blocking_judge.close_calls == 1
        assert blocking_judge.thread.is_alive()
    finally:
        blocking_judge.finish()


def test_once_closes_shadow_once_after_exception_while_shadow_is_blocked(tmp_path, monkeypatch):
    """Catches an exception path that bypasses --once shadow cleanup."""
    blocking_judge = _BlockingShadowJudge()
    blocking_judge.start()
    _patch_once_main(monkeypatch, blocking_judge, raise_from_run_once=True)
    try:
        with pytest.raises(RuntimeError, match="synthetic --once failure"):
            d.main(["--state-dir", str(tmp_path), "--once"])
        assert blocking_judge.close_calls == 1
        assert blocking_judge.thread.is_alive()
    finally:
        blocking_judge.finish()


# ---- exit codes: a failed --once cycle or a failed startup must not exit 0 ----

class _PlainJudge:
    """Stand-in daemon judge with no shadow lane; never called by the exit-code tests."""

    def __call__(self, candidate, *, now_et, premarket, edgar=None):
        raise AssertionError("judge must not run in the exit-code tests")


def _patch_main_common(monkeypatch, *, universe_fn):
    from momentum_scanner import fundamentals, universe

    monkeypatch.delenv("SCANNER_SCORE_CAP", raising=False)
    monkeypatch.delenv("SCANNER_SCORE_CAP_VALUE", raising=False)
    monkeypatch.setattr(d, "_configure_daemon_logging", lambda state_dir: None)
    monkeypatch.setattr(d, "build_news_client", lambda source: object())
    monkeypatch.setattr(d, "_build_real_judge_fn", lambda client: _PlainJudge())
    monkeypatch.setattr(d, "load_rate_limiter", lambda path: _FakeLimiter(True))
    monkeypatch.setattr(universe, "load_universe", universe_fn)
    monkeypatch.setattr(fundamentals, "load_float_cache", lambda state_dir: {})
    monkeypatch.setattr(fundamentals, "save_float_cache", lambda state_dir, cache: None)


@pytest.mark.parametrize("status, ok", [
    ("ok", True), ("closed", True),
    ("error", False), ("rate_backoff", False), ("limiter_absent", False),
])
def test_once_exit_code_follows_cycle_status(tmp_path, monkeypatch, status, ok):
    _patch_main_common(monkeypatch, universe_fn=lambda *args, **kwargs: ["AAA"])
    monkeypatch.setattr(d, "run_once", lambda **kwargs: {"status": status})
    rc = d.main(["--state-dir", str(tmp_path), "--once"])
    if ok:
        assert rc == 0
    else:
        assert isinstance(rc, int) and rc != 0


def test_once_exits_nonzero_when_the_real_cycle_writes_status_error(tmp_path, monkeypatch):
    # Real run_once: an empty universe in a trading session publishes status:error.
    _patch_main_common(monkeypatch, universe_fn=lambda *args, **kwargs: [])
    monkeypatch.setattr(d, "session_mode", lambda now_et: "intraday")
    rc = d.main(["--state-dir", str(tmp_path), "--once"])
    assert _wl(tmp_path)["status"] == "error"
    assert isinstance(rc, int) and rc != 0


def test_startup_universe_failure_exits_nonzero(tmp_path, monkeypatch):
    def _universe_down(*args, **kwargs):
        raise RuntimeError("synthetic universe outage")

    _patch_main_common(monkeypatch, universe_fn=_universe_down)
    rc = d.main(["--state-dir", str(tmp_path), "--once"])
    assert _wl(tmp_path)["status"] == "error"
    assert isinstance(rc, int) and rc != 0
    rc = d.main(["--state-dir", str(tmp_path)])                    # same path without --once
    assert isinstance(rc, int) and rc != 0

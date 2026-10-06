"""Persistent intraday movers DISCOVERY DAEMON. One movers scan per cycle; a single
background worker thread runs the decoupled Gemma judge so discovery never blocks on the
slow judge, and the judge's clock travels with each batch. Self-gates to premarket/intraday windows,
pre-acquires the shared Schwab discovery budget (fail-CLOSED + self-sufficient on sustained
starvation), holds an OS single-instance lock with deterministic teardown, detects a dead worker, and
writes a rolling heartbeat. Writes the unchanged watchlist.json v2 (judged-only; first-sighting tickers
appear once judged). Fail-closed: outage / empty-universe / dead-worker / limiter-absent / sustained
rate-starvation / cycle-crash -> status:error. A failed --once cycle or a failed startup exits non-zero."""
import argparse
import json
import logging
import logging.handlers
import os
import signal
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from momentum_scanner import config
from momentum_scanner import movers_scan
from momentum_scanner import score_cap_state
from momentum_scanner import watchlist_io
from momentum_scanner.daemon_clock import session_mode
from momentum_scanner.daemon_heartbeat import write_heartbeat
from momentum_scanner.daemon_lock import single_instance_lock, AlreadyRunningError  # re-exported
from momentum_scanner.edgar_prejudge import EdgarContext
from momentum_scanner.errors import ScannerNetworkError
from momentum_scanner.judge_batch import process_judge_batch
from momentum_scanner.judge_runtime import make_verdict_lookup
from momentum_scanner.rate_gate import acquire_discovery_budget, discovery_chunks, load_rate_limiter

ET = ZoneInfo("America/New_York")
_log = logging.getLogger(__name__)

_DEFAULT_SHADOW_METRICS = {
    "shadow_status": "off",
    "shadow_worker_alive": False,
    "shadow_queue_depth": 0,
    "shadow_submitted": 0,
    "shadow_completed": 0,
    "shadow_dropped": 0,
    "shadow_dropped_daily_cap": 0,
    "shadow_dropped_queue_full": 0,
    "shadow_dropped_capture_failed": 0,
    "shadow_dropped_worker_dead": 0,
    "shadow_dropped_closed": 0,
    "shadow_fatal_errors": 0,
    "shadow_abandoned_backlog": 0,
    "shadow_write_failures": 0,
    "shadow_last_error_type": None,
}

# The qwen3 lane's own metrics, same key shape as above under a qwen3_ prefix -- judge.py's
# _wrap_with_qwen3_shadow merges these into the SAME metrics_fn() dict the haiku keys come
# from (see resolve_news_judge_runtime), so one shadow_metrics() call reports both lanes.
_DEFAULT_QWEN3_SHADOW_METRICS = {f"qwen3_{key}": value for key, value in _DEFAULT_SHADOW_METRICS.items()}
_DEFAULT_ALL_SHADOW_METRICS = {**_DEFAULT_SHADOW_METRICS, **_DEFAULT_QWEN3_SHADOW_METRICS}


def _shadow_metrics_projection(shadow_metrics):
    """Return a bounded shadow metrics snapshot (both lanes) without making telemetry
    load-bearing."""
    values = dict(_DEFAULT_ALL_SHADOW_METRICS)
    if shadow_metrics is None:
        return values
    try:
        reported = shadow_metrics()
        if not isinstance(reported, dict):
            raise TypeError("shadow metrics must be a dictionary")
    except Exception as exc:                              # noqa: BLE001 -- heartbeat must still be written
        values["shadow_status"] = "degraded"
        values["shadow_last_error_type"] = type(exc).__name__
        values["qwen3_shadow_status"] = "degraded"
        values["qwen3_shadow_last_error_type"] = type(exc).__name__
        return values
    for key in values:
        if key in reported:
            values[key] = reported[key]
    return values


class ScoreCapStartupAbort(RuntimeError):
    """Score-cap startup enforcement failed: INVALID config,
    malformed marker, or marker/config mismatch. The daemon must NOT start the scan loop and
    must NOT publish anything -- a requested-ON cap can never silently run uncapped, and an
    uncapped process can never masquerade as capped."""


def score_cap_startup(state_dir):
    """Resolve the score-cap config and check it against the durable marker (if any).

    Returns (resolution, banner) on success -- banner is one of the two healthy states
    ('score cap: ENABLED value=...' / 'score cap: OFF'). Raises ScoreCapStartupAbort with the
    third-state banner text on INVALID config (naming the raw rejected token), a malformed
    marker, or a marker/config mismatch. Marker ABSENT is the inert pre-activation state:
    startup proceeds exactly as today."""
    res = config.resolve_score_cap()
    if res.state == "invalid":
        raise ScoreCapStartupAbort("score cap: STARTUP ABORT (invalid config: %s)" % res.reason)
    try:
        marker = score_cap_state.read_marker(state_dir)
    except score_cap_state.MalformedMarkerError as exc:
        raise ScoreCapStartupAbort("score cap: STARTUP ABORT (malformed marker: %s)" % exc) from exc
    mismatch = score_cap_state.marker_matches_config(marker, res)
    if mismatch is not None:
        raise ScoreCapStartupAbort("score cap: STARTUP ABORT (marker mismatch: %s)" % mismatch)
    banner = ("score cap: ENABLED value=%s" % res.value) if res.state == "on" else "score cap: OFF"
    return res, banner


def _judge_systemic_failures():
    """Cumulative systemic (news-feed/EDGAR) judge failures, for the heartbeat. Imported lazily to
    match _build_real_judge_fn and keep judge's import chain off the daemon's module load. Never let
    an observability read kill a cycle -- an unreadable counter reports -1, not an exception."""
    try:
        from momentum_scanner.judge import systemic_failures
        return systemic_failures()
    except Exception:                                  # noqa: BLE001 -- telemetry is never load-bearing
        _log.exception("daemon: could not read judge systemic-failure counter (non-fatal)")
        return -1


class JudgeWorker:
    """Single background judge thread. Coalescing single-slot handoff (latest snapshot wins):
    discovery never blocks and the worker always judges the FRESHEST batch. The clock travels
    WITH each batch (now_et, premarket) -> fixes the frozen clock. Any worker exit (incl BaseException)
    flips _alive so the daemon detects a dead worker and fails closed. A normal Exception
    inside a batch is isolated -- the worker keeps running."""

    def __init__(self, cache, judge_fn):
        self._cache = cache
        self._judge_fn = judge_fn
        self._slot = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._alive = True
        self._drops = 0
        self._thread = threading.Thread(target=self._run, name="movers-judge", daemon=False)

    def start(self):
        self._thread.start()

    def enqueue(self, batch):
        """Non-blocking, latest-wins. batch = (candidates, now_et, premarket, edgar_ctx) (edgar_ctx
        is the per-cycle EdgarContext built in run_once, threaded through to process_judge_batch so a
        ticker pre-judged this cycle reuses the same EDGAR cache when judged)."""
        with self._lock:
            if self._slot is not None:
                self._drops += 1
            self._slot = batch
        self._wake.set()

    def _run(self):
        try:
            while not self._stop.is_set():
                if not self._wake.wait(timeout=1.0):
                    continue
                self._wake.clear()
                with self._lock:
                    batch, self._slot = self._slot, None
                if batch is None:
                    continue
                candidates, now_et, premarket, edgar_ctx = batch
                try:
                    process_judge_batch(candidates, self._cache, now_et=now_et,
                                        premarket=premarket, judge_fn=self._judge_fn, edgar=edgar_ctx)
                except Exception:
                    _log.exception("judge worker: batch failed (isolated, continuing)")
        except BaseException:                          # noqa: BLE001 -- any exit flips health
            _log.exception("judge worker: fatal exit")
            raise
        finally:
            self._alive = False

    def is_alive(self):
        return self._thread.is_alive() and self._alive

    def queue_depth(self):
        with self._lock:
            return 1 if self._slot is not None else 0

    def drops(self):
        return self._drops

    def stop(self, *, join_timeout):
        self._stop.set()
        self._wake.set()
        if self._thread.ident is not None:             # robust if start() never ran
            self._thread.join(timeout=join_timeout)
        return not self._thread.is_alive()

    def wait_stopped(self):
        """Wait outside the daemon main thread until the primary thread has really exited."""
        if self._thread.ident is not None:
            self._thread.join()


def run_once(*, symbols, token_fn, session, limiter, cache, enqueue_fn, float_cache, float_fetch_fn,
             now_utc, now_et, mode, out_path, heartbeat_path, started_at_iso, pid, worker_alive,
             queue_depth, drops, rate_backoff, shadow_metrics=None):
    premarket = (mode == "premarket")
    stats = {}

    def _hb(last_status, rb):
        write_heartbeat(heartbeat_path, pid=pid, started_at_iso=started_at_iso,
                        now_iso=now_utc.isoformat(), mode=mode, last_status=last_status,
                        cycle_stats=stats,
                        extra={"judge_worker_alive": worker_alive, "queue_depth": queue_depth,
                               "drops": drops, "rate_backoff": rb,
                               "judge_systemic_failures": _judge_systemic_failures(),
                               **_shadow_metrics_projection(shadow_metrics)})

    def _result(status, rb):
        return {"status": status, "mode": mode, "candidates": stats.get("candidates", 0),
                "stats": stats, "rate_backoff": rb}

    if not worker_alive:                               # dead worker -> fail closed (even when closed)
        movers_scan.write_from_candidates([], out_path=out_path, status="error",
                                          error="judge worker not alive")
        _hb("judge_dead", rate_backoff)
        return _result("error", rate_backoff)

    if mode == "closed":                               # still heartbeat (continuous liveness)
        _hb("closed", rate_backoff)
        return _result("closed", rate_backoff)

    chunks = discovery_chunks(len(symbols))            # guard chunks<=0 (acquire(0)=ValueError)
    if chunks <= 0:
        movers_scan.write_from_candidates([], out_path=out_path, status="error", error="empty universe")
        _hb("error", rate_backoff)
        return _result("error", rate_backoff)

    granted = acquire_discovery_budget(limiter, chunks, now=now_utc.timestamp(),
                                       allow_ungated=config.DAEMON_ALLOW_UNGATED)
    if not granted:                                    # absent limiter fails CLOSED, not ungated
        rb = rate_backoff + 1
        status = "limiter_absent" if limiter is None else "rate_backoff"
        # Self-sufficient: once backoff outlasts the freshness window, proactively fail closed
        # instead of leaving a stale status:ok watchlist for a downstream age gate to catch.
        if rb * config.DISCOVERY_INTERVAL_SEC >= config.WATCHLIST_MAX_AGE_SEC:
            movers_scan.write_from_candidates([], out_path=out_path, status="error",
                                              error=f"discovery starved ({status})")
        _hb(status, rb)
        return _result(status, rb)

    # One EdgarContext per cycle, shared by the tier-2 pre-judge signal now and the judge batch later
    # -- so a ticker resolved here is not re-fetched from EDGAR when judged. Threaded through the
    # enqueue_fn payload below into process_judge_batch -> judge_fn -> judge_candidate -> evaluate_news.
    edgar_ctx = EdgarContext(now=now_utc) if config.EDGAR_PREJUDGE_ENABLED else None

    try:
        candidates = movers_scan.discover_movers(
            symbols, token_fn=token_fn, session=session, float_cache=float_cache,
            float_fetch_fn=float_fetch_fn, now_utc=now_utc, now_et=now_et, premarket=premarket,
            edgar=edgar_ctx, stats=stats)
    except ScannerNetworkError as exc:                 # whole-scan outage -> status:error
        movers_scan.write_from_candidates([], out_path=out_path, status="error", error=str(exc))
        _hb("error", rate_backoff)
        return _result("error", rate_backoff)

    enqueue_fn((candidates, now_et, premarket, edgar_ctx))  # stage for the worker (clock
                                                        # travels); same-cycle edgar_ctx rides along
    verdict_lookup = make_verdict_lookup(cache, now_et=now_et, premarket=premarket)  # per cycle
    movers_scan.write_from_candidates(candidates, verdict_lookup=verdict_lookup, out_path=out_path,
                                      status="ok")     # unjudged tickers absent (movers_watchlist)
    stats.setdefault("candidates", len(candidates))
    _log.info("discovery mode=%s tier1_hits=%s candidates=%s queue_depth=%s drops=%s",
              mode, stats.get("tier1_hits"), stats.get("candidates"), queue_depth, drops)  # per-cycle funnel log
    _hb("ok", 0)                                       # granted cycle resets consecutive-backoff
    return _result("ok", 0)


def _install_signal_handlers(stop_event):
    def _handler(signum, _frame):
        _log.info("daemon: signal %s -> stopping after the current cycle", signum)
        stop_event.set()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):                  # not main thread / unsupported -> rely on finally
            _log.warning("daemon: could not install handler for signal %s", sig)


def _configure_daemon_logging(state_dir):
    """stderr + rotating discovery_daemon.log so premarket funnel evidence survives."""
    handlers = [logging.StreamHandler()]
    try:
        handlers.append(logging.handlers.RotatingFileHandler(
            str(Path(state_dir) / "discovery_daemon.log"),
            maxBytes=5 * 1024 * 1024, backupCount=5))
    except OSError:
        pass
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        handlers=handlers)


def _accumulate_funnel(funnel_path, result, now_et):
    """Cumulative per-ET-day discovery funnel (cycles, by_status, summed numeric
    stats). Best-effort: any failure logs and returns -- never kills a cycle."""
    if not funnel_path:
        return
    try:
        p = Path(funnel_path)
        data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        day = data.setdefault(now_et.date().isoformat(),
                              {"cycles": 0, "by_status": {}, "stats": {}})
        day["cycles"] += 1
        status = result.get("status", "unknown")
        day["by_status"][status] = day["by_status"].get(status, 0) + 1
        for k, v in (result.get("stats") or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                day["stats"][k] = day["stats"].get(k, 0) + v
        watchlist_io.atomic_write_json(p, data)
    except Exception:
        _log.exception("daemon: funnel counter write failed (non-fatal)")


def _defer_shadow_close_until_primary_stops(worker, close_shadow):
    """Close exactly once on a reaper after a bounded main-thread shutdown times out."""
    def reap():
        worker.wait_stopped()
        close_shadow()

    threading.Thread(target=reap, name="movers-shadow-close", daemon=False).start()


def run_daemon(*, symbols, token_fn, session, limiter, cache, judge_fn, float_cache, float_fetch_fn,
               out_path, heartbeat_path, lock_path, clock=None, max_cycles=None, funnel_path=None):
    clock = clock or (lambda: datetime.now(timezone.utc))
    shadow_metrics = getattr(judge_fn, "shadow_metrics", None)
    with single_instance_lock(lock_path):              # lock is the OUTER guard
        worker = JudgeWorker(cache, judge_fn)
        stop = threading.Event()
        try:                                           # worker started + torn down INSIDE the try
            worker.start()
            _install_signal_handlers(stop)
            started_at_iso = clock().isoformat()
            pid = os.getpid()
            rate_backoff = 0
            cycles = 0
            while not stop.is_set() and (max_cycles is None or cycles < max_cycles):
                now_utc = clock()                      # OUTSIDE the per-cycle guard: clock errors propagate (teardown test)
                try:
                    now_et = now_utc.astimezone(ET)    # one clock read -> both (no skew)
                    mode = session_mode(now_et)
                    result = run_once(
                        symbols=symbols, token_fn=token_fn, session=session, limiter=limiter,
                        cache=cache, enqueue_fn=worker.enqueue, float_cache=float_cache,
                        float_fetch_fn=float_fetch_fn, now_utc=now_utc, now_et=now_et, mode=mode,
                        out_path=out_path, heartbeat_path=heartbeat_path, started_at_iso=started_at_iso,
                        pid=pid, worker_alive=worker.is_alive(), queue_depth=worker.queue_depth(),
                        drops=worker.drops(), rate_backoff=rate_backoff, shadow_metrics=shadow_metrics)
                    rate_backoff = result["rate_backoff"]
                    _accumulate_funnel(funnel_path, result, now_et)
                    nap = config.DAEMON_IDLE_INTERVAL_SEC if mode == "closed" else config.DISCOVERY_INTERVAL_SEC
                except Exception:                      # per-cycle fail-closed + continue
                    _log.exception("daemon: cycle crashed")
                    try:
                        movers_scan.write_from_candidates([], out_path=out_path, status="error",
                                                          error="daemon cycle crash")
                        write_heartbeat(heartbeat_path, pid=os.getpid(), started_at_iso=started_at_iso,
                                        now_iso=clock().isoformat(), mode="unknown", last_status="crash",
                                        cycle_stats={}, extra={"judge_worker_alive": worker.is_alive(),
                                                               "queue_depth": worker.queue_depth(),
                                                                "drops": worker.drops(),
                                                                "rate_backoff": rate_backoff,
                                                                "judge_systemic_failures":
                                                                    _judge_systemic_failures(),
                                                                **_shadow_metrics_projection(shadow_metrics)})
                    except Exception:
                        _log.exception("daemon: failed to record crash state")
                    _accumulate_funnel(funnel_path, {"status": "crash", "stats": {}},
                                      clock().astimezone(ET))
                    nap = config.DISCOVERY_INTERVAL_SEC
                cycles += 1
                if stop.is_set() or (max_cycles is not None and cycles >= max_cycles):
                    break
                stop.wait(timeout=nap)                 # interruptible sleep -> low shutdown latency
        finally:                                       # unconditional teardown on ALL paths
            stop.set()
            primary_stopped = worker.stop(join_timeout=config.DAEMON_WORKER_JOIN_SEC)
            if primary_stopped:
                close_shadow = getattr(judge_fn, "close_shadow", None)
                if close_shadow is not None:
                    close_shadow()
            else:
                _log.warning("daemon: primary judge worker still alive after bounded shutdown; "
                             "deferring shadow close until primary termination")
                close_shadow = getattr(judge_fn, "close_shadow", None)
                if close_shadow is not None:
                    _defer_shadow_close_until_primary_stops(worker, close_shadow)


@dataclass
class DaemonNewsJudge:
    """Daemon-lifetime primary judge plus its shadow runtime descriptor."""

    news_client: object
    resolved: object

    @property
    def shadow_status(self):
        return self.resolved.status

    def __call__(self, candidate, *, now_et, premarket, edgar=None):
        from momentum_scanner.judge import judge_candidate

        return judge_candidate(
            candidate,
            news_client=self.news_client,
            now_et=now_et,
            premarket=premarket,
            news_judge=self.resolved.callable,
            edgar=edgar,
        )

    def shadow_metrics(self):
        try:
            return dict(self.resolved.metrics_fn())
        except Exception as exc:                        # noqa: BLE001 -- telemetry is never load-bearing
            # A total metrics_fn() failure means BOTH lanes are unreadable, not just the
            # haiku one -- without the qwen3_* keys here, _shadow_metrics_projection fills
            # them from _DEFAULT_QWEN3_SHADOW_METRICS ("off"), which reads as "deliberately
            # disabled" rather than "broken". Degrade both.
            return {
                "shadow_status": "degraded",
                "shadow_worker_alive": False,
                "shadow_last_error_type": type(exc).__name__,
                "qwen3_shadow_status": "degraded",
                "qwen3_shadow_worker_alive": False,
                "qwen3_shadow_last_error_type": type(exc).__name__,
            }

    def close_shadow(self):
        try:
            self.resolved.close_fn()
        except Exception:                               # noqa: BLE001 -- shutdown must continue
            _log.warning("daemon: shadow close failed (non-fatal)")


def _build_real_judge_fn(news_client):
    from momentum_scanner.judge import resolve_news_judge_runtime

    return DaemonNewsJudge(news_client, resolve_news_judge_runtime())


def build_news_client(source):
    """Select the judge's news client. Default 'finnhub' preserves current behavior;
    'alpaca' uses the Benzinga-sourced Alpaca News API as PRIMARY with a Finnhub FALLBACK so a single-
    feed outage degrades rather than blanks the catalyst signal. Any unknown/None value fails safe back
    to Finnhub."""
    from momentum_scanner.finnhub_news import FinnhubClient
    src = (source or "finnhub").strip().lower()
    if src == "alpaca":
        from momentum_scanner.alpaca_news import AlpacaNewsClient, FallbackNewsClient
        alpaca = AlpacaNewsClient()
        # The Finnhub free tier is non-commercial. In live mode, refuse it as a fallback (mirrors the
        # run_premarket_scan live-gate) so an Alpaca outage fails CLOSED rather than silently sourcing
        # live signals from an unlicensed feed. This daemon is a separate entry point from
        # run_premarket_scan, so it needs its own guard.
        live = os.environ.get("SCANNER_MODE", "paper") == "live"
        if live and not config.FINNHUB_COMMERCIAL_USE:
            _log.warning("live mode: Finnhub fallback disabled (non-commercial); Alpaca-only, "
                         "fail-closed on outage")
            return alpaca
        return FallbackNewsClient(alpaca, FinnhubClient())
    if src != "finnhub":
        _log.warning("unknown news source %r; using finnhub", source)
    return FinnhubClient()


def _sync_enqueue(cache, judge_fn):
    """--once / deterministic single-cycle: judge BEFORE the same-cycle verdict read."""
    def enqueue(batch):
        candidates, now_et, premarket, edgar_ctx = batch
        process_judge_batch(candidates, cache, now_et=now_et, premarket=premarket, judge_fn=judge_fn,
                            edgar=edgar_ctx)
    return enqueue


# --once cycle statuses that count as a successful run. "closed" (outside the session windows) did
# its job by writing the heartbeat. Everything else -- "error" (status:error published),
# "rate_backoff" and "limiter_absent" (discovery skipped) -- is a failed cycle and exits non-zero.
_ONCE_OK_STATUSES = frozenset({"ok", "closed"})


def _once_exit_code(result):
    status = result.get("status") if isinstance(result, dict) else None
    return 0 if status in _ONCE_OK_STATUSES else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="Movers discovery daemon")
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--once", action="store_true", help="run a single cycle under the lock and exit")
    ap.add_argument("--source", choices=["finnhub", "alpaca"], default=config.SCANNER_NEWS_SOURCE,
                    help="news source: finnhub (default) | alpaca (Benzinga wire + Finnhub fallback). "
                         "Env SCANNER_NEWS_SOURCE sets the default; this flag overrides it.")
    args = ap.parse_args(argv)

    import requests
    from momentum_scanner import universe, universe_fetch, fundamentals
    from momentum_scanner.verdict_cache import VerdictCache
    from momentum_scanner.schwab_data import _default_token_fn

    state_dir = args.state_dir
    _configure_daemon_logging(state_dir)

    # Score-cap startup enforcement BEFORE anything can publish: on INVALID
    # config, malformed marker, or marker/config mismatch -> ERROR banner, NO scan loop, NO
    # watchlist, NO heartbeat. This must precede the universe load (which writes status:error).
    try:
        _cap_resolution, cap_banner = score_cap_startup(state_dir)
    except ScoreCapStartupAbort as exc:
        _log.error(str(exc))
        return 2
    _log.info(cap_banner)

    out_path = os.path.join(state_dir, "watchlist.json")
    heartbeat_path = os.path.join(state_dir, config.DAEMON_HEARTBEAT_FILE)
    lock_path = os.path.join(state_dir, "daemon.lock")
    funnel_path = os.path.join(state_dir, "discovery_funnel.json")
    rate_state = os.environ.get("SCHWAB_RATE_LIMIT_STATE",
                                os.path.join(state_dir, "schwab_rate_limit.json"))

    token_fn = _default_token_fn
    session = requests.Session()
    now_utc = datetime.now(timezone.utc)

    # WRAP the startup universe load -> a no-cache fetch failure writes status:error, not a crash.
    try:
        symbols = universe.load_universe(
            state_dir, fetch_fn=lambda name: universe_fetch.fetch_symbol_directory(name, session=session),
            now_utc=now_utc)
    except Exception as exc:
        _log.exception("daemon: universe load failed at startup; writing status:error")
        movers_scan.write_from_candidates([], out_path=out_path, status="error",
                                          error=f"universe unavailable: {type(exc).__name__}")
        write_heartbeat(heartbeat_path, pid=os.getpid(), started_at_iso=now_utc.isoformat(),
                        now_iso=now_utc.isoformat(), mode="startup", last_status="error", cycle_stats={},
                        extra={"judge_worker_alive": False, "queue_depth": 0, "drops": 0, "rate_backoff": 0,
                               "judge_systemic_failures": _judge_systemic_failures(),
                               **_shadow_metrics_projection(None)})
        return 1                                                   # failed startup -> non-zero exit

    cache = VerdictCache()
    float_cache = fundamentals.load_float_cache(state_dir)         # warm-load (avoid restart yfinance burst)
    judge_fn = _build_real_judge_fn(build_news_client(args.source))
    limiter = load_rate_limiter(rate_state)

    def float_fetch_fn(ticker):
        return fundamentals.fetch_float(ticker)

    if args.once:
        try:
            with single_instance_lock(lock_path):                 # --once shares the daemon lock
                now_et = now_utc.astimezone(ET)
                result = run_once(
                    symbols=symbols, token_fn=token_fn, session=session, limiter=limiter, cache=cache,
                    enqueue_fn=_sync_enqueue(cache, judge_fn), float_cache=float_cache,
                    float_fetch_fn=float_fetch_fn, now_utc=now_utc, now_et=now_et,
                    mode=session_mode(now_et), out_path=out_path, heartbeat_path=heartbeat_path,
                    started_at_iso=now_utc.isoformat(), pid=os.getpid(), worker_alive=True,
                    queue_depth=0, drops=0, rate_backoff=0,
                    shadow_metrics=getattr(judge_fn, "shadow_metrics", None))
        finally:
            close_shadow = getattr(judge_fn, "close_shadow", None)
            if close_shadow is not None:
                close_shadow()
        fundamentals.save_float_cache(state_dir, float_cache)     # persist any newly-fetched floats
        return _once_exit_code(result)

    try:
        run_daemon(symbols=symbols, token_fn=token_fn, session=session, limiter=limiter, cache=cache,
                   judge_fn=judge_fn, float_cache=float_cache, float_fetch_fn=float_fetch_fn,
                   out_path=out_path, heartbeat_path=heartbeat_path, lock_path=lock_path,
                   funnel_path=funnel_path)
    finally:
        fundamentals.save_float_cache(state_dir, float_cache)     # persist on shutdown


if __name__ == "__main__":          # pragma: no cover
    raise SystemExit(main())

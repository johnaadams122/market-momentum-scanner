# judge.py
"""Single-ticker catalyst judge for the decoupled movers worker (Gemma-only). For a
discovered mover it fetches the ticker's same-day company news within a MODE-AWARE window (premarket:
[04:00, min(now,09:30)]; intraday: [04:00, now]) and runs the existing fail-closed gate
(evaluate_news, raise_systemic=True) -- EDGAR dilution classification (unresolvable CIK -> demote,
not veto) -> news/EDGAR-content floor -> bounded judge feeding the score -- then stamps judged_at +
the catalyst headline. A SYSTEMIC failure (Finnhub OR EDGAR ScannerError) leaves the ticker UNJUDGED
(returns None) so it is retried next cycle, never an outage-induced cached REJECT. The backend is a
SEAM: Gemma only; Haiku/sonnet are not enabled as deciding backends and fall back to Gemma here so
they slot in later with no interface change."""
import json
import logging
import os
import threading
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from momentum_scanner import catalyst_evaluator, config, haiku_judge, qwen3_judge, shadow_judge
from momentum_scanner.catalyst_evaluator import default_news_judge, evaluate_news
from momentum_scanner.errors import ScannerError
from momentum_scanner.finnhub_news import ET, et_window

_log = logging.getLogger(__name__)

# --- Systemic-failure visibility ----------------------------------------------
# A systemic failure leaves the ticker UNJUDGED, which is the correct fail-closed behavior -- but it
# would be invisible at DEBUG, and the daemons run at INFO. A total news-feed outage would then
# produce a run that looks IDENTICAL to an honest one with no qualifying catalysts: candidates
# discovered, zero PROCEED, last_status "ok". You cannot tell a dead feed from a quiet tape without
# this counter, so a degraded run reads as a clean one.
#
# Volume is bounded: an outage fails every ticker of every cycle, so warning per-ticker would be the
# reason someone turns the log level back down. Warn on the FIRST failure of a burst (a burst ends
# when any judge succeeds, so recovery re-arms the alarm) and every _SYSTEMIC_LOG_EVERY after that,
# always carrying the cumulative count. Everything else stays at DEBUG.
_SYSTEMIC_LOG_EVERY = 25
_systemic_lock = threading.Lock()
_systemic_failures = 0          # cumulative since process start; surfaced in the daemon heartbeat
_systemic_consecutive = 0       # current burst length; reset by any successful judge


def systemic_failures():
    """Cumulative count of judge_candidate calls left UNJUDGED by a systemic (news-feed or EDGAR)
    failure since process start. Read by the movers daemon for the heartbeat."""
    with _systemic_lock:
        return _systemic_failures


def reset_systemic_failures():
    """Zero both counters. For tests and for a caller that wants per-run rather than per-process
    accounting; the daemon never calls this."""
    global _systemic_failures, _systemic_consecutive
    with _systemic_lock:
        _systemic_failures = 0
        _systemic_consecutive = 0


def _record_systemic_failure():
    """Count one systemic failure; return (cumulative, should_log_above_debug)."""
    global _systemic_failures, _systemic_consecutive
    with _systemic_lock:
        _systemic_failures += 1
        _systemic_consecutive += 1
        return _systemic_failures, (_systemic_consecutive == 1
                                    or _systemic_consecutive % _SYSTEMIC_LOG_EVERY == 0)


def _record_judge_success():
    """End the current failure burst so the next outage warns immediately."""
    global _systemic_consecutive
    with _systemic_lock:
        _systemic_consecutive = 0


@dataclass(frozen=True)
class ResolvedNewsJudge:
    """The deciding callable plus shadow lifecycle state owned by the caller."""

    callable: Callable[[str, str], object]
    status: str
    metrics_fn: Callable[[], dict]
    close_fn: Callable[[], None]


def _bare_news_judge(primary, status):
    """Return a no-throw descriptor for every non-active admission outcome."""
    return ResolvedNewsJudge(
        callable=primary,
        status=status,
        metrics_fn=lambda: {"shadow_status": status},
        close_fn=lambda: None,
    )


# Log dedup: resolve_news_judge runs PER CANDIDATE via the compatibility wrapper
# (`news_judge or resolve_news_judge()`), so without dedup every candidate would emit an
# identical INFO line and flood a size-rotated log. ONLY the logging is deduped, never
# the resolution: haiku_judge.has_api_key() stays a CALL-time read (a launcher that gains
# the key mid-life needs no code change), and a STATE CHANGE re-logs, so a mid-run degrade
# is never swallowed.
_shadow_log_lock = threading.Lock()
_last_shadow_log_state = None


def _reset_shadow_log_state():
    """Test seam: the dedupe is module state and would leak across tests."""
    global _last_shadow_log_state
    with _shadow_log_lock:
        _last_shadow_log_state = None


def _log_shadow_state_once(state, msg, *args):
    """Emit `msg` only when the shadow's resolved state differs from the last one."""
    global _last_shadow_log_state
    with _shadow_log_lock:
        if _last_shadow_log_state == state:
            return
        _last_shadow_log_state = state
    _log.info(msg, *args)


# Independent dedup state for the qwen3 lane below. Deliberately a SEPARATE
# lock/variable from the haiku pair above, not a shared one keyed by lane: the two lanes
# resolve on every candidate, and a single "last state" shared across both would see them
# alternate on every call (haiku's state, then qwen3's, then haiku's...) and never dedupe
# either one -- each lane needs to compare only against ITS OWN prior state.
_qwen3_shadow_log_lock = threading.Lock()
_last_qwen3_shadow_log_state = None


def _reset_qwen3_shadow_log_state():
    """Test seam: the dedupe is module state and would leak across tests."""
    global _last_qwen3_shadow_log_state
    with _qwen3_shadow_log_lock:
        _last_qwen3_shadow_log_state = None


def _log_qwen3_shadow_state_once(state, msg, *args):
    global _last_qwen3_shadow_log_state
    with _qwen3_shadow_log_lock:
        if _last_qwen3_shadow_log_state == state:
            return
        _last_qwen3_shadow_log_state = state
    _log.info(msg, *args)


def _resolve_haiku_layer(backend=None):
    """Resolve the news judge's gemma+haiku layer. gemma ALWAYS decides.

    Two orthogonal knobs meet here and are easy to confuse:
      - SCANNER_JUDGE_BACKEND picks the DECIDING model. haiku/sonnet are not enabled
        and fall back to gemma, so this seam still cannot change a trade.
      - SCANNER_JUDGE_SHADOW adds an OBSERVED-only second judge. The returned callable
        is transparent -- it forwards gemma's verdict unchanged and logs both.
    Every failure mode below degrades to the bare gemma judge rather than raising: a
    missing key, a broken wrapper, anything. The scanner keeps judging.

    This is layer 1 of resolve_news_judge_runtime; see _wrap_with_qwen3_shadow for the
    second, independent shadow lane composed on top of whatever this returns.
    """
    b = (backend or config.SCANNER_JUDGE_BACKEND or "gemma").lower()
    if b in ("haiku", "sonnet"):
        _log.info("judge backend %s is not enabled; using gemma", b)
    elif b != "gemma":
        _log.warning("unknown judge backend %r; using gemma", b)

    primary = catalyst_evaluator.default_news_judge
    if not config.SCANNER_JUDGE_SHADOW:
        return _bare_news_judge(primary, "off")
    if not haiku_judge.has_api_key():
        _log_shadow_state_once(
            "no_key",
            "judge shadow requested but ANTHROPIC_API_KEY is unset; "
            "running gemma-only (no shadow rows will be written)")
        return _bare_news_judge(primary, "disabled_no_key")
    runtime = None
    try:
        path = Path(config.SHADOW_JUDGE_LOG_PATH)
        if not path.is_absolute() or not path.parent.is_dir():
            raise ValueError("shadow log path must be absolute with an existing parent")
        runtime = shadow_judge.ShadowJudgeRuntime(
            primary, haiku_judge.haiku_news_judge,
            log_path=config.SHADOW_JUDGE_LOG_PATH,
            error_fn=haiku_judge.take_last_error,
            primary_backend="gemma",
            shadow_backend=config.HAIKU_MODEL,
            queue_cap=config.SHADOW_JUDGE_QUEUE_CAP,
            daily_call_cap=config.SHADOW_JUDGE_DAILY_CALL_CAP,
        )
        runtime.start()
        control_row = {
            "record_type": "shadow_start",
            "ts": datetime.now(timezone.utc).isoformat(),
            "shadow_backend": config.HAIKU_MODEL,
            "queue_cap": config.SHADOW_JUDGE_QUEUE_CAP,
            "daily_call_cap": config.SHADOW_JUDGE_DAILY_CALL_CAP,
        }
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(control_row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _log_shadow_state_once(
            "active", "judge shadow ACTIVE: gemma decides, %s logged to %s",
            config.HAIKU_MODEL, config.SHADOW_JUDGE_LOG_PATH)
        def metrics():
            values = runtime.metrics()
            values.setdefault("shadow_status", "active")
            return values
        return ResolvedNewsJudge(runtime, "active", metrics, runtime.close)
    except Exception as exc:
        # NOT deduped (deliberate): an exception
        # carries detail that may differ per occurrence; losing a repeat would hide
        # a setup failure that changes over time.
        if runtime is not None:
            try:
                runtime.close()
            except Exception:
                pass
        message = shadow_judge.redact(f"{type(exc).__name__}: {exc}")[:500]
        _log.warning("judge shadow setup failed; running gemma-only: %s", message)
        return _bare_news_judge(primary, "setup_failed")


def _wrap_with_qwen3_shadow(resolved):
    """Optionally add a SECOND, independent passive shadow lane (qwen3:8b) around
    whatever the haiku layer above produced. Fully orthogonal to it: `resolved.status`
    is left describing the haiku layer unchanged, so no existing status-string
    assertion anywhere in the daemon or tests can observe this addition -- qwen3's own
    state is surfaced only through the merged metrics_fn, under a qwen3_ prefix.
    Degrades to `resolved` completely unchanged (same object) on ANY problem, same
    fail-open contract as the haiku layer.

    Composition, not duplication: ShadowJudgeRuntime.__call__ already returns whatever
    its `primary` callable returns, unmodified -- wrapping `resolved.callable` (which is
    either bare gemma or the haiku-shadowed gemma) a second time costs nothing new on
    the invariant that a verdict is always gemma's, byte-for-byte, and needs zero
    changes to shadow_judge.py or the existing haiku wiring above.
    """
    if not config.SCANNER_JUDGE_SHADOW_QWEN3:
        def metrics():
            values = resolved.metrics_fn()
            values["qwen3_shadow_status"] = "off"
            return values
        return ResolvedNewsJudge(resolved.callable, resolved.status, metrics, resolved.close_fn)

    runtime = None
    try:
        path = Path(config.SHADOW_JUDGE_LOG_PATH_QWEN3)
        if not path.is_absolute() or not path.parent.is_dir():
            raise ValueError("qwen3 shadow log path must be absolute with an existing parent")
        runtime = shadow_judge.ShadowJudgeRuntime(
            resolved.callable, qwen3_judge.qwen3_news_judge,
            log_path=config.SHADOW_JUDGE_LOG_PATH_QWEN3,
            error_fn=qwen3_judge.take_last_error,
            primary_backend="gemma",
            shadow_backend=config.QWEN3_MODEL,
            queue_cap=config.SHADOW_JUDGE_QUEUE_CAP,
            daily_call_cap=config.SHADOW_JUDGE_QWEN3_DAILY_CALL_CAP,
        )
        runtime.start()
        control_row = {
            "record_type": "shadow_start",
            "ts": datetime.now(timezone.utc).isoformat(),
            "shadow_backend": config.QWEN3_MODEL,
            "queue_cap": config.SHADOW_JUDGE_QUEUE_CAP,
            "daily_call_cap": config.SHADOW_JUDGE_QWEN3_DAILY_CALL_CAP,
        }
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(control_row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _log_qwen3_shadow_state_once(
            "active", "qwen3 shadow ACTIVE: gemma decides, %s logged to %s",
            config.QWEN3_MODEL, config.SHADOW_JUDGE_LOG_PATH_QWEN3)
    except Exception as exc:
        if runtime is not None:
            try:
                runtime.close()
            except Exception:
                pass
        message = shadow_judge.redact(f"{type(exc).__name__}: {exc}")[:500]
        _log.warning("qwen3 shadow setup failed; running without it: %s", message)
        def metrics():
            values = resolved.metrics_fn()
            values["qwen3_shadow_status"] = "setup_failed"
            return values
        return ResolvedNewsJudge(resolved.callable, resolved.status, metrics, resolved.close_fn)

    def metrics():
        values = resolved.metrics_fn()
        try:
            qvalues = runtime.metrics()
        except Exception as exc:
            qvalues = {"shadow_status": "degraded", "shadow_worker_alive": False,
                      "shadow_last_error_type": type(exc).__name__}
        for key, value in qvalues.items():
            values[f"qwen3_{key}"] = value
        return values

    def close():
        # Run both lanes' closes concurrently (each bounded at SHADOW_JUDGE_SHUTDOWN_SEC)
        # so daemon shutdown stays ~2s worst case instead of doubling to ~4s from a
        # sequential join when both lanes are active.
        def _close_qwen3():
            try:
                runtime.close()
            except Exception:
                pass
        closer = threading.Thread(target=_close_qwen3, daemon=True)
        closer.start()
        try:
            resolved.close_fn()
        finally:
            closer.join(timeout=config.SHADOW_JUDGE_SHUTDOWN_SEC + 0.5)

    return ResolvedNewsJudge(runtime, resolved.status, metrics, close)


def resolve_news_judge_runtime(backend=None):
    """Resolve the full news-judge stack: gemma decides, with up to two independent,
    observed-only shadow lanes layered on top (haiku, then qwen3). See
    _resolve_haiku_layer and _wrap_with_qwen3_shadow for each layer's own contract."""
    return _wrap_with_qwen3_shadow(_resolve_haiku_layer(backend))


def resolve_news_judge(backend=None):
    """Compatibility resolver returning only the deciding callable."""
    return resolve_news_judge_runtime(backend).callable


def _prior_session_start(now_et):
    """Prior session close = the most recent weekday before today at 16:00 ET (RTH close). Catalysts
    released after prior close / overnight drive the premarket gap. Skips weekends, NOT holidays (a
    holiday just yields a harmless empty extra day; the window + fetch over-cover safely)."""
    d = now_et.astimezone(ET).date() - timedelta(days=1)
    while d.weekday() >= 5:                               # Sat=5, Sun=6 -> step back to Friday
        d -= timedelta(days=1)
    return datetime(d.year, d.month, d.day, 16, 0, tzinfo=ET)


def _window(now_et, premarket):
    """Premarket: [start, min(now,09:30)]. Intraday: [start, now_et] so post-09:30 catalysts are NOT
    dropped. start = today 04:00 ET, or -- when CATALYST_PRIOR_SESSION_ENABLED -- the
    prior session close (prior weekday 16:00 ET) so prior-evening/overnight catalysts are included."""
    start_et, end_pm = et_window(now_et)                 # start_et is ET-aware
    end_et = end_pm if premarket else now_et.astimezone(start_et.tzinfo)
    if config.CATALYST_PRIOR_SESSION_ENABLED:
        start_et = _prior_session_start(now_et)
    return start_et, end_et


def judge_candidate(candidate, *, news_client, now_et, premarket, news_judge=None, evaluate=evaluate_news,
                    edgar=None):
    news_judge = news_judge or resolve_news_judge()
    start_et, end_et = _window(now_et, premarket)
    frm = start_et.date().isoformat()                    # == today when the prior-session flag is OFF
    to = now_et.astimezone(start_et.tzinfo).date().isoformat()
    if config.SCANNER_JUDGE_SHADOW or config.SCANNER_JUDGE_SHADOW_QWEN3:
        # Shadow rows (either lane) are useless for the A/B unless they can be joined to the trade
        # ledger, and evaluate_news fixes the judge signature to (headline, summary).
        # Guarded + flag-gated so the gemma-only path is untouched.
        try:
            shadow_judge.set_current_ticker(candidate.ticker)
        except Exception:
            _log.debug("judge: could not stamp shadow ticker", exc_info=True)
    try:
        items = news_client.fetch_company_news(candidate.ticker, frm, to)
        in_window = [it for it in items if start_et <= it.published.astimezone(start_et.tzinfo) <= end_et]
        now_utc = now_et.astimezone(timezone.utc)
        verdict = evaluate(candidate, in_window, news_judge=news_judge, now=now_utc, raise_systemic=True,
                           edgar=edgar)
    except ScannerError as exc:                          # Finnhub OR EDGAR systemic -> unjudged
        total, loud = _record_systemic_failure()
        if loud:
            _log.warning("judge: systemic failure for %s: %s -- left unjudged "
                         "(systemic_failures=%d since start; suppressing until the next %d)",
                         candidate.ticker, exc, total, _SYSTEMIC_LOG_EVERY)
        else:
            _log.debug("judge: systemic failure for %s: %s -- left unjudged (systemic_failures=%d)",
                       candidate.ticker, exc, total)
        return None, None
    _record_judge_success()
    most_recent = max(in_window, key=lambda it: it.published, default=None)
    headline = most_recent.headline if most_recent is not None else None
    verdict = replace(verdict, judged_at=now_utc, headline=headline)
    max_item_ts = most_recent.published if most_recent is not None else None
    return verdict, max_item_ts

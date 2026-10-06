"""Nonblocking, observed-only shadow news judging.

The deciding thread calls the primary judge synchronously, then submits an immutable
snapshot to a bounded daemon queue. Shadow availability therefore cannot alter or
delay a scanner verdict.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import queue
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from zoneinfo import ZoneInfo

from momentum_scanner import config

_log = logging.getLogger(__name__)
_ET = ZoneInfo("America/New_York")

MAX_LOGGED_HEADLINE = 300
MAX_LOGGED_SUMMARY = 8000
MAX_LOGGED_ERROR = 500
_WARNING_CATEGORIES = (
    "capture_failed",
    "daily_cap",
    "queue_full",
    "abandoned_backlog",
)

_KEY_RE = re.compile(r"sk-ant-[A-Za-z0-9_\-]+")


def redact(text: str) -> str:
    """Strip Anthropic key material from text that could reach durable logs."""
    text = _KEY_RE.sub("sk-ant-***", text or "")
    live = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    return text.replace(live, "***") if live else text


_redact = redact

_current_ticker = None


def set_current_ticker(ticker):
    """Set the ticker used by compatibility callers that have no ticker callback."""
    global _current_ticker
    _current_ticker = ticker


def _agreement(primary, shadow):
    if not isinstance(primary, dict) or not isinstance(shadow, dict):
        return None
    return primary.get("label") == shadow.get("label")


class AdmissionResult(Enum):
    ACCEPTED = "accepted"
    DAILY_CAP = "daily_cap"
    QUEUE_FULL = "queue_full"


class DailyAdmission:
    """Atomically enforce per-ET-date caps while handing jobs to a bounded queue."""

    def __init__(self, cap, lock=None):
        self._cap = cap
        self._lock = lock if lock is not None else threading.Lock()
        self._accepted = defaultdict(int)

    def try_enqueue(self, et_date, enqueue):
        with self._lock:
            if self._accepted[et_date] >= self._cap:
                return AdmissionResult.DAILY_CAP
            try:
                enqueue()
            except queue.Full:
                return AdmissionResult.QUEUE_FULL
            self._accepted[et_date] += 1
            return AdmissionResult.ACCEPTED

    def accepted(self, et_date):
        with self._lock:
            return self._accepted[et_date]


@dataclass(frozen=True)
class ShadowJob:
    shadow_seq: int
    primary_ts: str
    ticker: str | None
    headline: str
    summary: str
    primary: object
    primary_backend: str
    shadow_backend: str


class ShadowJudgeRuntime:
    def __init__(
        self,
        primary,
        shadow,
        *,
        log_path,
        ticker_fn=None,
        error_fn=None,
        primary_backend="gemma",
        shadow_backend="haiku",
        queue_cap=config.SHADOW_JUDGE_QUEUE_CAP,
        daily_call_cap=config.SHADOW_JUDGE_DAILY_CALL_CAP,
        log_every_drops=config.SHADOW_JUDGE_LOG_EVERY_DROPS,
        primary_clock=None,
        worker_clock=None,
    ):
        self._primary = primary
        self._shadow = shadow
        self._path = Path(log_path)
        self._ticker_fn = ticker_fn
        self._error_fn = error_fn
        self._primary_backend = primary_backend
        self._shadow_backend = shadow_backend
        self._queue = queue.Queue(maxsize=queue_cap)
        self._counter_lock = threading.RLock()
        self._admission = DailyAdmission(daily_call_cap, lock=self._counter_lock)
        self._log_every_drops = log_every_drops
        self._primary_clock = primary_clock or (lambda: datetime.now(timezone.utc))
        self._worker_clock = worker_clock or (lambda: datetime.now(timezone.utc))
        self._worker = None
        self._closed = threading.Event()
        self._fatal_error = None
        self._fatal_error_type = None
        self._close_backlog_warning_queued = False
        self._seq = 0
        self._captured_dates = {}
        self._enqueued_mono = {}
        self._counters = defaultdict(int)
        self._warning_wakeup = queue.Queue(maxsize=1)
        self._pending_warnings = {
            reason: {"first": None, "latest": None}
            for reason in _WARNING_CATEGORIES
        }

    def start(self):
        with self._counter_lock:
            if self._worker is not None:
                return
            # Named after the shadow backend (not a fixed "haiku-shadow") so two lanes
            # running concurrently (e.g. haiku + qwen3) show up as distinct threads in
            # threading.enumerate()/thread dumps instead of two identically-named ones.
            thread_name = f"shadow-{self._shadow_backend}"
            self._worker = threading.Thread(target=self._run, name=thread_name, daemon=True)
            self._worker.start()

    def __call__(self, headline: str, summary: str) -> object:
        verdict = self._primary(headline, summary)
        with self._counter_lock:
            unavailable = self._unavailable_reason_locked()
            if unavailable is not None:
                self._record_unavailable_drop_locked(unavailable)
                return verdict
        try:
            job = self._capture_job(headline, summary, verdict)
            self._submit_nowait(job)
        except Exception as exc:
            self._record_drop("capture_failed", exc)
        return verdict

    def _capture_job(self, headline, summary, verdict):
        timestamp = self._primary_clock()
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("primary_clock must return an aware UTC datetime")
        primary_ts = timestamp.astimezone(timezone.utc).isoformat()
        ticker = self._ticker_fn() if self._ticker_fn is not None else _current_ticker
        with self._counter_lock:
            self._seq += 1
            shadow_seq = self._seq
        job = ShadowJob(
            shadow_seq=shadow_seq,
            primary_ts=primary_ts,
            ticker=ticker,
            headline=(headline or "")[:MAX_LOGGED_HEADLINE],
            summary=(summary or "")[:MAX_LOGGED_SUMMARY],
            primary=copy.deepcopy(verdict),
            primary_backend=self._primary_backend,
            shadow_backend=self._shadow_backend,
        )
        with self._counter_lock:
            self._captured_dates[shadow_seq] = timestamp.astimezone(_ET).date()
        return job

    def _submit_nowait(self, job):
        submitted_at = time.monotonic()
        with self._counter_lock:
            et_date = self._captured_dates[job.shadow_seq]

        def enqueue():
            self._queue.put_nowait(job)
            self._enqueued_mono[job.shadow_seq] = submitted_at
            self._counters["shadow_submitted"] += 1

        with self._counter_lock:
            unavailable = self._unavailable_reason_locked()
            if unavailable is not None:
                self._captured_dates.pop(job.shadow_seq, None)
                self._record_unavailable_drop_locked(unavailable)
                return
            result = self._admission.try_enqueue(et_date, enqueue)
            self._captured_dates.pop(job.shadow_seq, None)
        if result is AdmissionResult.ACCEPTED:
            return
        if result is AdmissionResult.DAILY_CAP:
            self._record_drop("daily_cap")
        else:
            self._record_drop("queue_full")

    def _record_drop(self, reason, exc=None):
        with self._counter_lock:
            self._counters["shadow_dropped"] += 1
            self._counters[f"shadow_dropped_{reason}"] += 1
            drops = self._counters["shadow_dropped"]
            if drops == 1 or drops % self._log_every_drops == 0:
                detail = f": {redact(f'{type(exc).__name__}: {exc}')}" if exc else ""
                pending = self._pending_warnings[reason]
                payload = (reason, drops, detail[:MAX_LOGGED_ERROR])
                if pending["first"] is None:
                    pending["first"] = payload
                else:
                    pending["latest"] = payload
            else:
                return
        try:
            self._warning_wakeup.put_nowait(None)
        except queue.Full:
            pass

    def _unavailable_reason_locked(self):
        if self._closed.is_set():
            return "closed"
        if self._fatal_error is not None:
            return "worker_dead"
        return None

    def _record_unavailable_drop_locked(self, reason):
        self._counters["shadow_dropped"] += 1
        self._counters[f"shadow_dropped_{reason}"] += 1

    def _run(self):
        job = None
        try:
            while not self._closed.is_set():
                self._emit_pending_drop_warnings()
                if self._closed.is_set():
                    break
                try:
                    job = self._queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                try:
                    self._process_job(job)
                finally:
                    self._queue.task_done()
                job = None
            self._emit_pending_drop_warnings(force=True)
        except BaseException as exc:
            self._record_worker_fatal(exc, job)

    def _record_worker_fatal(self, exc, job):
        message = redact(f"{type(exc).__name__}: {exc}")[:MAX_LOGGED_ERROR]
        with self._counter_lock:
            if self._fatal_error is not None:
                return
            self._fatal_error = message
            self._fatal_error_type = type(exc).__name__[:MAX_LOGGED_ERROR]
            self._counters["shadow_fatal_errors"] += 1
            self._counters["shadow_abandoned_backlog"] = self._queue.qsize()
            if job is not None:
                self._enqueued_mono.pop(job.shadow_seq, None)
        try:
            _log.warning("shadow worker fatal: %s", message)
        except BaseException:
            pass

    def _emit_pending_drop_warnings(self, *, force=False):
        if not force:
            try:
                self._warning_wakeup.get_nowait()
            except queue.Empty:
                return
        with self._counter_lock:
            if force:
                try:
                    self._warning_wakeup.get_nowait()
                except queue.Empty:
                    pass
            payloads = []
            for pending in self._pending_warnings.values():
                if pending["first"] is not None:
                    payloads.append(pending["first"])
                if pending["latest"] is not None:
                    payloads.append(pending["latest"])
                pending["first"] = None
                pending["latest"] = None
        for reason, drops, detail in payloads:
            if reason == "abandoned_backlog":
                _log.warning("shadow close abandoned backlog: %d", drops)
            else:
                _log.warning("shadow drop (%s, count=%s)%s", reason, drops, detail)

    def _process_job(self, job):
        started_dt = self._aware_worker_now()
        started_mono = time.monotonic()
        shadow_verdict = None
        shadow_error = None
        try:
            shadow_verdict = self._shadow(job.headline, job.summary)
        except Exception as exc:
            shadow_error = redact(f"{type(exc).__name__}: {exc}")
        if shadow_verdict is None and shadow_error is None and self._error_fn is not None:
            try:
                reported = self._error_fn()
                shadow_error = redact(reported) if reported else None
            except Exception as exc:
                message = redact(f"{type(exc).__name__}: {exc}")[:MAX_LOGGED_ERROR]
                _log.warning("shadow error_fn failed: %s", message)
        completed_dt = self._aware_worker_now()
        elapsed_ms = round((time.monotonic() - started_mono) * 1000, 1)
        with self._counter_lock:
            queued_at = self._enqueued_mono.pop(job.shadow_seq, started_mono)
            self._counters["shadow_completed"] += 1
        row = {
            "record_type": "comparison",
            "ts": job.primary_ts,
            "primary_ts": job.primary_ts,
            "shadow_started_ts": started_dt.isoformat(),
            "shadow_completed_ts": completed_dt.isoformat(),
            "shadow_seq": job.shadow_seq,
            "ticker": job.ticker,
            "headline": job.headline,
            "primary_backend": job.primary_backend,
            "shadow_backend": job.shadow_backend,
            "primary": job.primary,
            "shadow": shadow_verdict,
            "agree": _agreement(job.primary, shadow_verdict),
            "shadow_error": shadow_error,
            "queue_ms": round((started_mono - queued_at) * 1000, 1),
            "shadow_ms": elapsed_ms,
        }
        try:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
                fh.flush()
        except Exception as exc:
            self._record_write_failure(exc)

    def _aware_worker_now(self):
        timestamp = self._worker_clock()
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("worker_clock must return an aware UTC datetime")
        return timestamp.astimezone(timezone.utc)

    def _record_write_failure(self, exc):
        with self._counter_lock:
            self._counters["shadow_write_failures"] += 1
            failures = self._counters["shadow_write_failures"]
        if failures == 1 or failures % self._log_every_drops == 0:
            _log.warning("shadow log write failed (count=%s): %s", failures, redact(f"{type(exc).__name__}: {exc}"))

    def metrics(self):
        with self._counter_lock:
            metrics = dict(self._counters)
            fatal_error = self._fatal_error
            fatal_error_type = self._fatal_error_type
        for key in (
            "shadow_submitted", "shadow_completed", "shadow_dropped",
            "shadow_dropped_daily_cap", "shadow_dropped_queue_full",
            "shadow_dropped_capture_failed", "shadow_dropped_worker_dead",
            "shadow_dropped_closed", "shadow_write_failures",
            "shadow_fatal_errors", "shadow_abandoned_backlog",
        ):
            metrics.setdefault(key, 0)
        metrics["shadow_queue_depth"] = self._queue.qsize()
        metrics["shadow_worker_alive"] = bool(self._worker and self._worker.is_alive())
        metrics["shadow_status"] = "degraded" if fatal_error is not None else "active"
        metrics["shadow_last_error_type"] = fatal_error_type
        metrics["shadow_last_error"] = fatal_error
        return metrics

    def close(self, timeout: float = 2.0):
        signal_warning = False
        with self._counter_lock:
            self._closed.set()
            abandoned_backlog = max(
                self._counters["shadow_abandoned_backlog"],
                self._queue.qsize(),
            )
            self._counters["shadow_abandoned_backlog"] = abandoned_backlog
            if abandoned_backlog > 0 and not self._close_backlog_warning_queued:
                self._pending_warnings["abandoned_backlog"]["first"] = (
                    "abandoned_backlog",
                    abandoned_backlog,
                    "",
                )
                self._close_backlog_warning_queued = True
                signal_warning = True
        if signal_warning:
            try:
                self._warning_wakeup.put_nowait(None)
            except queue.Full:
                pass
        worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            # Leave a small scheduling margin so the public 2.0-second shutdown
            # bound remains true even on a contended Windows interpreter.
            worker.join(timeout=max(0.0, timeout - 0.05))


def iter_comparison_rows(path):
    """Stream valid comparison rows while rejecting ambiguous control records."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            record_type = row.get("record_type")
            if record_type is None or record_type == "comparison":
                yield row
            elif record_type == "shadow_start":
                continue
            else:
                raise ValueError(f"unexpected shadow record type: {record_type!r}")


def make_shadow_news_judge(primary, shadow, *, log_path, ticker_fn=None, error_fn=None,
                           primary_backend="gemma", shadow_backend="haiku"):
    """Compatibility constructor returning a started nonblocking runtime."""
    runtime = ShadowJudgeRuntime(
        primary,
        shadow,
        log_path=log_path,
        ticker_fn=ticker_fn,
        error_fn=error_fn,
        primary_backend=primary_backend,
        shadow_backend=shadow_backend,
    )
    runtime.start()
    return runtime

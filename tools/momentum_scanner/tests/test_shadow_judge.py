# tests/test_shadow_judge.py
"""Dual-log shadow judge (Haiku A/B).

THE LOAD-BEARING INVARIANT, and the reason this module exists: gemma keeps DECIDING.
The shadow backend is observed and logged, never consulted. Every test below that
asserts on a return value is asserting that invariant from a different angle -- a
shadow that raises, hangs, returns junk, or disagrees must still yield gemma's verdict
byte-for-byte, because a silent behavior change would alter the deciding path's
verdicts.
"""
import json
import logging
import threading
import time
from datetime import date, datetime, timezone

import pytest

from momentum_scanner import shadow_judge

GEMMA_REAL = {"label": "real", "confidence": 0.91, "is_fixed_price_buyout": False, "rationale": "FDA approval"}
HAIKU_PUMP = {"label": "pump", "confidence": 0.80, "is_fixed_price_buyout": False, "rationale": "promo language"}


def _primary(result=GEMMA_REAL, calls=None):
    def fn(headline, summary):
        if calls is not None:
            calls.append((headline, summary))
        return result
    return fn


def _rows(path):
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached before timeout")


def _wait_for_rows(path, count=1):
    _wait_until(lambda: len(_rows(path)) >= count)
    return _rows(path)


class ThreadSafeSequenceClock:
    def __init__(self, values):
        self._values = iter(values)
        self._lock = threading.Lock()

    def __call__(self):
        with self._lock:
            value = next(self._values)
        assert value.tzinfo is not None and value.utcoffset() == timezone.utc.utcoffset(value)
        return value


def _call_concurrently(runtime, count):
    barrier = threading.Barrier(count)
    errors = []

    def call(index):
        try:
            barrier.wait(timeout=2.0)
            runtime(f"headline-{index}", "summary")
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=call, args=(index,)) for index in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)
    assert not any(thread.is_alive() for thread in threads)
    assert not errors


# ---- the invariant: the primary verdict is returned untouched ----------------

def test_returns_primary_verdict_when_both_agree(tmp_path):
    log = tmp_path / "judge_shadow.jsonl"
    j = shadow_judge.make_shadow_news_judge(_primary(), _primary(GEMMA_REAL), log_path=log)
    assert j("ABCD wins FDA approval", "primary endpoint met") == GEMMA_REAL


def test_returns_primary_verdict_when_shadow_disagrees(tmp_path):
    # The whole point of the A/B: disagreement is DATA, not a vote. gemma still wins.
    log = tmp_path / "judge_shadow.jsonl"
    j = shadow_judge.make_shadow_news_judge(_primary(GEMMA_REAL), _primary(HAIKU_PUMP), log_path=log)
    assert j("ABCD wins FDA approval", "primary endpoint met") == GEMMA_REAL


def test_returns_primary_verdict_when_shadow_raises(tmp_path):
    log = tmp_path / "judge_shadow.jsonl"
    def boom(headline, summary):
        raise RuntimeError("anthropic 500")
    j = shadow_judge.make_shadow_news_judge(_primary(GEMMA_REAL), boom, log_path=log)
    assert j("ABCD wins FDA approval", "x") == GEMMA_REAL


def test_returns_primary_none_when_primary_fails(tmp_path):
    # A gemma failure must stay a gemma failure -- the shadow must NEVER rescue it into
    # a verdict, or the fail-closed contract in judge.py silently acquires a second opinion.
    log = tmp_path / "judge_shadow.jsonl"
    j = shadow_judge.make_shadow_news_judge(_primary(None), _primary(HAIKU_PUMP), log_path=log)
    assert j("ABCD wins FDA approval", "x") is None


def test_primary_receives_the_unmodified_arguments(tmp_path):
    log = tmp_path / "judge_shadow.jsonl"
    calls = []
    j = shadow_judge.make_shadow_news_judge(_primary(GEMMA_REAL, calls), _primary(HAIKU_PUMP), log_path=log)
    j("headline text", "summary text")
    assert calls == [("headline text", "summary text")]


def test_unwritable_log_path_does_not_break_the_verdict(tmp_path):
    # Fail-open on the OBSERVABILITY half only: losing a log row is acceptable,
    # losing a trading verdict is not.
    log = tmp_path / "no_such_dir" / "nested" / "judge_shadow.jsonl"
    j = shadow_judge.make_shadow_news_judge(_primary(GEMMA_REAL), _primary(HAIKU_PUMP), log_path=log)
    assert j("ABCD wins FDA approval", "x") == GEMMA_REAL


# ---- the dual log ------------------------------------------------------------

def test_logs_both_verdicts_on_one_row(tmp_path):
    log = tmp_path / "judge_shadow.jsonl"
    j = shadow_judge.make_shadow_news_judge(_primary(GEMMA_REAL), _primary(HAIKU_PUMP), log_path=log)
    j("ABCD wins FDA approval", "primary endpoint met")
    rows = _wait_for_rows(log)
    assert len(rows) == 1
    assert rows[0]["primary"] == GEMMA_REAL
    assert rows[0]["shadow"] == HAIKU_PUMP
    assert rows[0]["agree"] is False
    assert rows[0]["primary_backend"] == "gemma"
    assert rows[0]["shadow_backend"] == "haiku"


def test_agree_is_true_only_when_labels_match(tmp_path):
    # Agreement is on LABEL, not on the confidence float -- two models will never
    # agree to 2dp and a confidence-sensitive flag would report ~0% agreement forever.
    log = tmp_path / "judge_shadow.jsonl"
    other_conf = dict(GEMMA_REAL, confidence=0.55, rationale="different wording")
    j = shadow_judge.make_shadow_news_judge(_primary(GEMMA_REAL), _primary(other_conf), log_path=log)
    j("h", "s")
    assert _wait_for_rows(log)[0]["agree"] is True


def test_logs_shadow_error_and_null_shadow_when_shadow_raises(tmp_path):
    log = tmp_path / "judge_shadow.jsonl"
    def boom(headline, summary):
        raise RuntimeError("anthropic 500")
    j = shadow_judge.make_shadow_news_judge(_primary(GEMMA_REAL), boom, log_path=log)
    j("h", "s")
    row = _wait_for_rows(log)[0]
    assert row["shadow"] is None
    assert "anthropic 500" in row["shadow_error"]
    assert row["agree"] is None            # unknown, NOT False -- an outage is not a disagreement


def test_rows_append_across_calls(tmp_path):
    log = tmp_path / "judge_shadow.jsonl"
    j = shadow_judge.make_shadow_news_judge(_primary(GEMMA_REAL), _primary(HAIKU_PUMP), log_path=log)
    j("h1", "s1")
    j("h2", "s2")
    assert len(_wait_for_rows(log, 2)) == 2


def test_row_carries_headline_and_latency(tmp_path):
    log = tmp_path / "judge_shadow.jsonl"
    j = shadow_judge.make_shadow_news_judge(_primary(GEMMA_REAL), _primary(HAIKU_PUMP), log_path=log)
    j("ABCD wins FDA approval", "primary endpoint met")
    row = _wait_for_rows(log)[0]
    assert row["headline"] == "ABCD wins FDA approval"
    assert isinstance(row["shadow_ms"], (int, float)) and row["shadow_ms"] >= 0
    assert row["ts"].endswith("+00:00") or row["ts"].endswith("Z")


def test_headline_is_truncated(tmp_path):
    log = tmp_path / "judge_shadow.jsonl"
    j = shadow_judge.make_shadow_news_judge(_primary(GEMMA_REAL), _primary(HAIKU_PUMP), log_path=log)
    j("x" * 5000, "s")
    assert len(_wait_for_rows(log)[0]["headline"]) <= shadow_judge.MAX_LOGGED_HEADLINE


def test_ticker_is_recorded_when_supplied(tmp_path):
    # Without a ticker the log cannot be joined back to the outcome journal, which is
    # the entire analysis path for this A/B.
    log = tmp_path / "judge_shadow.jsonl"
    j = shadow_judge.make_shadow_news_judge(_primary(GEMMA_REAL), _primary(HAIKU_PUMP),
                                            log_path=log, ticker_fn=lambda: "ABCD")
    j("h", "s")
    assert _wait_for_rows(log)[0]["ticker"] == "ABCD"


def test_ticker_is_null_when_no_ticker_fn(tmp_path):
    log = tmp_path / "judge_shadow.jsonl"
    j = shadow_judge.make_shadow_news_judge(_primary(GEMMA_REAL), _primary(HAIKU_PUMP), log_path=log)
    j("h", "s")
    assert _wait_for_rows(log)[0]["ticker"] is None


def test_a_raising_ticker_fn_does_not_break_the_verdict(tmp_path):
    log = tmp_path / "judge_shadow.jsonl"
    def bad_ticker():
        raise KeyError("no ticker in scope")
    j = shadow_judge.make_shadow_news_judge(_primary(GEMMA_REAL), _primary(HAIKU_PUMP),
                                            log_path=log, ticker_fn=bad_ticker)
    assert j("h", "s") == GEMMA_REAL


# ---- failure attribution via error_fn ----------------------------------------
# A shadow that returns None without raising (haiku_news_judge swallows its own
# failures) is ambiguous. error_fn lets the backend hand over WHY, so an outage row
# is distinguishable from a genuine "the model produced junk" row.

def test_error_fn_supplies_the_reason_when_shadow_returns_none(tmp_path):
    log = tmp_path / "judge_shadow.jsonl"
    j = shadow_judge.make_shadow_news_judge(
        _primary(GEMMA_REAL), _primary(None), log_path=log,
        error_fn=lambda: "RuntimeError: anthropic overloaded")
    j("h", "s")
    row = _wait_for_rows(log)[0]
    assert row["shadow"] is None
    assert "anthropic overloaded" in row["shadow_error"]
    assert row["agree"] is None


def test_error_fn_is_not_consulted_when_the_shadow_succeeded(tmp_path):
    # A stale error from a previous call must never be stamped onto a good row.
    log = tmp_path / "judge_shadow.jsonl"
    calls = []
    def err():
        calls.append(1)
        return "stale error"
    j = shadow_judge.make_shadow_news_judge(_primary(GEMMA_REAL), _primary(HAIKU_PUMP),
                                            log_path=log, error_fn=err)
    j("h", "s")
    assert _wait_for_rows(log)[0]["shadow_error"] is None
    assert calls == []


def test_a_raising_error_fn_does_not_break_the_verdict(tmp_path):
    log = tmp_path / "judge_shadow.jsonl"
    j = shadow_judge.make_shadow_news_judge(
        _primary(GEMMA_REAL), _primary(None), log_path=log,
        error_fn=lambda: (_ for _ in ()).throw(RuntimeError("sink broke")))
    assert j("h", "s") == GEMMA_REAL


def test_error_fn_output_is_redacted(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SECRETVALUE123")
    log = tmp_path / "judge_shadow.jsonl"
    j = shadow_judge.make_shadow_news_judge(
        _primary(GEMMA_REAL), _primary(None), log_path=log,
        error_fn=lambda: "401 for key sk-ant-SECRETVALUE123")
    j("h", "s")
    _wait_for_rows(log)
    assert "SECRETVALUE123" not in log.read_text(encoding="utf-8")


# ---- ticker join -------------------------------------------------------------

def test_current_ticker_is_stamped_on_the_row(tmp_path):
    # Set by judge_candidate so shadow rows can be joined to the outcome journal.
    log = tmp_path / "judge_shadow.jsonl"
    shadow_judge.set_current_ticker("ABCD")
    try:
        j = shadow_judge.make_shadow_news_judge(_primary(GEMMA_REAL), _primary(HAIKU_PUMP),
                                                log_path=log)
        j("h", "s")
        assert _wait_for_rows(log)[0]["ticker"] == "ABCD"
    finally:
        shadow_judge.set_current_ticker(None)


def test_explicit_ticker_fn_wins_over_the_module_default(tmp_path):
    log = tmp_path / "judge_shadow.jsonl"
    shadow_judge.set_current_ticker("ABCD")
    try:
        j = shadow_judge.make_shadow_news_judge(_primary(GEMMA_REAL), _primary(HAIKU_PUMP),
                                                log_path=log, ticker_fn=lambda: "WXYZ")
        j("h", "s")
        assert _wait_for_rows(log)[0]["ticker"] == "WXYZ"
    finally:
        shadow_judge.set_current_ticker(None)


# ---- the shadow must never be able to stall the trading loop ------------------

def test_shadow_call_is_not_awaited_beyond_its_own_failure(tmp_path, caplog):
    log = tmp_path / "judge_shadow.jsonl"
    def boom(headline, summary):
        raise TimeoutError("read timeout")
    j = shadow_judge.make_shadow_news_judge(_primary(GEMMA_REAL), boom, log_path=log)
    with caplog.at_level(logging.DEBUG, logger="momentum_scanner.shadow_judge"):
        assert j("h", "s") == GEMMA_REAL
    assert _wait_for_rows(log)[0]["shadow"] is None


def test_no_api_key_material_reaches_the_log(tmp_path, monkeypatch):
    # A key in an SDK exception string must not be persisted to an append-only file.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SECRETVALUE123")
    log = tmp_path / "judge_shadow.jsonl"
    def boom(headline, summary):
        raise RuntimeError("401 unauthorized for key sk-ant-SECRETVALUE123")
    j = shadow_judge.make_shadow_news_judge(_primary(GEMMA_REAL), boom, log_path=log)
    j("h", "s")
    _wait_for_rows(log)
    assert "SECRETVALUE123" not in log.read_text(encoding="utf-8")


# ---- nonblocking runtime -----------------------------------------------------

def test_blocked_shadow_cannot_delay_or_replace_primary(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    primary = {"label": "real"}

    def blocked(headline, summary):
        entered.set()
        release.wait(10)
        return {"label": "pump"}

    runtime = shadow_judge.make_shadow_news_judge(
        lambda h, s: primary, blocked, log_path=tmp_path / "shadow.jsonl")
    returned = threading.Event()
    result = {}

    def call_runtime():
        result["verdict"] = runtime("headline", "summary")
        returned.set()

    caller = threading.Thread(target=call_runtime)
    try:
        caller.start()
        assert returned.wait(1.0)
        assert result["verdict"] is primary
        assert entered.wait(1.0)
    finally:
        release.set()
        caller.join(timeout=1.0)
        assert not caller.is_alive()
        runtime.close()


def test_submission_captures_ticker_and_primary_snapshot(tmp_path):
    release = threading.Event()
    primary = {"label": "real", "nested": {"confidence": 0.9}}
    ticker = {"value": "ABCD"}
    runtime = shadow_judge.make_shadow_news_judge(
        lambda h, s: primary,
        lambda h, s: (release.wait(1.0), {"label": "real"})[1],
        log_path=tmp_path / "shadow.jsonl",
        ticker_fn=lambda: ticker["value"],
    )
    try:
        runtime("headline", "x" * 9000)
        ticker["value"] = "CHANGED"
        primary["nested"]["confidence"] = 0.1
        release.set()
        _wait_until(lambda: runtime.metrics()["shadow_completed"] == 1)
        row = _wait_for_rows(tmp_path / "shadow.jsonl")[0]
        assert row["ticker"] == "ABCD"
        assert row["primary"]["nested"]["confidence"] == 0.9
    finally:
        runtime.close()


def test_queue_full_drops_newest_without_waiting(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    def blocked(headline, summary):
        entered.set()
        release.wait(10)
        return {"label": "real"}

    runtime = shadow_judge.ShadowJudgeRuntime(
        lambda h, s: {"label": "real"}, blocked,
        log_path=tmp_path / "shadow.jsonl", queue_cap=20, daily_call_cap=250)
    runtime.start()
    try:
        runtime("inflight", "s")
        assert entered.wait(1.0)
        for index in range(20):
            runtime(f"queued-{index}", "s")
        before = time.monotonic()
        runtime("dropped", "s")
        assert time.monotonic() - before < 0.1
        metrics = runtime.metrics()
        assert metrics["shadow_queue_depth"] == 20
        assert metrics["shadow_dropped"] == 1
    finally:
        release.set()
        runtime.close()


def test_daily_cap_is_atomic_and_resets_on_next_et_date(tmp_path):
    primary_clock = ThreadSafeSequenceClock(
        [datetime(2026, 8, 7, 19, 59, 59, tzinfo=timezone.utc)] * 251
        + [datetime(2026, 8, 8, 4, 0, 0, tzinfo=timezone.utc)]
    )
    runtime = shadow_judge.ShadowJudgeRuntime(
        lambda h, s: {"label": "real"},
        lambda h, s: {"label": "real"},
        log_path=tmp_path / "shadow.jsonl",
        queue_cap=300,
        daily_call_cap=250,
        primary_clock=primary_clock,
    )
    runtime.start()
    try:
        _call_concurrently(runtime, count=251)
        _wait_until(lambda: runtime.metrics()["shadow_completed"] == 250)
        assert runtime.metrics()["shadow_dropped_daily_cap"] == 1
        runtime("next-et-day", "summary")
        _wait_until(lambda: runtime.metrics()["shadow_completed"] == 251)
    finally:
        runtime.close()


def test_daily_cap_cannot_reopen_when_old_date_arrives_after_new_date():
    admission = shadow_judge.DailyAdmission(cap=250)
    old_date = date(2026, 8, 7)
    new_date = date(2026, 8, 8)
    assert all(
        admission.try_enqueue(old_date, lambda: None) is shadow_judge.AdmissionResult.ACCEPTED
        for _ in range(250)
    )
    assert admission.try_enqueue(new_date, lambda: None) is shadow_judge.AdmissionResult.ACCEPTED
    assert admission.try_enqueue(old_date, lambda: None) is shadow_judge.AdmissionResult.DAILY_CAP
    assert admission.accepted(old_date) == 250
    assert admission.accepted(new_date) == 1


def test_comparison_reader_excludes_control_and_accepts_legacy(tmp_path):
    path = tmp_path / "shadow.jsonl"
    path.write_text(
        "\n".join((
            '{"record_type":"shadow_start"}',
            '{"record_type":"comparison","shadow_seq":2}',
            '{"shadow_seq":1}',
        )) + "\n",
        encoding="ascii",
    )
    assert [row["shadow_seq"] for row in shadow_judge.iter_comparison_rows(path)] == [2, 1]


def test_accepted_jobs_are_processed_fifo(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    called = []

    def blocked_then_record(headline, summary):
        called.append(headline)
        entered.set()
        release.wait(10)
        return {"label": "real"}

    runtime = shadow_judge.make_shadow_news_judge(
        lambda h, s: {"label": "real"}, blocked_then_record,
        log_path=tmp_path / "shadow.jsonl")
    try:
        runtime("first", "s")
        assert entered.wait(1.0)
        runtime("second", "s")
        runtime("third", "s")
        release.set()
        _wait_until(lambda: runtime.metrics()["shadow_completed"] == 3)
        assert called == ["first", "second", "third"]
    finally:
        runtime.close()


def test_capture_deep_copy_failure_drops_shadow_without_changing_primary(tmp_path):
    class CannotCopy:
        def __deepcopy__(self, memo):
            raise TypeError("not copyable")

    verdict = CannotCopy()
    shadow_called = threading.Event()
    runtime = shadow_judge.make_shadow_news_judge(
        lambda h, s: verdict,
        lambda h, s: shadow_called.set(),
        log_path=tmp_path / "shadow.jsonl")
    try:
        assert runtime("headline", "summary") is verdict
        _wait_until(lambda: runtime.metrics()["shadow_dropped_capture_failed"] == 1)
        assert not shadow_called.wait(0.05)
    finally:
        runtime.close()


def test_dead_worker_does_not_change_primary_result(tmp_path):
    primary = {"label": "real"}
    runtime = shadow_judge.make_shadow_news_judge(
        lambda h, s: primary, lambda h, s: {"label": "real"}, log_path=tmp_path / "shadow.jsonl")
    try:
        runtime.close()
        assert not runtime.metrics()["shadow_worker_alive"]
        assert runtime("headline", "summary") is primary
    finally:
        runtime.close()


def test_worker_fatal_degrades_and_post_fatal_calls_return_primary_without_logging_delay(
        tmp_path, monkeypatch):
    class FatalShadow(BaseException):
        pass

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-CANARYSECRET")
    warning_entered = threading.Event()
    warning_release = threading.Event()
    returned = threading.Event()
    warning_rows = []
    primary = {"label": "real"}
    result = {}

    def fatal_shadow(headline, summary):
        raise FatalShadow("fatal sk-ant-CANARYSECRET")

    def blocked_warning(message, *args, **kwargs):
        warning_rows.append((message % args, kwargs))
        warning_entered.set()
        warning_release.wait(10)

    monkeypatch.setattr(shadow_judge._log, "warning", blocked_warning)
    runtime = shadow_judge.make_shadow_news_judge(
        lambda h, s: primary, fatal_shadow, log_path=tmp_path / "shadow.jsonl")

    def call_after_fatal():
        result["verdict"] = runtime("after-fatal", "summary")
        returned.set()

    caller = threading.Thread(target=call_after_fatal)
    try:
        assert runtime("fatal", "summary") is primary
        assert warning_entered.wait(1.0)
        caller.start()
        assert returned.wait(1.0)
        assert result["verdict"] is primary
        metrics = runtime.metrics()
        assert metrics["shadow_status"] == "degraded"
        assert metrics["shadow_fatal_errors"] == 1
        assert metrics["shadow_dropped_worker_dead"] == 1
        assert metrics["shadow_submitted"] == 1
        assert metrics["shadow_last_error_type"] == "FatalShadow"
        assert "CANARYSECRET" not in metrics["shadow_last_error"]
        assert len(metrics["shadow_last_error"]) <= shadow_judge.MAX_LOGGED_ERROR
    finally:
        warning_release.set()
        if caller.ident is not None:
            caller.join(timeout=1.0)
            assert not caller.is_alive()
        runtime.close()
    assert len(warning_rows) == 1
    assert "CANARYSECRET" not in warning_rows[0][0]
    assert len(warning_rows[0][0]) <= shadow_judge.MAX_LOGGED_ERROR + 50
    assert warning_rows[0][1].get("exc_info") in (None, False)


def test_close_abandons_bounded_backlog_and_rejects_new_submissions(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    returned = threading.Event()
    primary = {"label": "real"}
    result = {}

    def blocked_shadow(headline, summary):
        entered.set()
        release.wait(10)
        return {"label": "real"}

    runtime = shadow_judge.ShadowJudgeRuntime(
        lambda h, s: primary, blocked_shadow,
        log_path=tmp_path / "shadow.jsonl", queue_cap=2, daily_call_cap=10)
    runtime.start()

    def call_after_close():
        result["verdict"] = runtime("after-close", "summary")
        returned.set()

    caller = threading.Thread(target=call_after_close)
    try:
        runtime("inflight", "summary")
        assert entered.wait(1.0)
        runtime("queued-1", "summary")
        runtime("queued-2", "summary")
        runtime.close(timeout=0.0)
        caller.start()
        assert returned.wait(1.0)
        assert result["verdict"] is primary
        metrics = runtime.metrics()
        assert metrics["shadow_abandoned_backlog"] == 2
        assert metrics["shadow_dropped_closed"] == 1
        assert metrics["shadow_submitted"] == 3
        assert metrics["shadow_queue_depth"] == 2
    finally:
        release.set()
        caller.join(timeout=1.0)
        assert not caller.is_alive()
        runtime.close(timeout=0.0)


def test_close_warns_once_with_count_only_without_waiting_on_provider_or_logger(
        tmp_path, monkeypatch):
    provider_entered = threading.Event()
    provider_release = threading.Event()
    warning_entered = threading.Event()
    warning_release = threading.Event()
    close_returned = threading.Event()
    warning_rows = []

    def blocked_shadow(headline, summary):
        provider_entered.set()
        provider_release.wait(10)
        return {"label": "real"}

    def blocked_warning(message, *args, **kwargs):
        warning_rows.append((message % args, kwargs))
        warning_entered.set()
        warning_release.wait(10)

    monkeypatch.setattr(shadow_judge._log, "warning", blocked_warning)
    runtime = shadow_judge.ShadowJudgeRuntime(
        lambda h, s: {"label": "real"},
        blocked_shadow,
        log_path=tmp_path / "shadow.jsonl",
        queue_cap=2,
        daily_call_cap=10,
    )
    runtime.start()

    def close_now():
        runtime.close(timeout=0.0)
        close_returned.set()

    closer = threading.Thread(target=close_now)
    try:
        runtime("inflight-private-content", "secret summary")
        assert provider_entered.wait(1.0)
        runtime("queued-private-content-1", "secret summary")
        runtime("queued-private-content-2", "secret summary")
        closer.start()
        assert close_returned.wait(1.0)
        assert not warning_entered.is_set()
        provider_release.set()
        assert warning_entered.wait(1.0)
        assert len(warning_rows) == 1
        assert warning_rows[0][0] == "shadow close abandoned backlog: 2"
        assert "private-content" not in warning_rows[0][0]
        assert "secret" not in warning_rows[0][0]
        assert warning_rows[0][1].get("exc_info") in (None, False)
    finally:
        provider_release.set()
        warning_release.set()
        if closer.ident is not None:
            closer.join(timeout=1.0)
            assert not closer.is_alive()
        runtime.close(timeout=0.0)
    _wait_until(lambda: not runtime.metrics()["shadow_worker_alive"])
    assert len(warning_rows) == 1


def test_close_warning_cannot_be_lost_between_snapshot_and_wakeup(
        tmp_path, monkeypatch):
    provider_entered = threading.Event()
    provider_release = threading.Event()
    wakeup_entered = threading.Event()
    wakeup_release = threading.Event()
    final_drain_entered = threading.Event()
    warning_entered = threading.Event()
    warning_rows = []

    def blocked_shadow(headline, summary):
        provider_entered.set()
        provider_release.wait(10)
        return {"label": "real"}

    monkeypatch.setattr(
        shadow_judge._log,
        "warning",
        lambda message, *args, **kwargs: (
            warning_rows.append((message % args, kwargs)),
            warning_entered.set(),
        ),
    )
    runtime = shadow_judge.ShadowJudgeRuntime(
        lambda h, s: {"label": "real"},
        blocked_shadow,
        log_path=tmp_path / "shadow.jsonl",
        queue_cap=1,
        daily_call_cap=10,
    )
    original_put_nowait = runtime._warning_wakeup.put_nowait
    original_emit = runtime._emit_pending_drop_warnings

    def delayed_put_nowait(item):
        wakeup_entered.set()
        wakeup_release.wait(10)
        return original_put_nowait(item)

    def observed_emit(*args, **kwargs):
        if runtime._closed.is_set():
            final_drain_entered.set()
        return original_emit(*args, **kwargs)

    monkeypatch.setattr(runtime._warning_wakeup, "put_nowait", delayed_put_nowait)
    monkeypatch.setattr(runtime, "_emit_pending_drop_warnings", observed_emit)
    runtime.start()
    closer = threading.Thread(target=lambda: runtime.close(timeout=0.0))
    try:
        runtime("inflight", "summary")
        assert provider_entered.wait(1.0)
        runtime("queued", "summary")
        closer.start()
        assert wakeup_entered.wait(1.0)
        provider_release.set()
        assert final_drain_entered.wait(1.0)
        wakeup_release.set()
        closer.join(timeout=1.0)
        assert not closer.is_alive()
        assert warning_entered.wait(1.0)
        assert warning_rows == [("shadow close abandoned backlog: 1", {})]
        _wait_until(lambda: not runtime.metrics()["shadow_worker_alive"])
    finally:
        provider_release.set()
        wakeup_release.set()
        if closer.ident is not None:
            closer.join(timeout=1.0)


def test_close_rechecks_closed_after_prior_warning_before_dequeue(
        tmp_path, monkeypatch):
    fail_capture = True
    prior_warning_entered = threading.Event()
    prior_warning_release = threading.Event()
    close_returned = threading.Event()
    provider_calls = []
    warning_rows = []

    def ticker():
        nonlocal fail_capture
        if fail_capture:
            fail_capture = False
            raise ValueError("capture canary")
        return "TEST"

    def shadow_provider(headline, summary):
        provider_calls.append((headline, summary))
        return {"label": "real"}

    def blocked_warning(message, *args, **kwargs):
        rendered = message % args
        warning_rows.append((rendered, kwargs))
        if rendered.startswith("shadow drop"):
            prior_warning_entered.set()
            prior_warning_release.wait(10)

    monkeypatch.setattr(shadow_judge._log, "warning", blocked_warning)
    runtime = shadow_judge.ShadowJudgeRuntime(
        lambda h, s: {"label": "real"},
        shadow_provider,
        log_path=tmp_path / "shadow.jsonl",
        ticker_fn=ticker,
        queue_cap=3,
        daily_call_cap=10,
    )
    runtime.start()

    def close_now():
        runtime.close(timeout=0.0)
        close_returned.set()

    closer = threading.Thread(target=close_now)
    try:
        runtime("drop-trigger", "summary")
        assert prior_warning_entered.wait(1.0)
        for index in range(3):
            runtime(f"queued-{index}", "summary")
        closer.start()
        assert close_returned.wait(1.0)
        assert runtime.metrics()["shadow_abandoned_backlog"] == 3
        prior_warning_release.set()
        _wait_until(lambda: not runtime.metrics()["shadow_worker_alive"])
        metrics = runtime.metrics()
        assert provider_calls == []
        assert metrics["shadow_queue_depth"] == 3
        assert metrics["shadow_abandoned_backlog"] == 3
        assert [row[0] for row in warning_rows].count(
            "shadow close abandoned backlog: 3") == 1
    finally:
        prior_warning_release.set()
        if closer.ident is not None:
            closer.join(timeout=1.0)
            assert not closer.is_alive()


def test_close_with_zero_backlog_does_not_warn(tmp_path, monkeypatch):
    warning_rows = []
    monkeypatch.setattr(
        shadow_judge._log, "warning",
        lambda message, *args, **kwargs: warning_rows.append(message % args),
    )
    runtime = shadow_judge.make_shadow_news_judge(
        lambda h, s: {"label": "real"},
        lambda h, s: {"label": "real"},
        log_path=tmp_path / "shadow.jsonl",
    )
    runtime.close(timeout=0.0)
    _wait_until(lambda: not runtime.metrics()["shadow_worker_alive"])
    assert runtime.metrics()["shadow_abandoned_backlog"] == 0
    assert warning_rows == []


def test_metrics_counters_are_monotonic(tmp_path):
    runtime = shadow_judge.make_shadow_news_judge(
        lambda h, s: {"label": "real"}, lambda h, s: {"label": "real"},
        log_path=tmp_path / "shadow.jsonl")
    try:
        runtime("one", "s")
        _wait_until(lambda: runtime.metrics()["shadow_completed"] == 1)
        before = runtime.metrics()
        runtime("two", "s")
        _wait_until(lambda: runtime.metrics()["shadow_completed"] == 2)
        after = runtime.metrics()
        for key in ("shadow_submitted", "shadow_completed", "shadow_dropped", "shadow_write_failures"):
            assert after[key] >= before[key]
    finally:
        runtime.close()


def test_queue_full_rolls_back_admission_for_that_date():
    admission = shadow_judge.DailyAdmission(cap=1)
    et_date = date(2026, 8, 7)

    def full_queue():
        raise shadow_judge.queue.Full

    assert admission.try_enqueue(et_date, full_queue) is shadow_judge.AdmissionResult.QUEUE_FULL
    assert admission.accepted(et_date) == 0
    assert admission.try_enqueue(et_date, lambda: None) is shadow_judge.AdmissionResult.ACCEPTED
    assert admission.accepted(et_date) == 1


def test_first_and_every_25th_drop_warns(tmp_path, caplog):
    entered = threading.Event()
    release = threading.Event()

    def blocked(headline, summary):
        entered.set()
        release.wait(10)
        return {"label": "real"}

    runtime = shadow_judge.ShadowJudgeRuntime(
        lambda h, s: {"label": "real"}, blocked,
        log_path=tmp_path / "shadow.jsonl", queue_cap=1, daily_call_cap=100,
        log_every_drops=25)
    runtime.start()
    try:
        with caplog.at_level(logging.WARNING, logger="momentum_scanner.shadow_judge"):
            runtime("inflight", "s")
            assert entered.wait(1.0)
            runtime("queued", "s")
            for index in range(26):
                runtime(f"dropped-{index}", "s")
            release.set()
            _wait_until(
                lambda: len([
                    record for record in caplog.records
                    if "shadow drop" in record.getMessage()
                ]) == 2
            )
            warnings = [record for record in caplog.records if "shadow drop" in record.getMessage()]
            assert len(warnings) == 2
    finally:
        release.set()
        runtime.close()


def test_drop_warning_cannot_block_the_deciding_thread(tmp_path, monkeypatch):
    shadow_entered = threading.Event()
    shadow_release = threading.Event()
    warning_entered = threading.Event()
    warning_release = threading.Event()
    returned = threading.Event()
    result = {}

    def blocked_shadow(headline, summary):
        shadow_entered.set()
        shadow_release.wait(10)
        return {"label": "real"}

    def blocked_warning(*args, **kwargs):
        warning_entered.set()
        warning_release.wait(10)

    monkeypatch.setattr(shadow_judge._log, "warning", blocked_warning)
    runtime = shadow_judge.ShadowJudgeRuntime(
        lambda h, s: {"label": "real"}, blocked_shadow,
        log_path=tmp_path / "shadow.jsonl", queue_cap=1, daily_call_cap=100)
    runtime.start()

    def drop_call():
        result["verdict"] = runtime("dropped", "s")
        returned.set()

    caller = threading.Thread(target=drop_call)
    try:
        runtime("inflight", "s")
        assert shadow_entered.wait(1.0)
        runtime("queued", "s")
        caller.start()
        assert returned.wait(1.0)
        assert result["verdict"] == {"label": "real"}
        shadow_release.set()
        assert warning_entered.wait(1.0)
    finally:
        warning_release.set()
        shadow_release.set()
        caller.join(timeout=1.0)
        assert not caller.is_alive()
        runtime.close()


def test_blocked_worker_coalesces_warning_signals_without_delaying_primary(tmp_path, monkeypatch):
    shadow_entered = threading.Event()
    shadow_release = threading.Event()
    calls_finished = threading.Event()
    warning_counts = []
    primary = {"label": "real"}

    def blocked_shadow(headline, summary):
        shadow_entered.set()
        shadow_release.wait(10)
        return {"label": "real"}

    def record_warning(message, reason, count, detail):
        warning_counts.append(count)

    monkeypatch.setattr(shadow_judge._log, "warning", record_warning)
    runtime = shadow_judge.ShadowJudgeRuntime(
        lambda h, s: primary,
        blocked_shadow,
        log_path=tmp_path / "shadow.jsonl",
        queue_cap=1,
        daily_call_cap=1000,
        log_every_drops=25,
    )
    runtime.start()

    def burst_drops():
        for index in range(625):
            assert runtime(f"dropped-{index}", "s") is primary
        calls_finished.set()

    caller = threading.Thread(target=burst_drops)
    try:
        runtime("inflight", "s")
        assert shadow_entered.wait(1.0)
        runtime("queued", "s")
        caller.start()
        assert calls_finished.wait(2.0)
        shadow_release.set()
        _wait_until(lambda: runtime.metrics()["shadow_completed"] == 2)
        assert warning_counts == [1, 625]
    finally:
        shadow_release.set()
        caller.join(timeout=1.0)
        assert not caller.is_alive()
        runtime.close()


def test_raising_error_fn_logs_only_a_bounded_redacted_message(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-CANARYSECRET")

    def raising_error_fn():
        raise RuntimeError("canary sk-ant-CANARYSECRET")

    runtime = shadow_judge.make_shadow_news_judge(
        lambda h, s: {"label": "real"},
        lambda h, s: None,
        log_path=tmp_path / "shadow.jsonl",
        error_fn=raising_error_fn,
    )
    try:
        with caplog.at_level(logging.DEBUG, logger="momentum_scanner.shadow_judge"):
            runtime("headline", "summary")
            _wait_until(lambda: runtime.metrics()["shadow_completed"] == 1)
        assert "CANARYSECRET" not in caplog.text
        assert all(record.exc_info is None for record in caplog.records)
        assert any("RuntimeError: canary sk-ant-***" in record.getMessage() for record in caplog.records)
    finally:
        runtime.close()


def test_shadow_summary_is_truncated_to_8000_characters(tmp_path):
    seen = []
    runtime = shadow_judge.make_shadow_news_judge(
        lambda h, s: {"label": "real"},
        lambda h, s: seen.append(s) or {"label": "real"},
        log_path=tmp_path / "shadow.jsonl")
    try:
        runtime("headline", "x" * 9000)
        _wait_until(lambda: len(seen) == 1)
        assert len(seen[0]) == 8000
    finally:
        runtime.close()


def test_close_returns_within_two_seconds_while_provider_is_blocked(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    def blocked(headline, summary):
        entered.set()
        release.wait(10)
        return {"label": "real"}

    runtime = shadow_judge.make_shadow_news_judge(
        lambda h, s: {"label": "real"}, blocked, log_path=tmp_path / "shadow.jsonl")
    try:
        runtime("headline", "summary")
        assert entered.wait(1.0)
        before = time.monotonic()
        runtime.close()
        assert time.monotonic() - before < 2.0
    finally:
        release.set()


def test_worker_thread_name_reflects_its_own_shadow_backend(tmp_path):
    # Two ShadowJudgeRuntime instances can now run concurrently in the same process
    # (haiku + qwen3 lanes, see judge.py's _wrap_with_qwen3_shadow) -- a hardcoded
    # "haiku-shadow" thread name for every instance made them indistinguishable in
    # threading.enumerate()/thread dumps.
    haiku = shadow_judge.ShadowJudgeRuntime(
        lambda h, s: None, lambda h, s: None,
        log_path=tmp_path / "haiku.jsonl", shadow_backend="claude-haiku-4-5")
    qwen3 = shadow_judge.ShadowJudgeRuntime(
        lambda h, s: None, lambda h, s: None,
        log_path=tmp_path / "qwen3.jsonl", shadow_backend="qwen3:8b")
    haiku.start()
    qwen3.start()
    try:
        assert haiku._worker.name != qwen3._worker.name
        assert "claude-haiku-4-5" in haiku._worker.name
        assert "qwen3:8b" in qwen3._worker.name
    finally:
        haiku.close()
        qwen3.close()

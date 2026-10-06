# tests/test_judge_shadow_wiring.py
"""resolve_news_judge's shadow seam.

Default OFF is load-bearing, not stylistic: the CODE default must stay the proven
gemma-only path so the shadow build is a one-line rollback (unset the env) rather
than a redeploy.
"""
import builtins
import base64
import json
import logging
from pathlib import Path
import re
import subprocess
import threading
import time

import pytest

from momentum_scanner import judge
from momentum_scanner import qwen3_judge as qj
from momentum_scanner.catalyst_evaluator import default_news_judge


def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached before timeout")


def test_requirements_declare_supported_anthropic_range():
    requirements = Path("tools/momentum_scanner/requirements.txt").read_text(encoding="utf-8")
    assert "anthropic>=0.100.0,<1" in requirements.splitlines()


def test_scheduled_task_doc_points_to_generic_wrapper():
    doc = Path("tools/momentum_scanner/SCHEDULED_TASK.md").read_text(encoding="utf-8")
    assert "launch_scanner_discovery.ps1" in doc
    assert ".bat" not in doc          # only the tracked PowerShell wrapper, no batch launcher


def test_scheduled_task_doc_covers_rollback_and_validation_evidence():
    doc = Path("tools/momentum_scanner/SCHEDULED_TASK.md").read_text(encoding="utf-8")
    required = (
        "-PythonExe $PythonExe",
        "-StateDir $StateDir",
        "-Source \"alpaca\"",
        "remove `-EnableHaikuShadow` from the launch command",
        "restart the scanner",
        "judge shadow ACTIVE",
        "`shadow_status == active`",
        "`shadow_worker_alive == true`",
        "`record_type: comparison`",
        "`primary`",
        "`shadow`",
        "`primary_ts`",
        "`shadow_started_ts`",
        "`shadow_completed_ts`",
    )
    for statement in required:
        assert statement in doc


def test_scheduled_task_powershell_examples_parse_without_starting_daemon():
    doc = Path("tools/momentum_scanner/SCHEDULED_TASK.md").read_text(encoding="utf-8")
    examples = re.findall(r"```powershell\n(.*?)```", doc, flags=re.DOTALL)
    assert examples
    # Placeholders such as "<state-dir>" must stay inside quoted strings so each example
    # still parses as-is; the parse below catches a bare redirection-like token.
    assert any("$RepoRoot" in example and "Join-Path" in example for example in examples)

    for example in examples:
        script = "[void][scriptblock]::Create(@'\n" + example + "\n'@)"
        encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-EncodedCommand", encoded],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_runtime_resolver_off_returns_bare_gemma_descriptor(monkeypatch):
    # Catches a resolver that accidentally creates a worker or wrapper when shadow
    # observation is disabled.  The primary callable must remain identity-stable.
    monkeypatch.setattr(judge.config, "SCANNER_JUDGE_SHADOW", False)
    resolved = judge.resolve_news_judge_runtime()
    assert resolved.callable is default_news_judge
    assert resolved.status == "off"
    assert resolved.metrics_fn()["shadow_status"] == "off"
    resolved.close_fn()


def test_shadow_on_with_a_key_returns_a_wrapper(monkeypatch, tmp_path):
    monkeypatch.setattr(judge.config, "SCANNER_JUDGE_SHADOW", True)
    monkeypatch.setattr(judge.config, "SHADOW_JUDGE_LOG_PATH", str(tmp_path / "s.jsonl"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xxx")
    resolved = judge.resolve_news_judge_runtime()
    try:
        assert resolved.callable is not default_news_judge
        assert resolved.status == "active"
        assert callable(resolved.callable)
    finally:
        resolved.close_fn()


def test_shadow_on_without_a_key_degrades_to_gemma(monkeypatch, tmp_path, caplog):
    # Fail-safe: a launcher that never got the key must run the ordinary scanner,
    # not a wrapper that raises or logs an error per ticker per cycle.
    monkeypatch.setattr(judge.config, "SCANNER_JUDGE_SHADOW", True)
    monkeypatch.setattr(judge.config, "SHADOW_JUDGE_LOG_PATH", str(tmp_path / "s.jsonl"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with caplog.at_level(logging.INFO, logger="momentum_scanner.judge"):
        resolved = judge.resolve_news_judge_runtime()
    assert resolved.callable is default_news_judge
    assert resolved.status == "disabled_no_key"
    assert any("shadow" in r.getMessage().lower() for r in caplog.records)


def test_shadow_wrapper_still_returns_the_gemma_verdict(monkeypatch, tmp_path):
    # End-to-end through the real resolver: the wrapper it builds must be transparent.
    import momentum_scanner.haiku_judge as hj
    monkeypatch.setattr(judge.config, "SCANNER_JUDGE_SHADOW", True)
    monkeypatch.setattr(judge.config, "SHADOW_JUDGE_LOG_PATH", str(tmp_path / "s.jsonl"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xxx")
    monkeypatch.setattr(hj, "_anthropic_generate",
                        lambda p: '{"label":"pump","confidence":0.8}')
    gemma_verdict = {"label": "real", "confidence": 0.95,
                     "is_fixed_price_buyout": False, "rationale": "FDA"}
    monkeypatch.setattr(judge.catalyst_evaluator, "default_news_judge",
                        lambda h, s: gemma_verdict)
    resolved = judge.resolve_news_judge_runtime()
    try:
        assert resolved.callable("ABCD wins FDA approval", "endpoint met") == gemma_verdict
    finally:
        resolved.close_fn()


def test_worker_fatal_projects_degraded_descriptor_and_heartbeat(monkeypatch, tmp_path):
    from momentum_scanner import scanner_movers_daemon

    class FatalShadow(BaseException):
        pass

    primary = {"label": "real"}
    monkeypatch.setattr(judge.config, "SCANNER_JUDGE_SHADOW", True)
    monkeypatch.setattr(judge.config, "SHADOW_JUDGE_LOG_PATH", str(tmp_path / "s.jsonl"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SYNTHETIC")
    monkeypatch.setattr(judge.catalyst_evaluator, "default_news_judge", lambda h, s: primary)
    monkeypatch.setattr(
        judge.haiku_judge, "haiku_news_judge",
        lambda h, s: (_ for _ in ()).throw(FatalShadow("worker failed")),
    )
    resolved = judge.resolve_news_judge_runtime()
    try:
        assert resolved.callable("headline", "summary") is primary
        _wait_until(lambda: not resolved.metrics_fn()["shadow_worker_alive"])
        descriptor = resolved.metrics_fn()
        heartbeat = scanner_movers_daemon._shadow_metrics_projection(resolved.metrics_fn)
        assert descriptor["shadow_status"] == "degraded"
        assert descriptor["shadow_last_error_type"] == "FatalShadow"
        assert heartbeat["shadow_status"] == "degraded"
        assert heartbeat["shadow_last_error_type"] == "FatalShadow"
        assert heartbeat["shadow_fatal_errors"] == 1
        assert heartbeat["shadow_abandoned_backlog"] == 0
    finally:
        resolved.close_fn()


@pytest.mark.parametrize("raw_path", ["", "relative.jsonl"])
def test_enabled_invalid_path_is_setup_failed_without_active(monkeypatch, raw_path, caplog):
    # Catches relative-path fallback: a daemon cwd must never silently select where
    # a durable audit log lands.
    monkeypatch.setattr(judge.config, "SCANNER_JUDGE_SHADOW", True)
    monkeypatch.setattr(judge.config, "SHADOW_JUDGE_LOG_PATH", raw_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SYNTHETIC")
    with caplog.at_level(logging.INFO, logger="momentum_scanner.judge"):
        resolved = judge.resolve_news_judge_runtime()
    assert resolved.callable is default_news_judge
    assert resolved.status == "setup_failed"
    assert resolved.metrics_fn()["shadow_status"] == "setup_failed"
    assert not any("ACTIVE" in record.getMessage() for record in caplog.records)


def test_active_marker_appends_after_start_and_before_active_log(monkeypatch, tmp_path):
    # Catches an ACTIVE state which is published before the session marker survives
    # a process crash, or a marker that replaces earlier sessions instead of appending.
    log_path = tmp_path / "shadow.jsonl"
    log_path.write_text('{"record_type":"prior"}\n', encoding="utf-8")
    events = []
    original_start = judge.shadow_judge.ShadowJudgeRuntime.start
    original_fsync = judge.os.fsync
    original_info = judge._log.info

    class RecordingHandle:
        def __init__(self, handle):
            self._handle = handle

        def __enter__(self):
            events.append("enter")
            self._handle.__enter__()
            return self

        def __exit__(self, exc_type, exc, traceback):
            result = self._handle.__exit__(exc_type, exc, traceback)
            events.append("close")
            return result

        def write(self, text):
            events.append("write")
            return self._handle.write(text)

        def flush(self):
            events.append("flush")
            return self._handle.flush()

        def fileno(self):
            return self._handle.fileno()

    def record_start(runtime):
        events.append("start")
        return original_start(runtime)

    def record_open(*args, **kwargs):
        return RecordingHandle(builtins.open(*args, **kwargs))

    def record_fsync(fd):
        events.append("fsync")
        return original_fsync(fd)

    def record_info(message, *args, **kwargs):
        if "ACTIVE" in message:
            events.append("active")
        return original_info(message, *args, **kwargs)

    monkeypatch.setattr(judge.config, "SCANNER_JUDGE_SHADOW", True)
    monkeypatch.setattr(judge.config, "SHADOW_JUDGE_LOG_PATH", str(log_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SYNTHETIC")
    monkeypatch.setattr(judge.shadow_judge.ShadowJudgeRuntime, "start", record_start)
    monkeypatch.setattr(judge, "open", record_open, raising=False)
    monkeypatch.setattr(judge.os, "fsync", record_fsync)
    monkeypatch.setattr(judge._log, "info", record_info)
    # The ACTIVE log line is deduped via module state; without a reset, a prior shadow
    # test in the same process leaves the state at "active" and the dedup swallows the
    # line this test orders against.
    judge._reset_shadow_log_state()
    resolved = judge.resolve_news_judge_runtime()
    try:
        rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        row = rows[1]
        assert resolved.status == "active"
        assert rows[0] == {"record_type": "prior"}
        assert row == {
            "record_type": "shadow_start",
            "ts": row["ts"],
            "shadow_backend": judge.config.HAIKU_MODEL,
            "queue_cap": 20,
            "daily_call_cap": 250,
        }
        assert events == ["start", "enter", "write", "flush", "fsync", "close", "active"]
    finally:
        resolved.close_fn()


class _FlushFailureHandle:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def write(self, text):
        return len(text)

    def flush(self):
        raise OSError("flush failed")

    def fileno(self):
        return 1


class _ExitFailureHandle:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        raise OSError("close failed")

    def write(self, text):
        return len(text)

    def flush(self):
        return None

    def fileno(self):
        return 1


def _raise_open(*args, **kwargs):
    raise OSError("open failed")


def _raise_start(self):
    raise RuntimeError("worker start failed")


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param("open", id="open"),
        pytest.param("flush", id="flush"),
        pytest.param("fsync", id="fsync"),
        pytest.param("close", id="handle-close"),
        pytest.param("start", id="worker-start"),
        pytest.param("missing_parent", id="missing-parent"),
    ],
)
def test_failed_admission_is_bare_gemma_with_one_warning(monkeypatch, tmp_path, caplog, failure):
    # Each injected failure catches a fail-open admission branch that would expose
    # an active wrapper without a durable audit session.
    log_path = tmp_path / "shadow.jsonl"
    monkeypatch.setattr(judge.config, "SCANNER_JUDGE_SHADOW", True)
    monkeypatch.setattr(judge.config, "SHADOW_JUDGE_LOG_PATH", str(log_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SYNTHETIC")
    created = []
    closed = []
    original_init = judge.shadow_judge.ShadowJudgeRuntime.__init__
    original_close = judge.shadow_judge.ShadowJudgeRuntime.close

    def capture_init(runtime, *args, **kwargs):
        original_init(runtime, *args, **kwargs)
        created.append(runtime)

    def capture_close(runtime, *args, **kwargs):
        closed.append(runtime)
        return original_close(runtime, *args, **kwargs)

    monkeypatch.setattr(judge.shadow_judge.ShadowJudgeRuntime, "__init__", capture_init)
    monkeypatch.setattr(judge.shadow_judge.ShadowJudgeRuntime, "close", capture_close)
    if failure == "open":
        monkeypatch.setattr(judge, "open", _raise_open, raising=False)
    elif failure == "flush":
        monkeypatch.setattr(judge, "open", lambda *args, **kwargs: _FlushFailureHandle(), raising=False)
    elif failure == "fsync":
        monkeypatch.setattr(judge.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("fsync failed")))
    elif failure == "close":
        monkeypatch.setattr(judge, "open", lambda *args, **kwargs: _ExitFailureHandle(), raising=False)
    elif failure == "start":
        monkeypatch.setattr(judge.shadow_judge.ShadowJudgeRuntime, "start", _raise_start)
    else:
        monkeypatch.setattr(judge.config, "SHADOW_JUDGE_LOG_PATH", str(tmp_path / "missing" / "shadow.jsonl"))
    with caplog.at_level(logging.WARNING, logger="momentum_scanner.judge"):
        resolved = judge.resolve_news_judge_runtime()
    assert resolved.callable is default_news_judge
    assert resolved.status == "setup_failed"
    assert resolved.metrics_fn()["shadow_status"] == "setup_failed"
    assert len([record for record in caplog.records if record.levelno == logging.WARNING]) == 1
    assert not any("ACTIVE" in record.getMessage() for record in caplog.records)
    if failure == "missing_parent":
        assert created == []
        assert closed == []
    else:
        assert len(created) == 1
        assert closed == [created[0]]
        assert not created[0].metrics()["shadow_worker_alive"]


# ---- qwen3 shadow layer: a SECOND, independent wrap around whatever the haiku layer
# above produced. Unlike haiku (no credential needed -- local Ollama), the interesting
# axis here is GPU contention, not API-key admission; see config.py's comment on
# SHADOW_JUDGE_QWEN3_DAILY_CALL_CAP for why the cap is far below haiku's. ------------

def _off_qwen3(monkeypatch):
    monkeypatch.setattr(judge.config, "SCANNER_JUDGE_SHADOW_QWEN3", False)


def test_qwen3_off_leaves_haiku_layer_callable_identity_untouched(monkeypatch):
    monkeypatch.setattr(judge.config, "SCANNER_JUDGE_SHADOW", False)
    _off_qwen3(monkeypatch)
    resolved = judge.resolve_news_judge_runtime()
    try:
        assert resolved.callable is default_news_judge
        assert resolved.status == "off"
        assert resolved.metrics_fn()["qwen3_shadow_status"] == "off"
    finally:
        resolved.close_fn()


def test_qwen3_off_still_reports_off_when_haiku_layer_is_active(monkeypatch, tmp_path):
    # qwen3 being off must not depend on what the haiku layer resolved to.
    monkeypatch.setattr(judge.config, "SCANNER_JUDGE_SHADOW", True)
    monkeypatch.setattr(judge.config, "SHADOW_JUDGE_LOG_PATH", str(tmp_path / "haiku.jsonl"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SYNTHETIC")
    _off_qwen3(monkeypatch)
    resolved = judge.resolve_news_judge_runtime()
    try:
        assert resolved.status == "active"                 # haiku layer's own status, unaffected
        assert resolved.metrics_fn()["qwen3_shadow_status"] == "off"
    finally:
        resolved.close_fn()


def test_qwen3_on_wraps_bare_gemma_when_haiku_layer_is_off(monkeypatch, tmp_path):
    monkeypatch.setattr(judge.config, "SCANNER_JUDGE_SHADOW", False)
    monkeypatch.setattr(judge.config, "SCANNER_JUDGE_SHADOW_QWEN3", True)
    monkeypatch.setattr(judge.config, "SHADOW_JUDGE_LOG_PATH_QWEN3", str(tmp_path / "q.jsonl"))
    gemma_verdict = {"label": "real", "confidence": 0.9,
                     "is_fixed_price_buyout": False, "rationale": "FDA"}
    monkeypatch.setattr(judge.catalyst_evaluator, "default_news_judge", lambda h, s: gemma_verdict)
    monkeypatch.setattr(qj, "qwen3_news_judge", lambda h, s: {"label": "pump", "confidence": 0.5})
    resolved = judge.resolve_news_judge_runtime()
    try:
        assert resolved.status == "off"                    # haiku layer's own status, unaffected
        assert resolved.callable is not default_news_judge  # now wrapped by qwen3 alone
        assert resolved.callable("h", "s") == gemma_verdict  # transparent: gemma's verdict, unmodified
        assert resolved.metrics_fn()["qwen3_shadow_status"] == "active"
    finally:
        resolved.close_fn()


def test_qwen3_on_wraps_the_haiku_layer_and_still_returns_gemma_verdict(monkeypatch, tmp_path):
    # Both lanes active: gemma decides, haiku shadows it, qwen3 shadows the OUTPUT of
    # that composition -- the deciding thread's return value must survive both wraps
    # byte-for-byte identical to gemma's own verdict.
    monkeypatch.setattr(judge.config, "SCANNER_JUDGE_SHADOW", True)
    monkeypatch.setattr(judge.config, "SHADOW_JUDGE_LOG_PATH", str(tmp_path / "haiku.jsonl"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SYNTHETIC")
    monkeypatch.setattr(judge.config, "SCANNER_JUDGE_SHADOW_QWEN3", True)
    monkeypatch.setattr(judge.config, "SHADOW_JUDGE_LOG_PATH_QWEN3", str(tmp_path / "q.jsonl"))
    gemma_verdict = {"label": "real", "confidence": 0.95,
                     "is_fixed_price_buyout": False, "rationale": "FDA"}
    monkeypatch.setattr(judge.catalyst_evaluator, "default_news_judge", lambda h, s: gemma_verdict)
    monkeypatch.setattr(judge.haiku_judge, "haiku_news_judge", lambda h, s: {"label": "pump"})
    monkeypatch.setattr(qj, "qwen3_news_judge", lambda h, s: {"label": "no-data"})
    resolved = judge.resolve_news_judge_runtime()
    try:
        assert resolved.status == "active"
        returned = resolved.callable("ABCD wins FDA approval", "endpoint met")
        assert returned == gemma_verdict
        metrics = resolved.metrics_fn()
        assert metrics["shadow_status"] == "active"         # haiku's own (unprefixed) key, unaffected
        assert metrics["qwen3_shadow_status"] == "active"
    finally:
        resolved.close_fn()


def test_qwen3_control_row_uses_its_own_dedicated_daily_cap(monkeypatch, tmp_path):
    # The whole point of a separate constant is that it actually reaches the runtime,
    # not just exists in config.py unused.
    log_path = tmp_path / "q.jsonl"
    monkeypatch.setattr(judge.config, "SCANNER_JUDGE_SHADOW", False)
    monkeypatch.setattr(judge.config, "SCANNER_JUDGE_SHADOW_QWEN3", True)
    monkeypatch.setattr(judge.config, "SHADOW_JUDGE_LOG_PATH_QWEN3", str(log_path))
    resolved = judge.resolve_news_judge_runtime()
    try:
        row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
        assert row["record_type"] == "shadow_start"
        assert row["shadow_backend"] == judge.config.QWEN3_MODEL
        assert row["daily_call_cap"] == judge.config.SHADOW_JUDGE_QWEN3_DAILY_CALL_CAP
        assert row["daily_call_cap"] < judge.config.SHADOW_JUDGE_DAILY_CALL_CAP
    finally:
        resolved.close_fn()


@pytest.mark.parametrize("raw_path", ["", "relative.jsonl"])
def test_qwen3_invalid_path_degrades_to_the_haiku_layer_unchanged(monkeypatch, tmp_path, caplog, raw_path):
    monkeypatch.setattr(judge.config, "SCANNER_JUDGE_SHADOW", False)
    monkeypatch.setattr(judge.config, "SCANNER_JUDGE_SHADOW_QWEN3", True)
    monkeypatch.setattr(judge.config, "SHADOW_JUDGE_LOG_PATH_QWEN3", raw_path)
    with caplog.at_level(logging.WARNING, logger="momentum_scanner.judge"):
        resolved = judge.resolve_news_judge_runtime()
    assert resolved.callable is default_news_judge
    assert resolved.metrics_fn()["qwen3_shadow_status"] == "setup_failed"
    assert any("qwen3" in r.getMessage().lower() for r in caplog.records)
    resolved.close_fn()


def test_qwen3_close_fn_closes_both_runtimes(monkeypatch, tmp_path):
    monkeypatch.setattr(judge.config, "SCANNER_JUDGE_SHADOW", True)
    monkeypatch.setattr(judge.config, "SHADOW_JUDGE_LOG_PATH", str(tmp_path / "haiku.jsonl"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SYNTHETIC")
    monkeypatch.setattr(judge.config, "SCANNER_JUDGE_SHADOW_QWEN3", True)
    monkeypatch.setattr(judge.config, "SHADOW_JUDGE_LOG_PATH_QWEN3", str(tmp_path / "q.jsonl"))
    resolved = judge.resolve_news_judge_runtime()
    resolved.close_fn()
    metrics = resolved.metrics_fn()
    assert metrics["shadow_worker_alive"] is False
    assert metrics["qwen3_shadow_worker_alive"] is False


def test_qwen3_close_runs_concurrently_with_haiku_close_not_sequentially(monkeypatch, tmp_path):
    # Each lane's close() is individually bounded at SHADOW_JUDGE_SHUTDOWN_SEC (2.0s).
    # Sequentially joining both would double worst-case shutdown to ~4s when both
    # workers are mid-call; closing them concurrently keeps it near the single bound.
    haiku_entered = threading.Event()
    qwen3_entered = threading.Event()
    release = threading.Event()

    monkeypatch.setattr(judge.config, "SCANNER_JUDGE_SHADOW", True)
    monkeypatch.setattr(judge.config, "SHADOW_JUDGE_LOG_PATH", str(tmp_path / "haiku.jsonl"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SYNTHETIC")
    monkeypatch.setattr(judge.config, "SCANNER_JUDGE_SHADOW_QWEN3", True)
    monkeypatch.setattr(judge.config, "SHADOW_JUDGE_LOG_PATH_QWEN3", str(tmp_path / "q.jsonl"))
    monkeypatch.setattr(judge.catalyst_evaluator, "default_news_judge", lambda h, s: {"label": "real"})

    def blocked_haiku(h, s):
        haiku_entered.set()
        release.wait(10)
        return {"label": "real"}

    def blocked_qwen3(h, s):
        qwen3_entered.set()
        release.wait(10)
        return {"label": "real"}

    monkeypatch.setattr(judge.haiku_judge, "haiku_news_judge", blocked_haiku)
    monkeypatch.setattr(qj, "qwen3_news_judge", blocked_qwen3)

    resolved = judge.resolve_news_judge_runtime()
    try:
        resolved.callable("headline", "summary")   # dispatches both shadow lanes async
        assert haiku_entered.wait(1.0)
        assert qwen3_entered.wait(1.0)
        before = time.monotonic()
        resolved.close_fn()
        elapsed = time.monotonic() - before
        assert elapsed < 3.0, f"close() took {elapsed:.2f}s -- looks sequential, not concurrent"
    finally:
        release.set()


def test_qwen3_control_row_write_failure_after_start_reaps_the_worker(monkeypatch, tmp_path, caplog):
    # Mirrors the haiku layer's "start" failure-mode coverage: the qwen3 lane's own
    # runtime.start()-succeeds-then-write-fails path had no regression test.
    log_path = tmp_path / "q.jsonl"
    monkeypatch.setattr(judge.config, "SCANNER_JUDGE_SHADOW", False)
    monkeypatch.setattr(judge.config, "SCANNER_JUDGE_SHADOW_QWEN3", True)
    monkeypatch.setattr(judge.config, "SHADOW_JUDGE_LOG_PATH_QWEN3", str(log_path))
    created = []
    closed = []
    original_init = judge.shadow_judge.ShadowJudgeRuntime.__init__
    original_close = judge.shadow_judge.ShadowJudgeRuntime.close

    def capture_init(runtime_self, *args, **kwargs):
        original_init(runtime_self, *args, **kwargs)
        created.append(runtime_self)

    def capture_close(runtime_self, *args, **kwargs):
        closed.append(runtime_self)
        return original_close(runtime_self, *args, **kwargs)

    monkeypatch.setattr(judge.shadow_judge.ShadowJudgeRuntime, "__init__", capture_init)
    monkeypatch.setattr(judge.shadow_judge.ShadowJudgeRuntime, "close", capture_close)
    monkeypatch.setattr(judge, "open", _raise_open, raising=False)
    with caplog.at_level(logging.WARNING, logger="momentum_scanner.judge"):
        resolved = judge.resolve_news_judge_runtime()
    assert resolved.callable is default_news_judge
    assert resolved.metrics_fn()["qwen3_shadow_status"] == "setup_failed"
    assert len(created) == 1
    assert closed == [created[0]]
    assert not created[0].metrics()["shadow_worker_alive"]
    assert any("qwen3" in r.getMessage().lower() for r in caplog.records)
    resolved.close_fn()

"""Contract tests for the generic discovery launcher."""
from pathlib import Path
import subprocess

import pytest


WRAPPER = Path("tools/momentum_scanner/launch_scanner_discovery.ps1")


def _fake_child(tmp_path):
    fixture_root = tmp_path / "launcher fixture with spaces"
    fixture_root.mkdir()
    capture = fixture_root / "capture.txt"
    child = fixture_root / "fake python.cmd"
    child.write_text(
        "@echo off\r\n"
        f">\"{capture}\" echo(%SCANNER_JUDGE_SHADOW%\r\n"
        f">>\"{capture}\" echo(%SCANNER_SHADOW_LOG_PATH%\r\n"
        f">>\"{capture}\" echo(%PYTHONPATH%\r\n"
        f">>\"{capture}\" echo(arg1=[%~1]\r\n"
        f">>\"{capture}\" echo(arg2=[%~2]\r\n"
        f">>\"{capture}\" echo(arg3=[%~3]\r\n"
        f">>\"{capture}\" echo(arg4=[%~4]\r\n"
        f">>\"{capture}\" echo(arg5=[%~5]\r\n"
        f">>\"{capture}\" echo(arg6=[%~6]\r\n"
        f">>\"{capture}\" echo(arg7=[%~7]\r\n"
        "exit /b 37\r\n",
        encoding="ascii",
    )
    return child, capture


def _run_wrapper(*args):
    return subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(WRAPPER),
            *map(str, args),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_wrapper_derives_shadow_path_before_child_start(tmp_path):
    child, capture = _fake_child(tmp_path)
    state = tmp_path / "state directory with spaces"
    state.mkdir()

    completed = _run_wrapper(
        "-PythonExe", child,
        "-StateDir", state,
        "-Source", "alpaca",
        "-EnableHaikuShadow",
    )

    assert completed.returncode == 37
    lines = capture.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "1"
    assert Path(lines[1]) == state.resolve() / "judge_shadow.jsonl"
    assert Path("tools").resolve() == Path(lines[2].split(";")[0])
    assert lines[3:] == [
        "arg1=[-m]",
        "arg2=[momentum_scanner.scanner_movers_daemon]",
        "arg3=[--state-dir]",
        f"arg4=[{state.resolve()}]",
        "arg5=[--source]",
        "arg6=[alpaca]",
        "arg7=[]",
    ]


def test_wrapper_disables_shadow_and_clears_inherited_log_path_without_switch(
    tmp_path, monkeypatch,
):
    child, capture = _fake_child(tmp_path)
    state = tmp_path / "state directory with spaces"
    state.mkdir()
    monkeypatch.setenv("SCANNER_SHADOW_LOG_PATH", "INHERITED-SENTINEL")

    completed = _run_wrapper(
        "-PythonExe", child,
        "-StateDir", state,
        "-Source", "finnhub",
    )

    assert completed.returncode == 37
    lines = capture.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "0"
    assert lines[1] == ""
    assert lines[3:] == [
        "arg1=[-m]",
        "arg2=[momentum_scanner.scanner_movers_daemon]",
        "arg3=[--state-dir]",
        f"arg4=[{state.resolve()}]",
        "arg5=[--source]",
        "arg6=[finnhub]",
        "arg7=[]",
    ]


@pytest.mark.parametrize("state_kind", ["missing", "file"])
def test_wrapper_rejects_non_directory_state_before_child_start(tmp_path, state_kind):
    child, capture = _fake_child(tmp_path)
    state = tmp_path / "state directory with spaces"
    if state_kind == "file":
        state.write_text("not a directory", encoding="ascii")

    completed = _run_wrapper(
        "-PythonExe", child,
        "-StateDir", state,
        "-Source", "alpaca",
    )

    assert completed.returncode != 0
    assert not capture.exists()


def test_wrapper_validates_source_before_child_start(tmp_path):
    child, capture = _fake_child(tmp_path)
    state = tmp_path / "state directory with spaces"
    state.mkdir()

    completed = _run_wrapper(
        "-PythonExe", child,
        "-StateDir", state,
        "-Source", "invalid",
    )

    assert completed.returncode != 0
    assert not capture.exists()


def test_wrapper_has_no_credentials_or_hard_coded_local_paths():
    wrapper = WRAPPER.read_text(encoding="ascii")
    for forbidden in (
        "ANTHROPIC_API_KEY",
        "ALPACA_API_KEY",
        "C:\\Users\\",
    ):
        assert forbidden not in wrapper

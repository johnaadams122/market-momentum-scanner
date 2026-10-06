# tests/test_watchlist_io.py
import json
import os
import threading
import time
from datetime import datetime, timezone
from unittest.mock import patch
import pytest
from momentum_scanner import watchlist_io as wio
from momentum_scanner.catalyst_evaluator import CatalystVerdict

FIXED = datetime(2026, 6, 29, 13, 0, tzinfo=timezone.utc)
PROCEED = CatalystVerdict("PROCEED", "real", "FDA", 0.9, "llm")


def test_v2_envelope_ok_shape():
    env = wio.v2_envelope([{"ticker": "AAA"}], scan_id="sid", scanned_at=FIXED)
    assert env["schema_version"] == 2 and env["scan_id"] == "sid" and env["status"] == "ok"
    assert env["confirm_overflow"] == 0 and env["candidates"] == [{"ticker": "AAA"}]
    assert env["expires_at"] > env["scanned_at"] and "error" not in env


def test_v2_envelope_error_adds_error_key():
    env = wio.v2_envelope([], scan_id="s", scanned_at=FIXED, status="error", error="schwab down")
    assert env["status"] == "error" and env["candidates"] == [] and env["error"] == "schwab down"


def test_stamp_verdict_present_and_missing():
    d1 = {}
    wio.stamp_verdict(d1, PROCEED)
    assert d1["gate_decision"] == "PROCEED" and d1["catalyst_label"] == "real" \
        and d1["catalyst_confidence"] == 0.9 and d1["catalyst_source"] == "llm"
    d2 = {}
    wio.stamp_verdict(d2, None)
    assert d2["gate_decision"] == "REJECT" and d2["catalyst_label"] == "no-data" \
        and d2["gate_reason"] == "not evaluated" and d2["catalyst_confidence"] is None \
        and d2["catalyst_source"] == "rule"


def test_atomic_write_rejects_nonfinite(tmp_path):
    out = tmp_path / "w.json"
    with pytest.raises(ValueError):
        wio.atomic_write_json(str(out), {"x": float("inf")})
    assert not any(p.suffix == ".tmp" for p in tmp_path.iterdir())


def test_atomic_write_roundtrip(tmp_path):
    out = tmp_path / "sub" / "w.json"
    wio.atomic_write_json(str(out), {"schema_version": 2})
    assert json.loads(out.read_text(encoding="utf-8"))["schema_version"] == 2
    assert not any(p.suffix == ".tmp" for p in out.parent.iterdir())


def test_atomic_write_reader_never_sees_partial(tmp_path):
    # Concurrent writer/reader: a reader during a write sees old-complete or new-complete.
    out = tmp_path / "w.json"
    wio.atomic_write_json(str(out), {"v": 1, "candidates": ["old"]})
    stop = {"go": True}
    seen = []

    def reader():
        while stop["go"]:
            try:
                seen.append(json.loads(out.read_text(encoding="utf-8")))   # never raises mid-write
            except (json.JSONDecodeError, FileNotFoundError):
                seen.append("PARTIAL")
            except PermissionError:
                pass  # Windows transient during os.replace; not partial JSON
            time.sleep(0.001)

    t = threading.Thread(target=reader)
    t.start()
    try:
        for i in range(50):
            wio.atomic_write_json(str(out), {"v": i, "candidates": ["x"] * 200})
    finally:
        stop["go"] = False
        t.join()
    assert "PARTIAL" not in seen          # os.replace is atomic -> reader never catches a half-write


def test_atomic_write_retries_then_succeeds(tmp_path):
    """os.replace raises PermissionError 2 times, then succeeds; file written, no .tmp orphan."""
    out = tmp_path / "w.json"
    payload = {"schema_version": 2, "test": "data"}

    real_replace = os.replace
    replace_call_count = [0]

    def mock_replace(src, dst):
        replace_call_count[0] += 1
        if replace_call_count[0] <= 2:
            raise PermissionError("Windows NTFS lock")
        real_replace(src, dst)

    with patch('momentum_scanner.watchlist_io.os.replace', side_effect=mock_replace):
        with patch('momentum_scanner.watchlist_io.time.sleep'):
            wio.atomic_write_json(str(out), payload)

    assert out.exists(), "Output file should exist"
    assert json.loads(out.read_text(encoding="utf-8")) == payload
    assert replace_call_count[0] == 3, f"Expected 3 os.replace calls, got {replace_call_count[0]}"
    assert not any(p.suffix == ".tmp" for p in out.parent.iterdir()), ".tmp orphan found"


def test_atomic_write_persistent_permission_error_raises_and_cleans_tmp(tmp_path):
    """os.replace always raises PermissionError; atomic_write_json raises and cleans tmp."""
    out = tmp_path / "w.json"
    payload = {"schema_version": 2, "test": "data"}

    with patch('momentum_scanner.watchlist_io.os.replace', side_effect=PermissionError("Windows NTFS lock")):
        with patch('momentum_scanner.watchlist_io.time.sleep'):
            with pytest.raises(PermissionError):
                wio.atomic_write_json(str(out), payload)

    assert not out.exists(), "Output file should not exist"
    assert not any(p.suffix == ".tmp" for p in out.parent.iterdir()), ".tmp orphan found"


from types import SimpleNamespace


def test_stamp_verdict_includes_dilution_flag():
    v = SimpleNamespace(decision="PROCEED", label="real", reason="ok", confidence=0.9,
                        source="llm", dilution_flag="active_offering")
    d = {}
    wio.stamp_verdict(d, v)
    assert d["dilution_flag"] == "active_offering"


def test_stamp_verdict_none_verdict_flag_is_null():
    d = {}
    wio.stamp_verdict(d, None)
    assert d["dilution_flag"] is None

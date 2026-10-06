import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from momentum_scanner import scanner_movers_daemon as d
from momentum_scanner.daemon_heartbeat import write_heartbeat


def test_writes_atomic_heartbeat(tmp_path):
    p = tmp_path / "hb.json"
    now = datetime(2026, 6, 30, 13, 30, tzinfo=timezone.utc).isoformat()
    out = write_heartbeat(p, pid=1234, started_at_iso=now, now_iso=now, mode="premarket",
                          last_status="ok", cycle_stats={"candidates": 3},
                          extra={"judge_worker_alive": True, "queue_depth": 0, "drops": 0, "rate_backoff": 0})
    on_disk = json.loads(p.read_text())
    assert on_disk["pid"] == 1234 and on_disk["mode"] == "premarket"
    assert on_disk["heartbeat_at"] == now and on_disk["last_status"] == "ok"
    assert on_disk["stats"]["candidates"] == 3
    assert on_disk["judge_worker_alive"] is True and on_disk["queue_depth"] == 0 and on_disk["drops"] == 0
    assert out == on_disk


def test_run_once_writes_bounded_shadow_heartbeat_defaults(tmp_path):
    """Catches a daemon heartbeat with no shadow telemetry when the runtime is unavailable."""
    now = datetime(2026, 6, 30, 13, 30, tzinfo=timezone.utc)
    heartbeat_path = tmp_path / "hb.json"

    d.run_once(
        symbols=[], token_fn=lambda: "token", session=object(), limiter=None, cache=None,
        enqueue_fn=lambda batch: None, float_cache={}, float_fetch_fn=lambda ticker: None,
        now_utc=now, now_et=now.astimezone(ZoneInfo("America/New_York")), mode="closed",
        out_path=tmp_path / "watchlist.json", heartbeat_path=heartbeat_path,
        started_at_iso=now.isoformat(), pid=1234, worker_alive=True, queue_depth=0, drops=0,
        rate_backoff=0, shadow_metrics=lambda: {"shadow_status": "active", "shadow_submitted": 3},
    )

    on_disk = json.loads(heartbeat_path.read_text())
    assert {key: on_disk[key] for key in (
        "shadow_status", "shadow_worker_alive", "shadow_queue_depth", "shadow_submitted",
        "shadow_completed", "shadow_dropped", "shadow_dropped_daily_cap",
        "shadow_dropped_queue_full", "shadow_dropped_capture_failed", "shadow_write_failures",
        "shadow_dropped_worker_dead", "shadow_dropped_closed", "shadow_fatal_errors",
        "shadow_abandoned_backlog", "shadow_last_error_type",
    )} == {
        "shadow_status": "active",
        "shadow_worker_alive": False,
        "shadow_queue_depth": 0,
        "shadow_submitted": 3,
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


def test_run_once_writes_bounded_qwen3_shadow_heartbeat_defaults(tmp_path):
    """Same contract as the haiku shadow defaults above, prefixed -- the qwen3 lane
    must be present in the heartbeat (defaulting to 'off') even when the resolver's
    shadow_metrics callback never reports it, so an operator can always see whether
    it's running without reading judge_shadow_qwen3.jsonl directly."""
    now = datetime(2026, 6, 30, 13, 30, tzinfo=timezone.utc)
    heartbeat_path = tmp_path / "hb.json"

    d.run_once(
        symbols=[], token_fn=lambda: "token", session=object(), limiter=None, cache=None,
        enqueue_fn=lambda batch: None, float_cache={}, float_fetch_fn=lambda ticker: None,
        now_utc=now, now_et=now.astimezone(ZoneInfo("America/New_York")), mode="closed",
        out_path=tmp_path / "watchlist.json", heartbeat_path=heartbeat_path,
        started_at_iso=now.isoformat(), pid=1234, worker_alive=True, queue_depth=0, drops=0,
        rate_backoff=0,
        shadow_metrics=lambda: {"qwen3_shadow_status": "active", "qwen3_shadow_submitted": 2},
    )

    on_disk = json.loads(heartbeat_path.read_text())
    assert {key: on_disk[key] for key in (
        "qwen3_shadow_status", "qwen3_shadow_worker_alive", "qwen3_shadow_queue_depth",
        "qwen3_shadow_submitted", "qwen3_shadow_completed", "qwen3_shadow_dropped",
        "qwen3_shadow_dropped_daily_cap", "qwen3_shadow_last_error_type",
    )} == {
        "qwen3_shadow_status": "active",
        "qwen3_shadow_worker_alive": False,
        "qwen3_shadow_queue_depth": 0,
        "qwen3_shadow_submitted": 2,
        "qwen3_shadow_completed": 0,
        "qwen3_shadow_dropped": 0,
        "qwen3_shadow_dropped_daily_cap": 0,
        "qwen3_shadow_last_error_type": None,
    }


def test_run_once_omits_qwen3_when_shadow_metrics_is_none(tmp_path):
    now = datetime(2026, 6, 30, 13, 30, tzinfo=timezone.utc)
    heartbeat_path = tmp_path / "hb.json"

    d.run_once(
        symbols=[], token_fn=lambda: "token", session=object(), limiter=None, cache=None,
        enqueue_fn=lambda batch: None, float_cache={}, float_fetch_fn=lambda ticker: None,
        now_utc=now, now_et=now.astimezone(ZoneInfo("America/New_York")), mode="closed",
        out_path=tmp_path / "watchlist.json", heartbeat_path=heartbeat_path,
        started_at_iso=now.isoformat(), pid=1234, worker_alive=True, queue_depth=0, drops=0,
        rate_backoff=0, shadow_metrics=None,
    )
    on_disk = json.loads(heartbeat_path.read_text())
    assert on_disk["qwen3_shadow_status"] == "off"

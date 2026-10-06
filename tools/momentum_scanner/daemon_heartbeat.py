"""Rolling heartbeat for the movers daemon. Atomic so readers (watchdog, consumers)
never read a torn file. Single writer (the daemon) -> no lock needed."""
from momentum_scanner import watchlist_io


def write_heartbeat(path, *, pid, started_at_iso, now_iso, mode, last_status,
                    cycle_stats=None, extra=None):
    hb = {"pid": pid, "started_at": started_at_iso, "heartbeat_at": now_iso,
          "mode": mode, "last_status": last_status, "stats": cycle_stats or {}}
    if extra:
        hb.update(extra)
    watchlist_io.atomic_write_json(path, hb)
    return hb

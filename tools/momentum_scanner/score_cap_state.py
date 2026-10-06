'Score-cap state marker: read/validate the on-disk cap marker and compare it with the resolved config and published watchlists.'
import json
import math
import os
from pathlib import Path

MARKER_FILENAME = "score_cap_state.json"


class MalformedMarkerError(ValueError):
    """The marker exists but cannot be trusted -- unparseable, wrong shape, or wrong types.
    Callers must fail closed (discovery: abort startup; consumers: refuse fresh watchlists)."""


def marker_path(state_dir):
    return os.path.join(state_dir, MARKER_FILENAME)


def _is_finite_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def read_marker(state_dir):
    """None when the marker file is absent (inert). A validated dict when well-formed.
    Raises MalformedMarkerError otherwise -- never returns a guess."""
    p = Path(marker_path(state_dir))
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MalformedMarkerError("unreadable marker %s: %s" % (p, exc)) from exc
    if not isinstance(data, dict):
        raise MalformedMarkerError("marker %s is not a JSON object" % p)
    if not isinstance(data.get("cap_active"), bool):
        raise MalformedMarkerError("marker %s: cap_active missing or not a bool" % p)
    if not _is_finite_number(data.get("cap_value")):
        raise MalformedMarkerError("marker %s: cap_value missing or not a finite number" % p)
    return data


def marker_matches_config(marker, resolution):
    """None when the marker agrees with the resolved config (or the marker is absent);
    else a human-readable mismatch reason. cap_value is compared exactly, and only when the
    marker declares the cap active -- an inactive marker's value is a record, not an
    operative constraint on an OFF process."""
    if marker is None:
        return None
    active = resolution.state == "on"
    if marker["cap_active"] != active:
        return ("marker declares cap_active=%s but the resolved config is %s"
                % (marker["cap_active"], "ON" if active else "OFF"))
    if active and float(marker["cap_value"]) != float(resolution.value):
        return ("marker declares cap_value=%r but the resolved config value is %r"
                % (marker["cap_value"], resolution.value))
    return None


def watchlist_cap_mismatch(marker, envelope):
    """Consumer-side per-cycle comparison: None when the fresh watchlist envelope is
    consistent with the marker; else a fail-closed refusal reason beginning with
    'watchlist_cap_mismatch'.
    marker None -> no check (the check ships inert until activation writes the marker)."""
    if marker is None:
        return None
    key = envelope.get("score_cap") if isinstance(envelope, dict) else None
    if marker["cap_active"]:
        if not isinstance(key, dict) or key.get("active") is not True:
            return ("watchlist_cap_mismatch: marker declares the cap active but the watchlist "
                    "carries no active top-level score_cap (legacy/uncapped writer?)")
        if not _is_finite_number(key.get("value")) or float(key["value"]) != float(marker["cap_value"]):
            return ("watchlist_cap_mismatch: watchlist score_cap value %r != marker cap_value %r"
                    % (key.get("value"), marker["cap_value"]))
        return None
    if key is not None:
        return ("watchlist_cap_mismatch: marker declares the cap INACTIVE but the watchlist "
                "carries a top-level score_cap %r" % (key,))
    return None

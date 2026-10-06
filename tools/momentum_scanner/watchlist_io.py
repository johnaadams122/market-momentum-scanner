"""Single source of the watchlist.json v2 wire shape: the atomic write, the top-level envelope, and
the per-candidate verdict stamp. Both the EDGAR/Finnhub writer (run_premarket_scan) and the movers
writer (movers_watchlist) build on these so the v2 contract has exactly one implementation."""
import json
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone, timedelta

from momentum_scanner import config

# Windows: os.replace raises PermissionError (WinError 5) if the destination is open by a reader
# (NTFS default open has no FILE_SHARE_DELETE). Retry briefly -- read_text() closes immediately,
# so the contention window is tiny. Semantics are identical: a reader sees old-complete or
# new-complete, never a partial write.
_REPLACE_RETRIES = 20
_REPLACE_RETRY_DELAY_S = 0.002


def atomic_write_json(path, payload):
    """Temp file in the same dir -> flush -> os.replace. allow_nan=False rejects NaN/inf so a reader
    never sees invalid JSON. On any error the temp file is removed and the prior file is untouched."""
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, allow_nan=False)
        for attempt in range(_REPLACE_RETRIES):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt >= _REPLACE_RETRIES - 1:
                    raise
                time.sleep(_REPLACE_RETRY_DELAY_S)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def v2_envelope(candidate_dicts, *, scan_id=None, scanned_at=None, status="ok", confirm_overflow=0,
                error=None, score_cap=None):
    """`score_cap`: top-level cap provenance dict
    ({"active": True, "value": <cap>}) passed ONLY by the movers writer when the cap is ON, so
    every published cap-era artifact -- including a zero-candidate scan -- self-labels. None
    (the default, and the premarket writer always) omits the key: OFF output is byte-identical."""
    scan_id = scan_id or str(uuid.uuid4())
    scanned_at = scanned_at or datetime.now(timezone.utc)
    env = {
        "schema_version": 2,
        "scan_id": scan_id,
    }
    if score_cap is not None:
        env["score_cap"] = score_cap
    env.update({
        "scanned_at": scanned_at.isoformat(),
        "expires_at": (scanned_at + timedelta(minutes=config.WATCHLIST_TTL_MINUTES)).isoformat(),
        "status": status,
        "confirm_overflow": confirm_overflow,
        "candidates": candidate_dicts,
    })
    if error is not None:
        env["error"] = error
    return env


def stamp_verdict(d, verdict):
    v = verdict
    d["gate_decision"] = v.decision if v else "REJECT"
    d["catalyst_label"] = v.label if v else "no-data"
    d["gate_reason"] = v.reason if v else "not evaluated"
    d["catalyst_confidence"] = v.confidence if v else None
    d["catalyst_source"] = v.source if v else "rule"                       # provenance (rule|llm) -- unchanged
    d["dilution_flag"] = getattr(v, "dilution_flag", None) if v else None
    src = getattr(v, "content_source", None) if v else None
    d["catalyst_content_source"] = src if src else "none"
    # `catalyst_score` (the isolated bonus) is written from the candidate by the writer (asdict
    # already includes it because MoverCandidate declares the field); default 0.0 if unset.
    d.setdefault("catalyst_score", 0.0)

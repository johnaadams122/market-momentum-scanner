'Movers watchlist writer: builds and atomically writes the v2 envelope for judged movers candidates.'
from dataclasses import asdict

from momentum_scanner import config
from momentum_scanner import watchlist_io

_CAP_ROW_KEYS = ("score_uncapped", "score_cap_value")


def build_movers_watchlist_dict(candidates, verdicts, *, scan_id=None, scanned_at=None, status="ok",
                                confirm_overflow=0, error=None):
    cap_cfg = config.resolve_score_cap()
    if cap_cfg.state == "invalid":
        raise config.ScoreCapConfigError(cap_cfg.reason)
    cand_dicts = []
    for c in candidates:
        v = verdicts.get(c.ticker)
        if v is None:
            continue                                  # unjudged -> absent (never written/traded)
        d = asdict(c)                                 # includes score, edgar_signal, catalyst_score
        for k in _CAP_ROW_KEYS:                       # None = not stamped (cap OFF, or non-finite E)
            if d.get(k) is None:
                d.pop(k, None)
        d["catalyst"] = getattr(v, "headline", None)      # the judge-stamped catalyst headline
        watchlist_io.stamp_verdict(d, v)
        cand_dicts.append(d)
    score_cap = ({"active": True, "value": cap_cfg.value} if cap_cfg.state == "on" else None)
    return watchlist_io.v2_envelope(cand_dicts, scan_id=scan_id, scanned_at=scanned_at, status=status,
                                    confirm_overflow=confirm_overflow, error=error,
                                    score_cap=score_cap)


def write_movers_watchlist_json(path, candidates, verdicts, *, scan_id=None, scanned_at=None,
                                status="ok", confirm_overflow=0, error=None):
    payload = build_movers_watchlist_dict(candidates, verdicts, scan_id=scan_id, scanned_at=scanned_at,
                                          status=status, confirm_overflow=confirm_overflow, error=error)
    watchlist_io.atomic_write_json(path, payload)
    return payload

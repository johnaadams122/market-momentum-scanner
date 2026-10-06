"""Lazy per-ticker float cache (slow-changing). yfinance floatShares fetched on first use, refreshed
weekly. Missing/failed/non-finite -> None (discovery keeps the candidate with a flag; the low-float pillar
applies only when float is known)."""
import json
import logging
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path

import yfinance as yf

from momentum_scanner import config

_log = logging.getLogger(__name__)


def _fresh(entry, now_utc, refresh_days):
    try:
        return (now_utc - datetime.fromisoformat(entry["as_of"])).days < refresh_days
    except (KeyError, ValueError, TypeError):
        return False


def get_float(ticker, *, cache, fetch_fn, now_utc, refresh_days=None):
    rd = config.UNIVERSE_REFRESH_DAYS if refresh_days is None else refresh_days
    entry = cache.get(ticker)
    if entry and _fresh(entry, now_utc, rd):
        return entry["float"]
    try:
        val = fetch_fn(ticker)
    except Exception:
        # Transient fetch error: do NOT write to cache so the next call retries.
        # A week-long negative cache from a yfinance blip would freeze float as unknown.
        return None
    if val is not None:
        try:
            val = float(val)
            if not math.isfinite(val) or val <= 0:
                val = None
        except (TypeError, ValueError):
            val = None
    # Fetched value that is genuinely None/non-finite/<=0 is cached (data is bad, not transient).
    cache[ticker] = {"float": val, "as_of": now_utc.isoformat()}
    return val


def fetch_float(ticker):
    """yfinance floatShares (slow-changing). Returns None for a missing/non-finite/<=0
    value (get_float CACHES that as a known-bad). LETS yfinance exceptions RAISE so get_float treats a
    fetch error as TRANSIENT and does NOT negative-cache it."""
    fs = yf.Ticker(ticker).info.get("floatShares")     # a raise here propagates to get_float (transient)
    if fs is None:
        return None
    try:
        fs = float(fs)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(fs) or fs <= 0:
        return None
    return fs


def load_float_cache(state_dir):
    """Warm-load state/fundamentals.json into the dict get_float mutates (avoids a yfinance burst on
    restart). Missing/corrupt -> {} (rebuilds lazily)."""
    path = Path(state_dir) / config.FUNDAMENTALS_CACHE_FILE
    if not path.exists():
        return {}
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (ValueError, OSError):
        _log.warning("fundamentals: corrupt float cache at %s; starting empty", path)
        return {}


def save_float_cache(state_dir, cache):
    """Atomically persist the float cache dict so a restart warm-starts."""
    path = Path(state_dir) / config.FUNDAMENTALS_CACHE_FILE
    d = str(path.parent)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(cache, fh)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

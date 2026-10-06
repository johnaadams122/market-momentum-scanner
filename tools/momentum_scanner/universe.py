"""Build/cache the tradeable common-stock universe from the free nasdaqtrader SymbolDirectory
(nasdaqlisted.txt + otherlisted.txt -- which have DIFFERENT column orders). Column lookup is
header-driven so both layouts parse correctly. Warrants/units/rights/preferred are dropped by
TYPE DESCRIPTOR (the text after the last ' - ' in the Security Name -- nasdaqtrader convention).
ADRs and depositary shares are KEPT (valid low-float movers). Cached to state/universe.json,
refreshed weekly; a fetch failure falls back to the last cache (never empty) and logs a warning."""
import json
import logging
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

from momentum_scanner import config
from momentum_scanner.edgar_rss import normalize_ticker

_log = logging.getLogger(__name__)

# Match only the TYPE DESCRIPTOR (after the last ' - '), not the full name.
# ADRs / depositary shares are intentionally excluded: low-float small-cap ADRs are valid movers.
_NONCOMMON_DESC = re.compile(r"\b(warrants?|units?|rights?|preferred)\b", re.I)


def _index_map(header):
    return {name.strip(): i for i, name in enumerate(header.split("|"))}


def _col(cols, idx, default="N"):
    return cols[idx].strip() if (idx is not None and idx < len(cols)) else default


def parse_symbol_directory(text):
    lines = (text or "").splitlines()
    if not lines:
        return []
    idx = _index_map(lines[0])
    sym_i = idx.get("Symbol", idx.get("ACT Symbol"))
    if sym_i is None:
        return []
    etf_i, test_i, name_i = idx.get("ETF"), idx.get("Test Issue"), idx.get("Security Name")
    out, seen = [], set()
    for line in lines[1:]:
        if "|" not in line or line.startswith("File Creation"):
            continue
        cols = line.split("|")
        if sym_i >= len(cols):
            continue
        symbol = cols[sym_i].strip()
        if not symbol:
            continue
        if _col(cols, etf_i) == "Y" or _col(cols, test_i) == "Y":
            continue
        name = _col(cols, name_i, "")
        desc = name.rsplit(" - ", 1)[1] if " - " in name else ""
        if _NONCOMMON_DESC.search(desc):
            continue
        t = normalize_ticker(symbol)
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _cache_path(state_dir):
    return Path(state_dir) / "universe.json"


def _atomic_write(path, payload):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def load_universe(state_dir, *, fetch_fn, now_utc):
    path = _cache_path(state_dir)
    cached = None
    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cached = None
    if cached and cached.get("as_of"):
        try:
            if (now_utc - datetime.fromisoformat(cached["as_of"])).days < config.UNIVERSE_REFRESH_DAYS:
                return list(cached.get("symbols", []))
        except (ValueError, TypeError):
            pass
    try:
        symbols, seen = [], set()
        for name in config.UNIVERSE_SOURCES:
            for t in parse_symbol_directory(fetch_fn(name)):
                if t not in seen:
                    seen.add(t)
                    symbols.append(t)
        if not symbols:
            _log.warning("universe: all sources parsed to 0 symbols -- falling back to stale cache")
            raise ValueError("empty universe parse")
    except Exception as exc:
        if cached and cached.get("symbols"):
            _log.warning(
                "universe: fetch/parse failed (%s: %s), using stale cache (%d symbols)",
                type(exc).__name__, exc, len(cached["symbols"]),
            )
            return list(cached["symbols"])     # fetch failed -> last cache, never empty
        raise
    _atomic_write(path, {"as_of": now_utc.isoformat(), "symbols": symbols})
    return symbols

"""Chunked Schwab batch /quotes adapter. RTH reads the regular quote block; premarket reads the
extended-hours block ONLY (absent extended -> last/volume None -> Tier-1 skips the symbol, never a
regular-hours fallback). avg_daily_volume = quote fundamental.avg10DaysVolume (no
prebuild, no cold-start). token_fn() and HTTP both normalize to ScannerNetworkError so a whole-chunk
or auth failure fails the scan closed; per-symbol non-finite/garbled fields -> None, never raise."""
import math
from dataclasses import dataclass
from datetime import datetime, timezone

from momentum_scanner import config
from momentum_scanner.errors import ScannerNetworkError


@dataclass(frozen=True)
class QuoteSnapshot:
    ticker: str
    last: float | None
    prev_close: float | None
    total_volume: int | None
    day_high: float | None
    avg_daily_volume: float | None
    quote_time: "datetime | None"
    source: str                 # "regular" | "extended"


def _num(d, key):
    v = d.get(key) if isinstance(d, dict) else None
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _vol(d, key):
    f = _num(d, key)
    return int(f) if f is not None else None


def _qtime(block):
    ms = (block.get("quoteTime") or block.get("tradeTime")) if isinstance(block, dict) else None
    try:
        return datetime.fromtimestamp(int(ms) / 1000.0, tz=timezone.utc) if ms else None
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _get_chunk(session, token, syms):
    last_exc = None
    for _ in range(2):
        try:
            resp = session.get(config.SCHWAB_QUOTES_URL,
                               headers={"Authorization": "Bearer %s" % token},
                               params={"symbols": ",".join(syms)},
                               timeout=config.SCANNER_HTTP_TIMEOUT)
            status = getattr(resp, "status_code", 200)
            if status >= 400:
                last_exc = ScannerNetworkError("Schwab quotes HTTP %s" % status)
                continue
            return resp.json() or {}
        except ScannerNetworkError as exc:
            last_exc = exc
        except Exception as exc:
            last_exc = ScannerNetworkError("Schwab quotes failed: %s" % type(exc).__name__)
    raise last_exc


def batch_quotes(symbols, *, token_fn, session, now_utc, premarket):
    try:
        token = token_fn()
    except Exception as exc:
        raise ScannerNetworkError("Schwab token failed: %s" % type(exc).__name__) from None
    out = {}
    for chunk in _chunks(list(symbols), config.SCHWAB_QUOTE_CHUNK):
        data = _get_chunk(session, token, chunk)      # raises -> caller fail-closes
        for sym in chunk:
            row = data.get(sym) or {}
            quote = row.get("quote") or {}
            fund = row.get("fundamental") or {}
            if premarket:
                src = row.get("extended") or {}
                last, vol, source = _num(src, "lastPrice"), _vol(src, "totalVolume"), "extended"
            else:
                src = quote
                last, vol, source = _num(src, "lastPrice"), _vol(src, "totalVolume"), "regular"
            out[sym] = QuoteSnapshot(
                ticker=sym, last=last, prev_close=_num(quote, "closePrice"),
                total_volume=vol, day_high=_num(quote, "highPrice"),
                avg_daily_volume=_num(fund, "avg10DaysVolume"),
                quote_time=_qtime(src), source=source)   # timestamp from the SAME block as last/vol
    return out

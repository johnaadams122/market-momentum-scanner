"""Production fetcher for the nasdaqtrader SymbolDirectory files -- the fetch_fn that
universe.load_universe injects. HTTP GET the pipe-delimited nasdaqlisted.txt /
otherlisted.txt; any HTTP/connection/timeout failure raises ScannerNetworkError so load_universe falls
back to its last cache (or, with no cache, raises -- the daemon main() wraps that into a status:error
watchlist). Network only; the column-driven parsing lives in universe.py."""
import requests

from momentum_scanner import config
from momentum_scanner.errors import ScannerNetworkError


def fetch_symbol_directory(name, *, session=None):
    sess = session or requests
    url = f"{config.NASDAQTRADER_SYMDIR_URL}/{name}"
    try:
        resp = sess.get(url, timeout=config.SCANNER_HTTP_TIMEOUT)
    except Exception as exc:                       # connection/timeout/etc -> systemic
        raise ScannerNetworkError(f"nasdaqtrader fetch failed for {name}: {type(exc).__name__}")
    status = getattr(resp, "status_code", 200)
    if status >= 400:
        raise ScannerNetworkError(f"nasdaqtrader HTTP {status} for {name}")
    return resp.text

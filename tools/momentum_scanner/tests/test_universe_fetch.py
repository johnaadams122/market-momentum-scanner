import pytest
from momentum_scanner.universe_fetch import fetch_symbol_directory
from momentum_scanner.errors import ScannerNetworkError
from momentum_scanner import universe, config


class _Resp:
    def __init__(self, status, text=""):
        self.status_code = status
        self.text = text


class _Session:
    def __init__(self, resp=None, exc=None):
        self._resp = resp
        self._exc = exc
        self.calls = []
    def get(self, url, timeout=None):
        self.calls.append((url, timeout))
        if self._exc:
            raise self._exc
        return self._resp


def test_returns_text_and_hits_the_right_url():
    s = _Session(_Resp(200, "Symbol|Security Name\nAAA|Alpha Inc\n"))
    out = fetch_symbol_directory("nasdaqlisted.txt", session=s)
    assert "AAA" in out
    assert s.calls[0][0] == f"{config.NASDAQTRADER_SYMDIR_URL}/nasdaqlisted.txt"
    assert s.calls[0][1] == config.SCANNER_HTTP_TIMEOUT


def test_http_error_raises_scanner_network_error():
    with pytest.raises(ScannerNetworkError):
        fetch_symbol_directory("otherlisted.txt", session=_Session(_Resp(503)))


def test_connection_error_raises_scanner_network_error():
    with pytest.raises(ScannerNetworkError):
        fetch_symbol_directory("nasdaqlisted.txt", session=_Session(exc=OSError("conn reset")))


def test_load_universe_with_this_fetcher_parses_offline(tmp_path):
    """End-to-end with universe.load_universe + a fake session (no network)."""
    nasdaq = "Symbol|Security Name|ETF|Test Issue\nAAA|Alpha Inc - Common Stock|N|N\n"
    other = "ACT Symbol|Security Name|ETF|Test Issue\nBBB|Beta Co - Common Stock|N|N\n"
    def fetch_fn(name):
        text = nasdaq if name == "nasdaqlisted.txt" else other
        return fetch_symbol_directory(name, session=_Session(_Resp(200, text)))
    from datetime import datetime, timezone
    syms = universe.load_universe(tmp_path, fetch_fn=fetch_fn, now_utc=datetime(2026, 6, 30, tzinfo=timezone.utc))
    assert "AAA" in syms and "BBB" in syms

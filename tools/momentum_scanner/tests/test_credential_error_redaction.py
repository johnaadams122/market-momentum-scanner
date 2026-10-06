"""Credential-bearing request failures must never echo the credential.

A synthetic secret ending in CRLF makes the real `requests` library raise InvalidHeader whose text
contains the whole header value. These tests drive the REAL `requests` path (no network is reached,
the header is rejected before any connection) and assert the secret appears nowhere in the raised
message, the chained traceback, or the runner output and files.
"""
import json
import sys
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from momentum_scanner.errors import ScannerNetworkError

ET = ZoneInfo("America/New_York")
FAKE_VALUE = "synthetic-fake-bearer-alpha"
BAD = FAKE_VALUE + "\r\n"            # trailing CRLF -> requests raises InvalidHeader echoing the value


def _full_text(exc):
    """Message plus the complete chained traceback text (what a logger with exc_info would print)."""
    return str(exc) + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def _assert_clean(exc_info, *secrets):
    text = _full_text(exc_info.value)
    for s in secrets:
        assert s not in text


def test_requests_really_echoes_the_secret_so_the_tests_below_have_power():
    import requests
    with pytest.raises(requests.exceptions.InvalidHeader) as ei:
        requests.get("http://synthetic.invalid/x", headers={"Authorization": "Bearer " + BAD}, timeout=1)
    assert FAKE_VALUE in str(ei.value)


def test_schwab_data_request_failure_hides_bearer_token():
    from momentum_scanner.schwab_data import SchwabDataProvider
    p = SchwabDataProvider(token_fn=lambda: BAD)
    with pytest.raises(ScannerNetworkError) as ei:
        p._get({"symbol": "ABCD"})
    assert "InvalidHeader" in str(ei.value)
    _assert_clean(ei, FAKE_VALUE)


def test_schwab_data_session_exception_text_hidden():
    from momentum_scanner.schwab_data import SchwabDataProvider

    class S:
        def get(self, *a, **k):
            raise RuntimeError("boom Bearer " + FAKE_VALUE)
    with pytest.raises(ScannerNetworkError) as ei:
        SchwabDataProvider(token_fn=lambda: FAKE_VALUE, session=S())._get({})
    assert "RuntimeError" in str(ei.value)
    _assert_clean(ei, FAKE_VALUE)


def test_schwab_data_token_provider_failure_text_hidden():
    from momentum_scanner.schwab_data import SchwabDataProvider

    def bad_token_fn():
        raise RuntimeError("refresh failed for " + FAKE_VALUE)
    with pytest.raises(ScannerNetworkError) as ei:
        SchwabDataProvider(token_fn=bad_token_fn)._get({})
    _assert_clean(ei, FAKE_VALUE)


def test_schwab_quotes_failure_hides_bearer_token():
    import requests
    from momentum_scanner.adapters import schwab_quotes as sq
    now = datetime(2026, 6, 29, 14, 0, tzinfo=ET)
    with pytest.raises(ScannerNetworkError) as ei:
        sq.batch_quotes(["ABCD"], token_fn=lambda: BAD, session=requests, now_utc=now, premarket=False)
    _assert_clean(ei, FAKE_VALUE)


def test_schwab_quotes_token_provider_failure_hides_text():
    from momentum_scanner.adapters import schwab_quotes as sq
    now = datetime(2026, 6, 29, 14, 0, tzinfo=ET)

    def bad_token_fn():
        raise RuntimeError("refresh failed for " + FAKE_VALUE)
    with pytest.raises(ScannerNetworkError) as ei:
        sq.batch_quotes(["ABCD"], token_fn=bad_token_fn, session=None, now_utc=now, premarket=False)
    _assert_clean(ei, FAKE_VALUE)


def test_finnhub_failure_hides_api_key():
    from momentum_scanner.finnhub_news import FinnhubClient
    with pytest.raises(ScannerNetworkError) as ei:
        FinnhubClient(api_key=BAD)._get("/news", {})
    _assert_clean(ei, FAKE_VALUE)


def test_alpaca_failure_hides_key_and_secret():
    from momentum_scanner.alpaca_news import AlpacaNewsClient
    other = "synthetic-fake-secret-beta"
    with pytest.raises(ScannerNetworkError) as ei:
        AlpacaNewsClient(api_key=BAD, api_secret=other + "\r\n")._get({})
    _assert_clean(ei, FAKE_VALUE, other)


def test_edgar_fetch_failure_hides_contact_user_agent(monkeypatch):
    from momentum_scanner import edgar_rss
    contact = "synthetic-contact-for-tests"
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", contact + "\r\nX-Synthetic: 1")
    with pytest.raises(ScannerNetworkError) as ei:
        edgar_rss._get("http://synthetic.invalid/x")
    assert "InvalidHeader" in str(ei.value)
    _assert_clean(ei, contact)


def test_edgar_fetch_failure_reports_http_status_without_exception_text():
    import requests
    from momentum_scanner import edgar_rss

    class R:
        status_code = 503

        def raise_for_status(self):
            raise requests.exceptions.HTTPError("503 for url with Bearer " + FAKE_VALUE, response=self)
    with pytest.raises(ScannerNetworkError) as ei:
        edgar_rss._get("http://synthetic.invalid/x", fetcher=lambda u, **k: R())
    assert "503" in str(ei.value)
    _assert_clean(ei, FAKE_VALUE)


def test_run_premarket_scan_prints_and_writes_no_token(tmp_path, monkeypatch, capsys):
    """End to end through the CLI: stdout (FATAL line) and the error watchlist stay token-free."""
    import momentum_scanner.run_premarket_scan as r
    from momentum_scanner.schwab_data import SchwabDataProvider
    out = tmp_path / "watchlist.json"

    class _Filing:
        ticker, cik, url, headline, company_name = "ABCD", "111", "http://x", "h", "ABCD"
        filed_at = datetime(2026, 6, 29, 8, 0, tzinfo=ET)
        form = "8-K"
    monkeypatch.setattr(r, "_load_ticker_map", lambda: {})
    monkeypatch.setattr(r, "fetch_recent_8k_filings", lambda **k: [_Filing()])

    def scan(tickers, **k):
        SchwabDataProvider(token_fn=lambda: BAD).get_premarket("ABCD", k["now_et"])
    monkeypatch.setattr(r, "scan_candidates", scan)
    monkeypatch.setattr(sys, "argv", ["prog", "--json-out", str(out)])
    with pytest.raises(SystemExit) as e:
        r.main()
    assert e.value.code != 0
    captured = capsys.readouterr()
    assert FAKE_VALUE not in captured.out and FAKE_VALUE not in captured.err
    assert "FATAL" in captured.out
    assert FAKE_VALUE not in out.read_text(encoding="utf-8")
    assert json.loads(out.read_text(encoding="utf-8"))["status"] == "error"


def test_movers_scan_error_watchlist_carries_no_token(tmp_path):
    """The movers/daemon path stores str(exc) in the error watchlist: it must be token-free."""
    import requests
    from momentum_scanner import movers_scan
    out = tmp_path / "watchlist.json"
    now = datetime(2026, 6, 29, 8, 0, tzinfo=ET)
    movers_scan.run_movers_scan(["ABCD"], token_fn=lambda: BAD, session=requests, float_cache={},
                                float_fetch_fn=lambda s: None, now_utc=now, now_et=now,
                                premarket=True, out_path=str(out))
    assert FAKE_VALUE not in out.read_text(encoding="utf-8")

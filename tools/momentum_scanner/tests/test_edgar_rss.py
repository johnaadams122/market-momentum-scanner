from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta


def _cik(n):
    """SEC CIKs are 10-digit, zero-padded strings; build one from its short number."""
    return str(n).zfill(10)


SAMPLE_TICKER_JSON = {
    "0": {"cik_str": "320193", "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": "789019", "ticker": "MSFT", "title": "MICROSOFT CORP"},
    "2": {"cik_str": "1", "ticker": "TEST", "title": "Test Corp"},
}


def _make_atom_xml():
    t1 = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    t2 = (datetime.now(timezone.utc) - timedelta(minutes=60)).isoformat()
    return f"""\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>8-K - APPLE INC ({_cik(320193)}) (Filer)</title>
    <link rel="alternate" href="https://www.sec.gov/filing/123"/>
    <updated>{t1}</updated>
  </entry>
  <entry>
    <title>8-K - UNKNOWN CORP ({_cik(99999)}) (Filer)</title>
    <link rel="alternate" href="https://www.sec.gov/filing/456"/>
    <updated>{t2}</updated>
  </entry>
  <entry>
    <title>8-K - OLD CORP ({_cik(111111)}) (Filer)</title>
    <link rel="alternate" href="https://www.sec.gov/filing/789"/>
    <updated>2024-01-01T09:00:00-05:00</updated>
  </entry>
</feed>
"""


def _mock_get(url, **kwargs):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    if "company_tickers" in url:
        resp.json.return_value = SAMPLE_TICKER_JSON
    else:
        resp.text = _make_atom_xml()
    return resp


def test_load_ticker_map_returns_zero_padded_cik_keys():
    with patch("momentum_scanner.edgar_rss.requests.get", side_effect=_mock_get):
        from momentum_scanner.edgar_rss import _load_ticker_map
        result = _load_ticker_map()
    assert result[_cik(320193)] == "AAPL"
    assert result[_cik(789019)] == "MSFT"
    assert result[_cik(1)] == "TEST"


def test_fetch_recent_filings_returns_matching_ticker():
    with patch("momentum_scanner.edgar_rss.requests.get", side_effect=_mock_get):
        from momentum_scanner.edgar_rss import fetch_recent_8k_filings
        ticker_map = {_cik(320193): "AAPL", _cik(789019): "MSFT"}
        results = fetch_recent_8k_filings(lookback_minutes=120, ticker_map=ticker_map)

    tickers = [r.ticker for r in results]
    assert "AAPL" in tickers


def test_fetch_recent_filings_sets_none_ticker_for_unknown_cik():
    with patch("momentum_scanner.edgar_rss.requests.get", side_effect=_mock_get):
        from momentum_scanner.edgar_rss import fetch_recent_8k_filings
        ticker_map = {_cik(320193): "AAPL"}  # 99999 not in map
        results = fetch_recent_8k_filings(lookback_minutes=120, ticker_map=ticker_map)

    unknown = [r for r in results if r.cik == _cik(99999)]
    assert len(unknown) == 1
    assert unknown[0].ticker is None


def test_fetch_recent_filings_excludes_old_entries():
    with patch("momentum_scanner.edgar_rss.requests.get", side_effect=_mock_get):
        from momentum_scanner.edgar_rss import fetch_recent_8k_filings
        ticker_map = {}
        results = fetch_recent_8k_filings(lookback_minutes=60, ticker_map=ticker_map)

    # The 2024-01-01 entry is too old regardless of lookback_minutes at test time
    ciks = [r.cik for r in results]
    assert _cik(111111) not in ciks


def test_fetch_recent_filings_extracts_company_name():
    with patch("momentum_scanner.edgar_rss.requests.get", side_effect=_mock_get):
        from momentum_scanner.edgar_rss import fetch_recent_8k_filings
        ticker_map = {_cik(320193): "AAPL"}
        results = fetch_recent_8k_filings(lookback_minutes=120, ticker_map=ticker_map)

    aapl = next(r for r in results if r.ticker == "AAPL")
    assert aapl.company_name == "APPLE INC"


def test_fetch_recent_filings_includes_url():
    with patch("momentum_scanner.edgar_rss.requests.get", side_effect=_mock_get):
        from momentum_scanner.edgar_rss import fetch_recent_8k_filings
        ticker_map = {_cik(320193): "AAPL"}
        results = fetch_recent_8k_filings(lookback_minutes=120, ticker_map=ticker_map)

    aapl = next(r for r in results if r.ticker == "AAPL")
    assert "sec.gov" in aapl.url


def test_load_ticker_map_sends_user_agent_header():
    with patch("momentum_scanner.edgar_rss.requests.get", side_effect=_mock_get) as mock_get:
        from momentum_scanner.edgar_rss import _load_ticker_map
        _load_ticker_map()
    call_kwargs = mock_get.call_args[1]
    assert "User-Agent" in call_kwargs.get("headers", {})
    assert "market-momentum-synthetic-contact" == call_kwargs["headers"]["User-Agent"]


# ---- per-ticker submissions detail + body fetch + error wrapping ----

import requests
from momentum_scanner.errors import ScannerNetworkError

_NOW = datetime(2026, 6, 27, 13, 0, tzinfo=timezone.utc)

# Synthetic filer CIK 111 and its accession numbers (<filer>-<yy>-<sequence>).
_CIK_111 = _cik(111)
_ACC_44 = _CIK_111 + "-26-000044"
_ACC_45 = _CIK_111 + "-26-000045"
_ACC_46 = _CIK_111 + "-26-000046"
# EDGAR index URL: /data/<int cik>/<accession without dashes>/<accession>-index.htm
_INDEX_URL_46 = ("https://www.sec.gov/Archives/edgar/data/111/"
                 + _ACC_46.replace("-", "") + "/" + _ACC_46 + "-index.htm")


def _subs(form, items, dates, accs, docs):
    return {"filings": {"recent": {
        "form": form, "items": items, "filingDate": dates,
        "accessionNumber": accs, "primaryDocument": docs,
    }}}


def _ok(json_value=None, text_value=None):
    r = MagicMock()
    r.raise_for_status = MagicMock()
    if json_value is not None:
        r.json.return_value = json_value
    if text_value is not None:
        r.text = text_value
    return r


def test_archive_url_construction():
    from momentum_scanner.edgar_rss import archive_url
    # the path segment is the accession number 0000111-26-000045 with its dashes removed
    assert archive_url(_CIK_111, "0000111-26-000045", "a8k.htm") == \
        "https://www.sec.gov/Archives/edgar/data/111/" "0000111" "26" "000045" "/a8k.htm"


def test_accession_from_url():
    from momentum_scanner.edgar_rss import _accession_from_url
    assert _accession_from_url(_INDEX_URL_46) == _ACC_46
    assert _accession_from_url("https://example.com/none") is None


def test_fetch_filing_detail_parses_items_and_doc_url():
    from momentum_scanner.edgar_rss import fetch_filing_detail
    subs = _subs(["8-K", "424B5", "10-Q"], ["8.01", "", ""],
                 ["2026-06-27", "2026-06-26", "2026-05-01"],
                 ["0000111-26-000045", "0000111-26-000044", "0000111-26-000040"],
                 ["a8k.htm", "p424.htm", "q.htm"])
    d = fetch_filing_detail(_CIK_111, now=_NOW, fetcher=lambda u, **kw: _ok(json_value=subs))
    assert d.triggering_items == {"8.01"}
    assert d.triggering_doc_url.endswith("a8k.htm")
    assert any(f["form"] == "424B5" for f in d.recent_filings)


def test_fetch_filing_detail_selects_triggering_by_accession():
    from momentum_scanner.edgar_rss import fetch_filing_detail
    subs = _subs(["8-K", "8-K", "424B5"], ["5.02", "8.01", ""],
                 ["2026-06-27", "2026-06-27", "2026-06-26"],
                 [_ACC_45, _ACC_46, _ACC_44],
                 ["a.htm", "b.htm", "p.htm"])
    trig_url = _INDEX_URL_46
    d = fetch_filing_detail(_CIK_111, trigger_url=trig_url, now=_NOW,
                            fetcher=lambda u, **kw: _ok(json_value=subs))
    assert d.triggering_items == {"8.01"}  # the accession-matched 8-K, not the first-listed (5.02)
    assert d.triggering_doc_url.endswith("b.htm")


def test_fetch_filing_detail_fallback_most_recent_8k():
    from momentum_scanner.edgar_rss import fetch_filing_detail
    subs = _subs(["8-K", "8-K"], ["5.02", "8.01"], ["2026-06-27", "2026-06-27"],
                 ["0000111-26-000045", "0000111-26-000046"], ["a.htm", "b.htm"])
    d = fetch_filing_detail(_CIK_111, trigger_url=None, now=_NOW,
                            fetcher=lambda u, **kw: _ok(json_value=subs))
    assert d.triggering_items == {"5.02"}  # first-listed (most recent) when no accession given


def test_fetch_filing_detail_drops_filings_outside_365d():
    from momentum_scanner.edgar_rss import fetch_filing_detail
    subs = _subs(["8-K", "424B5"], ["8.01", ""],
                 ["2026-06-27", "2025-01-01"],  # second is >365d before _NOW
                 ["0000111-26-000045", "0000111-25-000001"], ["a.htm", "old.htm"])
    d = fetch_filing_detail(_CIK_111, now=_NOW, fetcher=lambda u, **kw: _ok(json_value=subs))
    assert all(f["form"] != "424B5" for f in d.recent_filings)  # old offering dropped


def test_fetch_filing_detail_network_error_raises():
    from momentum_scanner.edgar_rss import fetch_filing_detail
    def boom(u, **kw): raise requests.exceptions.ConnectionError("down")
    try:
        fetch_filing_detail(_CIK_111, now=_NOW, fetcher=boom)
        assert False, "expected ScannerNetworkError"
    except ScannerNetworkError:
        pass


def test_fetch_filing_detail_malformed_raises():
    from momentum_scanner.edgar_rss import fetch_filing_detail
    try:
        fetch_filing_detail(_CIK_111, now=_NOW, fetcher=lambda u, **kw: _ok(json_value={"nope": 1}))
        assert False, "expected ScannerNetworkError"
    except ScannerNetworkError:
        pass


def test_fetch_filing_body_returns_text_and_caps():
    from momentum_scanner.edgar_rss import fetch_filing_body
    out = fetch_filing_body("http://x/a.htm", max_bytes=5, fetcher=lambda u, **kw: _ok(text_value="FDA approval granted"))
    assert out == "FDA a"  # capped to 5 bytes


def test_fetch_filing_body_network_error_raises():
    from momentum_scanner.edgar_rss import fetch_filing_body
    def boom(u, **kw): raise requests.exceptions.Timeout("slow")
    try:
        fetch_filing_body("http://x/a.htm", fetcher=boom)
        assert False, "expected ScannerNetworkError"
    except ScannerNetworkError:
        pass


def test_load_ticker_map_network_error_wrapped():
    from momentum_scanner.edgar_rss import _load_ticker_map
    with patch("momentum_scanner.edgar_rss.requests.get",
               side_effect=requests.exceptions.ConnectionError("down")):
        try:
            _load_ticker_map()
            assert False, "expected ScannerNetworkError"
        except ScannerNetworkError:
            pass


def test_fetch_recent_8k_network_error_wrapped():
    from momentum_scanner.edgar_rss import fetch_recent_8k_filings
    with patch("momentum_scanner.edgar_rss.requests.get",
               side_effect=requests.exceptions.ConnectionError("down")):
        try:
            fetch_recent_8k_filings(ticker_map={})
            assert False, "expected ScannerNetworkError"
        except ScannerNetworkError:
            pass


def test_fetch_filing_detail_badrow_offering_kept_in_window():
    # An offering row (424B5) with a bad/missing filingDate must NOT vanish (fail-closed).
    from momentum_scanner.edgar_rss import fetch_filing_detail
    subs = _subs(["8-K", "424B5"], ["8.01", ""], ["2026-06-27", ""],
                 [_ACC_45, _ACC_44], ["a.htm", "p.htm"])
    d = fetch_filing_detail(_CIK_111, now=_NOW, fetcher=lambda u, **kw: _ok(json_value=subs))
    assert any(f["form"] == "424B5" for f in d.recent_filings)

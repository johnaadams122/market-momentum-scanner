"""archive_url must only ever build a URL for ONE file under the issuer's own filing folder."""
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from momentum_scanner.edgar_rss import archive_url, fetch_filing_detail
from momentum_scanner.errors import ScannerNetworkError

CIK = "0000111"
ACC = "0000111-26-000045"


@pytest.mark.parametrize("doc", [
    "../x.htm", "..", "a/b.htm", "/abs.htm", "a\\b.htm", "https://synthetic.invalid/x.htm",
    "x.htm?y=1", "x.htm#f", "x y.htm", "", ".hidden", "a%2Fb.htm", "x.htm\r\n", None,
])
def test_archive_url_rejects_unsafe_primary_doc(doc):
    with pytest.raises(ScannerNetworkError):
        archive_url(CIK, ACC, doc)


@pytest.mark.parametrize("acc", ["0000111-26-000045/../x", "../..", "", "00001-26-0000 45", "abc"])
def test_archive_url_rejects_unsafe_accession(acc):
    with pytest.raises(ScannerNetworkError):
        archive_url(CIK, acc, "a8k.htm")


def test_archive_url_accepts_ordinary_sec_file_names():
    for doc in ("a8k.htm", "d123456d8k.htm", "form8-k.htm", "tm_2026-8k.htm", "ex99_1.txt"):
        assert archive_url(CIK, ACC, doc).endswith("/" + doc)


def test_fetch_filing_detail_drops_unsafe_doc_url_instead_of_fetching_it():
    subs = {"filings": {"recent": {
        "form": ["8-K"], "items": ["8.01"], "filingDate": ["2026-06-27"],
        "accessionNumber": [ACC], "primaryDocument": ["../../evil.htm"]}}}
    r = MagicMock()
    r.raise_for_status = MagicMock()
    r.json.return_value = subs
    d = fetch_filing_detail(CIK, now=datetime(2026, 6, 29, tzinfo=timezone.utc),
                            fetcher=lambda u, **kw: r)
    assert d.triggering_doc_url is None

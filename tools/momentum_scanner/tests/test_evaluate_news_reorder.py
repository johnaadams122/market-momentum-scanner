"""evaluate_news ordering -- CIK-unresolvable vs EDGAR-outage split, news-dilution
independent of CIK, and the EDGAR-as-content branch."""
import datetime as _dt
from dataclasses import dataclass, field

import pytest

from momentum_scanner import catalyst_evaluator as ce
from momentum_scanner.errors import ScannerError

# Synthetic SEC CIKs (10-digit zero-padded identifiers).
CIK_111 = "111".zfill(10)


class _Cand:
    def __init__(self, ticker="AAA"):
        self.ticker = ticker
        self.float_shares = 5_000_000
        self.rel_volume = 5.0


class _News:
    def __init__(self, headline, summary=""):
        self.headline = headline
        self.summary = summary
        self.published = _dt.datetime(2026, 7, 2, 13, tzinfo=_dt.timezone.utc)


def _real_judge(headline, summary):
    return {"label": "real", "confidence": 0.9, "rationale": "ok"}


def test_unresolvable_cik_flags_unknown_not_reject():
    v = ce.evaluate_news(_Cand(), [_News("Company wins USD 50M contract")],
                         news_judge=_real_judge,
                         ticker_to_ciks=lambda: {},               # 0 CIKs -> unresolvable
                         fetch_detail=lambda *a, **k: None,
                         fetch_body=lambda *a, **k: "")
    assert v.decision == "PROCEED"
    assert v.dilution_flag == "unknown"


def test_news_dilution_regex_fires_with_no_cik():
    v = ce.evaluate_news(_Cand(), [_News("Announces pricing of public offering")],
                         news_judge=_real_judge,
                         ticker_to_ciks=lambda: {},               # no CIK -> classify_dilution cannot run
                         fetch_detail=lambda *a, **k: None, fetch_body=lambda *a, **k: "")
    # news regex still catches the offering even though EDGAR could not identify the filer
    assert v.dilution_flag == "active_offering"


def test_edgar_outage_stays_fail_closed():
    def boom():
        raise ScannerError("edgar 503")
    with pytest.raises(ScannerError):
        ce.evaluate_news(_Cand(), [_News("real news")], news_judge=_real_judge,
                         ticker_to_ciks=boom, raise_systemic=True)


def test_edgar_outage_non_systemic_rejects_no_data():
    def boom():
        raise ScannerError("edgar 503")
    v = ce.evaluate_news(_Cand(), [_News("real news")], news_judge=_real_judge,
                         ticker_to_ciks=boom, raise_systemic=False)
    assert v.decision == "REJECT" and v.label == "no-data"


# ---- EDGAR-as-content branch ----

@dataclass
class _Detail:
    cik: str = CIK_111
    triggering_items: set = field(default_factory=set)
    triggering_doc_url: str | None = None
    recent_filings: list = field(default_factory=list)


def _real_filing_judge(headline, items, body):
    return {"label": "real", "confidence": 0.9, "is_fixed_price_buyout": False, "rationale": "8-K catalyst"}


def test_edgar_8k_content_no_finnhub_news_proceeds():
    detail = _Detail(triggering_items={"8.01"}, triggering_doc_url="http://x/8k.htm",
                     recent_filings=[{"form": "8-K", "items": {"8.01"},
                                      "filing_date": _dt.date(2026, 7, 2),
                                      "accession": f"{CIK_111}-26-000001", "primary_doc": "8k.htm"}])
    v = ce.evaluate_news(_Cand(), [], news_judge=_real_judge, filing_judge=_real_filing_judge,
                         ticker_to_ciks=lambda: {"AAA": {CIK_111}},
                         fetch_detail=lambda cik, **k: detail,
                         fetch_body=lambda url, **k: "material 8-K catalyst text",
                         now=_dt.datetime(2026, 7, 2, 13, tzinfo=_dt.timezone.utc))
    assert v.decision == "PROCEED"
    assert v.content_source == "edgar"


def test_no_catalyst_item_not_foreign_rejects_no_data_none_source():
    detail = _Detail(triggering_items=set(), triggering_doc_url=None,
                     recent_filings=[{"form": "10-Q", "items": set(),
                                      "filing_date": _dt.date(2026, 7, 2),
                                      "accession": f"{CIK_111}-26-000001", "primary_doc": "q.htm"}])
    v = ce.evaluate_news(_Cand(), [], news_judge=_real_judge, filing_judge=_real_filing_judge,
                         ticker_to_ciks=lambda: {"AAA": {CIK_111}},
                         fetch_detail=lambda cik, **k: detail,
                         fetch_body=lambda url, **k: "routine",
                         now=_dt.datetime(2026, 7, 2, 13, tzinfo=_dt.timezone.utc))
    assert v.decision == "REJECT" and v.label == "no-data"
    assert v.content_source == "none"


def test_edgar_content_path_buyout_guard_hard_rejects():
    """_edgar_8k_recall's fixed-price-buyout guard must
    hard-reject on the EDGAR-CONTENT path (no Finnhub news -> STEP 2 recall), not just the
    news-judge path (the buyout branch inside catalyst_evaluator._edgar_8k_recall)."""
    detail = _Detail(triggering_items={"1.01"}, triggering_doc_url="http://x/8k.htm",
                     recent_filings=[{"form": "8-K", "items": {"1.01"},
                                      "filing_date": _dt.date(2026, 7, 2),
                                      "accession": f"{CIK_111}-26-000001", "primary_doc": "8k.htm"}])

    def _buyout_filing_judge(headline, items, body):
        return {"label": "real", "confidence": 0.9, "is_fixed_price_buyout": True,
                "rationale": "fixed-price cash buyout"}

    v = ce.evaluate_news(_Cand(), [], news_judge=_real_judge, filing_judge=_buyout_filing_judge,
                         ticker_to_ciks=lambda: {"AAA": {CIK_111}},
                         fetch_detail=lambda cik, **k: detail,
                         fetch_body=lambda url, **k: "fixed-price buyout at USD 5.00 per share",
                         now=_dt.datetime(2026, 7, 2, 13, tzinfo=_dt.timezone.utc))
    assert v.decision == "REJECT"
    assert v.label == "avoid"
    assert v.content_source == "edgar"

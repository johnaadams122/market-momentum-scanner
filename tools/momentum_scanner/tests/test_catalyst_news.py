"""evaluate_news -- Finnhub-triggered catalyst gate.

EDGAR dilution signals come from a single-CIK lookup; an unresolvable CIK (zero or multiple
matches) demotes the dilution flag to "unknown" rather than rejecting, and a systemic EDGAR outage
stays fail-closed. Dilution flags and the judged label feed the catalyst score on this path.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone

from momentum_scanner.errors import ScannerError

NOW = datetime(2026, 6, 29, 13, 0, tzinfo=timezone.utc)


@dataclass
class Cand:
    ticker: str = "AAA"
    float_shares: float = 2_000_000
    rel_volume: float = 6.0


@dataclass
class NI:
    headline: str = "FDA grants approval"
    summary: str = "Company received FDA approval for its lead drug"
    published: datetime = NOW


@dataclass
class FakeDetail:
    recent_filings: list = field(default_factory=list)
    # _edgar_8k_recall reads these on every detail-not-None path, so this stub must mirror
    # the real FilingDetail shape (see the recall-tests' own `_detail()` helper further down this file).
    triggering_items: set = field(default_factory=set)
    triggering_doc_url: str | None = None


def _real_judge(headline, summary):
    return {"label": "real", "confidence": 0.9, "is_fixed_price_buyout": False, "rationale": "FDA"}


def _eval(cand=None, news=None, *, ciks=None, detail=None, judge=_real_judge, fetch_detail=None):
    from momentum_scanner.catalyst_evaluator import evaluate_news
    cand = cand or Cand()
    news = news if news is not None else [NI()]
    ciks_map = {"AAA": ciks if ciks is not None else {CIK_111}}
    fd = fetch_detail or (lambda cik, **k: detail if detail is not None else FakeDetail([]))
    return evaluate_news(cand, news, news_judge=judge, ticker_to_ciks=lambda: ciks_map,
                         fetch_detail=fd, fetch_body=lambda url, **k: "", now=NOW)


def test_clean_single_cik_real_judge_proceeds():
    v = _eval()
    assert v.decision == "PROCEED" and v.label == "real"


def test_content_source_reflects_alpaca_provider():
    """A NewsItem fetched via AlpacaNewsClient carries provider="alpaca"; the verdict's
    content_source must reflect that provider rather than a fixed "finnhub" value."""
    item = NI()
    item.provider = "alpaca"    # NI is a plain (non-frozen) dataclass, so this is a normal attribute set
    v = _eval(news=[item])
    assert v.decision == "PROCEED"
    assert v.content_source == "alpaca"


def test_content_source_falls_back_to_finnhub_when_provider_absent():
    """Test doubles / callers that never set .provider get content_source="finnhub" -- the default
    when the provider attribute is absent."""
    v = _eval(news=[NI()])
    assert v.decision == "PROCEED"
    assert v.content_source == "finnhub"


def test_missing_cik_demotes_to_unknown_still_proceeds():
    # An unresolvable CIK (0 candidates) does not hard-reject -- it demotes the
    # dilution flag to "unknown" and continues; real news still confirmed by the judge -> PROCEED.
    v = _eval(ciks=set())
    assert v.decision == "PROCEED" and v.dilution_flag == "unknown"


def test_ambiguous_cik_demotes_to_unknown_still_proceeds():
    # Same demotion for >1 CIK candidates (still "unresolvable" per EdgarContext.detail_for).
    v = _eval(ciks={CIK_111, CIK_222})
    assert v.decision == "PROCEED" and v.dilution_flag == "unknown"


def test_edgar_shelf_form_flags_not_pump():
    detail = FakeDetail([{"filing_date": NOW.date(), "form": "S-3", "items": set()}])
    v = _eval(detail=detail)                      # default _real_judge
    assert v.decision == "PROCEED" and v.label == "real"
    assert v.dilution_flag == "shelf_risk"


def test_edgar_detail_error_rejects_no_data():
    def boom(cik, **k):
        raise ScannerError("edgar down")
    v = _eval(fetch_detail=boom)
    assert v.decision == "REJECT" and v.label == "no-data"


def test_cross_item_news_offering_flags_not_pump():
    news = [NI(headline="Great results"), NI(headline="Company announces pricing of public offering")]
    v = _eval(news=news)                          # default _real_judge
    assert v.decision == "PROCEED"
    assert v.dilution_flag == "active_offering"   # news offering language escalates the flag


def test_no_confirmed_news_rejects():
    v = _eval(news=[])
    assert v.decision == "REJECT" and v.label == "no-data"


def test_fixed_price_buyout_avoids():
    judge = lambda h, s: {"label": "real", "confidence": 0.9, "is_fixed_price_buyout": True, "rationale": "buyout"}
    v = _eval(judge=judge)
    assert v.decision == "REJECT" and v.label == "avoid"


def test_judge_non_real_now_proceeds_with_label():
    # The news judge is not a gate -- a non-"real" label feeds the score
    # (catalyst_score) instead of rejecting. See test_evaluate_news_softened.py.
    judge = lambda h, s: {"label": "pump", "confidence": 0.8, "is_fixed_price_buyout": False, "rationale": "promo"}
    v = _eval(judge=judge)
    assert v.decision == "PROCEED" and v.label == "pump"


def test_low_confidence_now_proceeds():
    # The CONF_MIN floor does not gate the news path -- confidence flows to the score
    # instead of rejecting. See test_evaluate_news_softened.py.
    judge = lambda h, s: {"label": "real", "confidence": 0.5, "is_fixed_price_buyout": False, "rationale": "weak"}
    v = _eval(judge=judge)
    assert v.decision == "PROCEED" and v.label == "real" and v.confidence == 0.5


def test_news_dilution_regex_matches_pr_language():
    from momentum_scanner import config
    for s in ["announces pricing of public offering", "gross proceeds of USD 10M", "private placement",
              "registered direct offering"]:
        assert config.NEWS_DILUTION_REGEX.search(s), s


import pytest
from datetime import date
from momentum_scanner.edgar_rss import FilingDetail
from momentum_scanner.errors import ScannerNetworkError
from momentum_scanner.catalyst_evaluator import evaluate_news, is_foreign_issuer

# Synthetic SEC CIKs (10-digit zero-padded identifiers).
CIK_111 = "111".zfill(10)
CIK_222 = "222".zfill(10)

_REAL_FILING = lambda h, items, body: {"label": "real", "confidence": 0.9,
                                       "is_fixed_price_buyout": False, "rationale": "6-K catalyst"}


def _detail(rows, cik=CIK_111):
    return FilingDetail(cik=cik, triggering_items=set(), triggering_doc_url=None, recent_filings=rows)


def _frow(form, days_ago=1, acc=f"{CIK_111}-26-000001", doc="d.htm"):
    fd = NOW.date() if days_ago == 0 else date.fromordinal(NOW.date().toordinal() - days_ago)
    return {"form": form, "items": set(), "filing_date": fd, "accession": acc, "primary_doc": doc}


def _recall_eval(detail, *, fetch_body, filing_judge=_REAL_FILING, raise_systemic=False):
    return evaluate_news(Cand(), [], news_judge=_real_judge, filing_judge=filing_judge,
                         ticker_to_ciks=lambda: {"AAA": {CIK_111}},
                         fetch_detail=lambda cik, **k: detail, fetch_body=fetch_body,
                         now=NOW, raise_systemic=raise_systemic)


def test_is_foreign_issuer():
    assert is_foreign_issuer(_detail([_frow("6-K")])) is True
    assert is_foreign_issuer(_detail([_frow("6-K"), _frow("8-K")])) is False
    assert is_foreign_issuer(_detail([_frow("10-K")])) is False


def test_foreign_6k_recall_proceeds():
    v = _recall_eval(_detail([_frow("6-K", days_ago=1)]),
                     fetch_body=lambda url, **k: "material contract awarded")
    assert v.decision == "PROCEED" and v.label == "real"


def test_foreign_body_fetch_outage_propagates():
    # A body-fetch outage on the foreign path must PROPAGATE (fail-closed), never a cached REJECT.
    # (classify_dilution's Pass-3 6-K body scan hits it first; the recall re-raise is the defensive
    # backstop. Either way the observable contract is: raises -> ticker unjudged.)
    def boom(url, **k):
        raise ScannerNetworkError("edgar down")
    with pytest.raises(ScannerNetworkError):
        _recall_eval(_detail([_frow("6-K", days_ago=1)]), fetch_body=boom, raise_systemic=True)


def test_foreign_no_recent_6k_falls_through():
    v = _recall_eval(_detail([_frow("6-K", days_ago=30)]), fetch_body=lambda url, **k: "x")
    assert v.decision == "REJECT" and v.label == "no-data"


def test_domestic_no_8k_item_still_no_data():
    v = _recall_eval(_detail([_frow("8-K", days_ago=1)]), fetch_body=lambda url, **k: "x")
    assert v.decision == "REJECT" and v.label == "no-data"


def test_news_reverse_split_only_maps_to_reverse_split_history():
    # Empty EDGAR flag; news mentions ONLY a reverse split -> reverse_split_history (factor 1.0),
    # NOT active_offering (0.3). Pins the severity-monotonic news escalation.
    news = [NI(headline="Company effects a 1-for-10 reverse stock split")]
    v = _eval(news=news)
    assert v.decision == "PROCEED"
    assert v.dilution_flag == "reverse_split_history"


def test_foreign_recall_body_fetch_reraise_reached():
    # Reach the recall's OWN re-raise: classify Pass-3 body scan returns benign text (flag None), then
    # the recall's subsequent fetch raises -> propagates under raise_systemic (not the classify pass).
    calls = {"n": 0}
    def fb(url, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return "routine update"       # classify Pass-3: no dilution language
        raise ScannerNetworkError("edgar down on recall fetch")
    with pytest.raises(ScannerNetworkError):
        _recall_eval(_detail([_frow("6-K", days_ago=1)]), fetch_body=fb, raise_systemic=True)
    assert calls["n"] == 2                 # classify fetched once; recall reached + raised on the 2nd


def test_foreign_recall_buyout_rejects_avoid():
    fj = lambda h, i, b: {"label": "real", "confidence": 0.9, "is_fixed_price_buyout": True, "rationale": "buyout"}
    v = _recall_eval(_detail([_frow("6-K", days_ago=1)]), fetch_body=lambda u, **k: "acquired for cash",
                     filing_judge=fj)
    assert v.decision == "REJECT" and v.label == "avoid"


def test_foreign_recall_judge_raises_falls_through():
    def fj(h, i, b):
        raise RuntimeError("judge crash")
    v = _recall_eval(_detail([_frow("6-K", days_ago=1)]), fetch_body=lambda u, **k: "material contract",
                     filing_judge=fj)
    assert v.decision == "REJECT" and v.label == "no-data"


def test_foreign_recall_non_real_falls_through():
    fj = lambda h, i, b: {"label": "pump", "confidence": 0.9, "is_fixed_price_buyout": False, "rationale": "promo"}
    v = _recall_eval(_detail([_frow("6-K", days_ago=1)]), fetch_body=lambda u, **k: "some promo", filing_judge=fj)
    assert v.decision == "REJECT" and v.label == "no-data"

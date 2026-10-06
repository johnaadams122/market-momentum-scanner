"""Selectivity: market-roundup filter. Roundup/listicle headlines ("N Stocks Moving In X
Session", "top gainers/losers") are not single-name catalysts and must not drive a PROCEED; a
roundup-only news set becomes "no confirmed news". Real single-name PRs must never be filtered."""
import pytest

from momentum_scanner import catalyst_evaluator as ce


# ---- is_roundup pure-function (the risky regex; lock catch + no-false-positive) ----

ROUNDUPS = [
    "12 Example Sector Stocks Moving In Thursday's Pre-Market Session",
    "12 Sample Industry Stocks Moving In Tuesday's After-Market Session",
    "11 Placeholder Sector Stocks Moving In Tuesday's Intraday Session",
    "10 Example Sector Stocks With Whale Alerts In Today's Session",
    "Why Acme Corp Shares Are Trading Higher By Over 40%; Here Are 20 Stocks Moving Premarket",
    "Acme, Example Co And Other Big Stocks Moving Lower In Wednesday's Pre-Market Session",
    "gap up and gap down stocks",
    "top gainers and losers pre-market",
]

REAL_CATALYSTS = [
    "Acme Corp Authorizes USD 20 Million Stock Buyback Program",
    "Example Co Announces Its Inclusion In Sample Partner Ecosystem",
    "Acme Corp To Acquire Example Co Subsidiary Sample Labs For 12.5M Shares Of AAA",
    "Sample Biotech Reaches Regulatory Milestone for Respiratory Test Platform",
    "Example Pharma's Widget Joins Sample Developer Program",
    "Acme Digital Investing USD 65M In A Midwest Site To Be Converted Into AI Data Center",
    "Example Therapeutics Engages Sample Advisory Associates",
    "AAA wins FDA approval",
]


@pytest.mark.parametrize("h", ROUNDUPS)
def test_is_roundup_catches_listicles(h):
    assert ce.is_roundup(h) is True


@pytest.mark.parametrize("h", REAL_CATALYSTS)
def test_is_roundup_does_not_flag_real_catalysts(h):
    assert ce.is_roundup(h) is False


def test_is_roundup_handles_none_and_empty():
    assert ce.is_roundup(None) is False
    assert ce.is_roundup("") is False


# ---- integration into evaluate_news (mirrors test_evaluate_news_softened scaffolding) ----

class _Cand:
    def __init__(self):
        self.ticker = "AAA"
        self.float_shares = 5_000_000
        self.rel_volume = 5.0


class _News:
    def __init__(self, headline, summary=""):
        import datetime as _dt
        self.headline = headline
        self.summary = summary
        self.published = _dt.datetime(2026, 7, 2, 13, tzinfo=_dt.timezone.utc)


def _one_cik():
    return {"AAA": {"1"}}


def _detail_ok(*a, **k):
    class D:
        recent_filings = []
        triggering_items = set()
    return D()


def _eval(news, judge_label="real"):
    return ce.evaluate_news(_Cand(), news,
                            news_judge=lambda h, s: {"label": judge_label, "confidence": 0.9},
                            ticker_to_ciks=_one_cik, fetch_detail=_detail_ok,
                            fetch_body=lambda *a, **k: "")


def test_roundup_only_news_becomes_no_confirmed_news():
    v = _eval([_News("12 Example Sector Stocks Moving In Tuesday's Pre-Market Session")])
    assert v.decision == "REJECT" and v.label == "no-data"
    assert v.content_source == "none"          # filtered -> no-confirmed-news floor, not STEP 3


def test_real_single_name_headline_still_proceeds():
    v = _eval([_News("AAA Authorizes USD 20 Million Stock Buyback Program")])
    assert v.decision == "PROCEED"
    assert v.content_source == "finnhub"       # real catalyst survives the filter


def test_mixed_roundup_and_real_keeps_the_real_one():
    v = _eval([_News("20 Stocks Moving Premarket"), _News("AAA wins FDA approval")])
    assert v.decision == "PROCEED"
    assert v.content_source == "finnhub"


def test_filter_can_be_disabled(monkeypatch):
    monkeypatch.setattr(ce.config, "ROUNDUP_FILTER_ENABLED", False)
    v = _eval([_News("12 Stocks Moving Premarket")], judge_label="no-data")
    assert v.content_source == "finnhub"       # not filtered -> reaches STEP 3 (proceeds on any content)

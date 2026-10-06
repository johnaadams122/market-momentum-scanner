"""STEP 3 of evaluate_news is soft -- the news judge does not GATE (real@>=CONF_MIN
PROCEED / else REJECT). Instead label + confidence flow to the score; only the fixed-price-buyout
guard and judge-output/confidence validation remain hard rejects. Every STEP-3 outcome is stamped
content_source="finnhub". The STEP-2 no-content-any-source reject (content_source="none")
must still work unchanged -- this file asserts the soft STEP 3 does not disturb it."""
from momentum_scanner import catalyst_evaluator as ce


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


def _detail_ok(*a, **k):  # minimal detail object with no dilution
    class D:  # noqa
        recent_filings = []
        triggering_items = set()
    return D()


def test_no_data_judge_now_proceeds_with_low_source(monkeypatch):
    v = ce.evaluate_news(_Cand(), [_News("some vague headline")],
                         news_judge=lambda h, s: {"label": "no-data", "confidence": 0.9},
                         ticker_to_ciks=_one_cik, fetch_detail=_detail_ok,
                         fetch_body=lambda *a, **k: "")
    assert v.decision == "PROCEED"           # judge bar dropped
    assert v.label == "no-data"
    assert v.content_source == "finnhub"


def test_pump_label_now_proceeds(monkeypatch):
    v = ce.evaluate_news(_Cand(), [_News("promo headline")],
                         news_judge=lambda h, s: {"label": "pump", "confidence": 0.85},
                         ticker_to_ciks=_one_cik, fetch_detail=_detail_ok,
                         fetch_body=lambda *a, **k: "")
    assert v.decision == "PROCEED"           # non-real label no longer rejects
    assert v.label == "pump"
    assert v.content_source == "finnhub"


def test_low_confidence_now_proceeds(monkeypatch):
    v = ce.evaluate_news(_Cand(), [_News("real-ish")],
                         news_judge=lambda h, s: {"label": "real", "confidence": 0.4},
                         ticker_to_ciks=_one_cik, fetch_detail=_detail_ok,
                         fetch_body=lambda *a, **k: "")
    assert v.decision == "PROCEED"           # below the old CONF_MIN floor, still proceeds
    assert v.label == "real"
    assert v.confidence == 0.4
    assert v.content_source == "finnhub"


def test_fixed_price_buyout_still_hard_rejects(monkeypatch):
    v = ce.evaluate_news(_Cand(), [_News("acquired at fixed price")],
                         news_judge=lambda h, s: {"label": "real", "confidence": 0.95,
                                                  "is_fixed_price_buyout": True},
                         ticker_to_ciks=_one_cik, fetch_detail=_detail_ok,
                         fetch_body=lambda *a, **k: "")
    assert v.decision == "REJECT" and v.label == "avoid"
    assert v.content_source == "finnhub"


def test_judge_output_validation_still_hard_rejects(monkeypatch):
    # malformed judge output (bad label) fails _validate_judge_obj -> still a hard reject.
    v = ce.evaluate_news(_Cand(), [_News("some headline")],
                         news_judge=lambda h, s: {"label": "maybe", "confidence": 0.9},
                         ticker_to_ciks=_one_cik, fetch_detail=_detail_ok,
                         fetch_body=lambda *a, **k: "")
    assert v.decision == "REJECT" and v.label == "no-data"
    assert v.content_source == "finnhub"


def test_judge_raises_still_hard_rejects(monkeypatch):
    def _boom(h, s):
        raise RuntimeError("judge crash")
    v = ce.evaluate_news(_Cand(), [_News("some headline")],
                         news_judge=_boom,
                         ticker_to_ciks=_one_cik, fetch_detail=_detail_ok,
                         fetch_body=lambda *a, **k: "")
    assert v.decision == "REJECT" and v.label == "no-data"
    assert v.content_source == "finnhub"


def test_no_content_any_source_still_rejects(monkeypatch):
    v = ce.evaluate_news(_Cand(), [],
                         news_judge=lambda h, s: {"label": "real", "confidence": 0.9},
                         ticker_to_ciks=_one_cik, fetch_detail=_detail_ok,
                         fetch_body=lambda *a, **k: "")
    assert v.decision == "REJECT" and v.label == "no-data"
    assert v.content_source == "none"

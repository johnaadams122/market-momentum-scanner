from datetime import datetime, timezone, timedelta

from momentum_scanner.edgar_rss import FilingDetail
from momentum_scanner.catalyst_evaluator import instrument_eligible, dilution_veto, default_judge
import momentum_scanner.catalyst_evaluator as ce

NOW = datetime(2026, 6, 27, 13, 0, tzinfo=timezone.utc)

# Synthetic filer. SEC CIKs are 10-digit, zero-padded strings, built here from the short number.
CIK = "111".zfill(10)
ACCESSION = CIK + "-26-000045"   # accession number: <filer>-<yy>-<sequence>


def _detail(rows, trig_items=None, trig_url="http://x/d.htm"):
    """rows: list of (form, items_list, days_ago)."""
    recent = []
    for form, items, days_ago in rows:
        recent.append({
            "form": form, "items": set(items),
            "filing_date": (NOW.date() - timedelta(days=days_ago)),
            "accession": ACCESSION, "primary_doc": "d.htm",
        })
    return FilingDetail(CIK, set(trig_items or []), trig_url, recent)


# ---- instrument eligibility ----

def test_instrument_eligible_basic():
    assert instrument_eligible("ABCD")
    assert not instrument_eligible("ABCDW")   # warrant
    assert not instrument_eligible("ABCDU")   # unit
    assert not instrument_eligible("")


# ---- dilution veto: offering forms / items within 14d ----

def test_veto_recent_offering_form():
    assert dilution_veto(_detail([("424B5", [], 2)]), now=NOW, fetch_body=lambda u: "x")


def test_veto_8k_dilution_item():
    assert dilution_veto(_detail([("8-K", ["3.02"], 1)]), now=NOW, fetch_body=lambda u: "x")


def test_no_veto_offering_outside_14d():
    assert dilution_veto(_detail([("424B5", [], 60)]), now=NOW, fetch_body=lambda u: "x") is None


# ---- reverse split within 365d (body-confirmed), outside the 14d window ----

def test_veto_reverse_split_within_365d():
    assert dilution_veto(_detail([("DEF 14A", [], 60)]), now=NOW,
                         fetch_body=lambda u: "Board approved a 1-for-10 reverse stock split")


def test_no_veto_reverse_split_form_without_language():
    assert dilution_veto(_detail([("DEF 14A", [], 60)]), now=NOW,
                         fetch_body=lambda u: "Annual meeting routine matters") is None


def test_veto_item_503_reverse_split_within_365d():
    assert dilution_veto(_detail([("8-K", ["5.03"], 90)]), now=NOW,
                         fetch_body=lambda u: "effected a reverse stock split")


# ---- body regex on ambiguous 8-K within 14d ----

def test_veto_body_regex_on_8k_8point01():
    assert dilution_veto(_detail([("8-K", ["8.01"], 1)]), now=NOW,
                         fetch_body=lambda u: "announces registered direct offering")


# ---- 6-K is body-scan-only, never blanket-vetoed ----

def test_no_veto_clean_6k():
    assert dilution_veto(_detail([("6-K", [], 1)]), now=NOW, fetch_body=lambda u: "routine update") is None


def test_veto_6k_with_offering_language():
    assert dilution_veto(_detail([("6-K", [], 1)]), now=NOW,
                         fetch_body=lambda u: "at-the-market offering commenced")


def test_no_veto_clean_filing():
    assert dilution_veto(_detail([("10-Q", [], 1)]), now=NOW, fetch_body=lambda u: "routine results") is None


# ---- bounded default judge ----

def test_default_judge_valid(monkeypatch):
    monkeypatch.setattr(ce, "_ollama_generate",
        lambda p: '{"label":"real","confidence":0.9,"is_fixed_price_buyout":false,"rationale":"FDA"}')
    out = default_judge("h", {"8.01"}, "FDA approval")
    assert out["label"] == "real" and out["confidence"] == 0.9 and out["is_fixed_price_buyout"] is False


def test_default_judge_strips_code_fences(monkeypatch):
    monkeypatch.setattr(ce, "_ollama_generate",
        lambda p: '```json\n{"label":"pump","confidence":0.8,"is_fixed_price_buyout":false,"rationale":"x"}\n```')
    assert default_judge("h", {"8.01"}, "x")["label"] == "pump"


def test_default_judge_malformed_none(monkeypatch):
    monkeypatch.setattr(ce, "_ollama_generate", lambda p: "not json at all")
    assert default_judge("h", {"8.01"}, "x") is None


def test_default_judge_bad_label_none(monkeypatch):
    monkeypatch.setattr(ce, "_ollama_generate", lambda p: '{"label":"maybe","confidence":0.9}')
    assert default_judge("h", {"8.01"}, "x") is None


def test_default_judge_nonnumeric_confidence_none(monkeypatch):
    monkeypatch.setattr(ce, "_ollama_generate", lambda p: '{"label":"real","confidence":"high"}')
    assert default_judge("h", {"8.01"}, "x") is None


def test_default_judge_ollama_failure_none(monkeypatch):
    def boom(p):
        raise RuntimeError("ollama down")
    monkeypatch.setattr(ce, "_ollama_generate", boom)
    assert default_judge("h", {"8.01"}, "x") is None


# ---- _ollama_generate: explicit num_ctx + truncate:false (never the server's
# implicit VRAM-based default) ----

class _FakeOllamaResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"response": "ok"}


def test_ollama_generate_sends_explicit_num_ctx_and_truncate_false(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeOllamaResponse()

    monkeypatch.setattr(ce.requests, "post", fake_post)
    out = ce._ollama_generate("hello")
    assert out == "ok"
    assert captured["json"]["truncate"] is False
    assert captured["json"]["options"]["num_ctx"] == ce.config.SCANNER_OLLAMA_NUM_CTX
    assert ce.config.SCANNER_OLLAMA_NUM_CTX > 0


def test_ollama_generate_num_ctx_is_configurable_via_env(monkeypatch):
    # An explicit ceiling must actually be tunable via env, not a hardcoded literal
    # wearing a config name.
    monkeypatch.setattr(ce.config, "SCANNER_OLLAMA_NUM_CTX", 12345)
    captured = {}

    def fake_post(url, json, timeout):
        captured["json"] = json
        return _FakeOllamaResponse()

    monkeypatch.setattr(ce.requests, "post", fake_post)
    ce._ollama_generate("hello")
    assert captured["json"]["options"]["num_ctx"] == 12345


# ---- evaluate(): full fail-closed flow ----

from momentum_scanner.catalyst_evaluator import evaluate
from momentum_scanner.edgar_rss import FilingResult
from momentum_scanner.gap_scanner import ScanResult
from momentum_scanner.errors import ScannerNetworkError

REAL_JUDGE = lambda *a: {"label": "real", "confidence": 0.9, "is_fixed_price_buyout": False, "rationale": "FDA"}


def _cand(ticker="ABCD", float_shares=2_000_000, rel_volume=7.0):
    return ScanResult(ticker=ticker, company_name="Abcd", premarket_price=5.0, prev_close=4.0,
                      gap_pct=0.25, float_shares=float_shares, premarket_volume=7_000_000,
                      baseline_volume=1_000_000, rel_volume=rel_volume)


def _filing(cik=CIK):
    return FilingResult(ticker="ABCD", company_name="Abcd", cik=cik, headline="8-K - Abcd",
                        url="http://x/" + ACCESSION + "-index.htm", filed_at=NOW)


def _detail_for(trig_items, rows=None):
    rows = rows or [("8-K", list(trig_items), 0)]
    recent = [{"form": f, "items": set(it), "filing_date": (NOW.date() - timedelta(days=d)),
               "accession": ACCESSION, "primary_doc": "x.htm"} for f, it, d in rows]
    return FilingDetail(CIK, set(trig_items), "http://x/x.htm", recent)


def _ev(cand, judge, trig_items, rows=None, body="clean catalyst text"):
    return evaluate(cand, _filing(), judge=judge,
                    fetch_detail=lambda cik, **kw: _detail_for(trig_items, rows),
                    fetch_body=lambda u: body, now=NOW)


def test_evaluate_missing_float_rejects():
    v = _ev(_cand(float_shares=None), REAL_JUDGE, {"8.01"}, body="FDA approval")
    assert v.decision == "REJECT" and v.label == "no-data"


def test_evaluate_missing_relvol_rejects():
    v = _ev(_cand(rel_volume=None), REAL_JUDGE, {"8.01"}, body="FDA approval")
    assert v.decision == "REJECT" and v.label == "no-data"


def test_evaluate_ineligible_instrument_rejects():
    v = _ev(_cand(ticker="ABCDW"), REAL_JUDGE, {"8.01"})
    assert v.decision == "REJECT"


def test_evaluate_dilution_item_pump():
    v = _ev(_cand(), REAL_JUDGE, {"3.02"}, rows=[("8-K", ["3.02"], 0)])
    assert v.decision == "REJECT" and v.label == "pump"


def test_evaluate_no_data_item():
    v = _ev(_cand(), REAL_JUDGE, {"5.02"}, rows=[("8-K", ["5.02"], 0)])
    assert v.decision == "REJECT" and v.label == "no-data"


def test_evaluate_real_proceeds():
    v = _ev(_cand(), REAL_JUDGE, {"8.01"}, rows=[("8-K", ["8.01"], 0)], body="FDA approval granted")
    assert v.decision == "PROCEED" and v.label == "real" and v.confidence == 0.9


def test_evaluate_low_confidence_rejects():
    judge = lambda *a: {"label": "real", "confidence": 0.4, "is_fixed_price_buyout": False, "rationale": ""}
    v = _ev(_cand(), judge, {"8.01"}, rows=[("8-K", ["8.01"], 0)])
    assert v.decision == "REJECT" and v.label == "no-data"


def test_evaluate_fixed_price_buyout_avoid():
    judge = lambda *a: {"label": "real", "confidence": 0.95, "is_fixed_price_buyout": True, "rationale": "buyout"}
    v = _ev(_cand(), judge, {"8.01"}, rows=[("8-K", ["8.01"], 0)], body="to be acquired for cash")
    assert v.decision == "REJECT" and v.label == "avoid"


def test_evaluate_detail_failure_no_data():
    def boom(cik, **kw):
        raise ScannerNetworkError("down")
    v = evaluate(_cand(), _filing(), judge=REAL_JUDGE, fetch_detail=boom, fetch_body=lambda u: "x", now=NOW)
    assert v.decision == "REJECT" and v.label == "no-data"


def test_evaluate_body_failure_no_data():
    def boom_body(u):
        raise ScannerNetworkError("body down")
    v = evaluate(_cand(), _filing(), judge=REAL_JUDGE,
                 fetch_detail=lambda cik, **kw: _detail_for({"8.01"}, [("8-K", ["8.01"], 0)]),
                 fetch_body=boom_body, now=NOW)
    assert v.decision == "REJECT" and v.label == "no-data"


def test_evaluate_judge_raises_no_data():
    def boom_judge(*a):
        raise RuntimeError("judge crash")
    v = _ev(_cand(), boom_judge, {"8.01"}, rows=[("8-K", ["8.01"], 0)])
    assert v.decision == "REJECT" and v.label == "no-data"


def test_evaluate_judge_malformed_no_data():
    v = _ev(_cand(), lambda *a: {"label": "maybe"}, {"8.01"}, rows=[("8-K", ["8.01"], 0)])
    assert v.decision == "REJECT" and v.label == "no-data"


def test_evaluate_judge_none_no_data():
    v = _ev(_cand(), lambda *a: None, {"8.01"}, rows=[("8-K", ["8.01"], 0)])
    assert v.decision == "REJECT" and v.label == "no-data"


def test_evaluate_judge_cannot_override_veto():
    # triggering 8.01 body carries offering language -> pump at the veto step, before the judge runs
    v = _ev(_cand(), REAL_JUDGE, {"8.01"}, rows=[("8-K", ["8.01"], 0)],
            body="announces a registered direct offering of shares")
    assert v.decision == "REJECT" and v.label == "pump"


# ---- judge confidence types, reverse-split regex, non-finite inputs ----

def test_default_judge_bool_confidence_none(monkeypatch):
    monkeypatch.setattr(ce, "_ollama_generate",
        lambda p: '{"label":"real","confidence":true,"is_fixed_price_buyout":false,"rationale":"x"}')
    assert default_judge("h", {"8.01"}, "x") is None


def test_default_judge_string_confidence_none(monkeypatch):
    monkeypatch.setattr(ce, "_ollama_generate",
        lambda p: '{"label":"real","confidence":"0.9","is_fixed_price_buyout":false,"rationale":"x"}')
    assert default_judge("h", {"8.01"}, "x") is None


def test_veto_reverse_split_pass_ignores_general_offering_language():
    # DEF 14A 60d (outside 14d offering window); body has offering language but NO reverse split
    # -> must NOT veto (the 365-day pass is reverse-split-specific).
    assert dilution_veto(_detail([("DEF 14A", [], 60)]), now=NOW,
                         fetch_body=lambda u: "registered direct offering of shares") is None


def test_evaluate_nonfinite_relvol_rejects():
    import math as _m
    v = _ev(_cand(rel_volume=_m.nan), REAL_JUDGE, {"8.01"}, rows=[("8-K", ["8.01"], 0)])
    assert v.decision == "REJECT" and v.label == "no-data"


from momentum_scanner.catalyst_evaluator import CatalystVerdict


def test_catalyst_verdict_has_dilution_flag_default_none():
    v = CatalystVerdict("PROCEED", "real", "ok", 0.9, "llm")
    assert v.dilution_flag is None
    v2 = CatalystVerdict("REJECT", "pump", "r", None, "rule", dilution_flag="active_offering")
    assert v2.dilution_flag == "active_offering"


# ---- classify_dilution + dilution_veto wrapper ----

from momentum_scanner.catalyst_evaluator import classify_dilution


def test_classify_active_offering_prospectus():
    assert classify_dilution(_detail([("424B5", [], 2)]), now=NOW, fetch_body=lambda u: "x") == "active_offering"


def test_classify_active_offering_8k_item():
    assert classify_dilution(_detail([("8-K", ["3.02"], 1)]), now=NOW, fetch_body=lambda u: "x") == "active_offering"


def test_classify_shelf_risk():
    assert classify_dilution(_detail([("S-3", [], 3)]), now=NOW, fetch_body=lambda u: "x") == "shelf_risk"


def test_classify_reverse_split_history():
    assert classify_dilution(_detail([("8-K", ["5.03"], 100)]), now=NOW,
                             fetch_body=lambda u: "effected a 1-for-10 reverse stock split") == "reverse_split_history"


def test_classify_most_severe_wins():
    assert classify_dilution(_detail([("S-3", [], 3), ("8-K", ["3.02"], 1)]),
                             now=NOW, fetch_body=lambda u: "x") == "active_offering"


def test_classify_none_when_clean():
    assert classify_dilution(_detail([("8-K", ["2.02"], 1)]), now=NOW, fetch_body=lambda u: "x") is None


def test_dilution_veto_wrapper_returns_flag_or_none():
    assert dilution_veto(_detail([("424B5", [], 2)]), now=NOW, fetch_body=lambda u: "x") == "active_offering"
    assert dilution_veto(_detail([("8-K", ["2.02"], 1)]), now=NOW, fetch_body=lambda u: "x") is None

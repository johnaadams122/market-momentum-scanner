# tests/test_qwen3_judge.py
"""Qwen3:8b news-judge backend (local Ollama) -- a SECOND, independent passive shadow
lane alongside haiku_judge's cloud-based one.

Contract is deliberately identical to catalyst_evaluator.default_news_judge and
haiku_judge.haiku_news_judge: (headline, summary) -> validated dict | None, and NEVER
raises. It shares default_news_judge's prompt block and _parse_judge_json on purpose --
a separate prompt or validator would make the comparison measure the scaffolding rather
than the model, same reasoning as haiku_judge.
"""
import logging

import pytest

from momentum_scanner import qwen3_judge

GOOD = '{"label":"real","confidence":0.9,"is_fixed_price_buyout":false,"rationale":"FDA approval"}'


class _FakeResponse:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        return {"response": self._body}


def test_returns_validated_dict_on_good_response(monkeypatch):
    monkeypatch.setattr(qwen3_judge, "_ollama_generate", lambda p: GOOD)
    v = qwen3_judge.qwen3_news_judge("ABCD wins FDA approval", "primary endpoint met")
    assert v == {"label": "real", "confidence": 0.9, "is_fixed_price_buyout": False,
                 "rationale": "FDA approval"}


def test_returns_none_when_ollama_raises(monkeypatch):
    def boom(prompt):
        raise RuntimeError("ollama down")
    monkeypatch.setattr(qwen3_judge, "_ollama_generate", boom)
    assert qwen3_judge.qwen3_news_judge("h", "s") is None


def test_returns_none_on_unparseable_output(monkeypatch):
    monkeypatch.setattr(qwen3_judge, "_ollama_generate", lambda p: "not json at all")
    assert qwen3_judge.qwen3_news_judge("h", "s") is None


def test_returns_none_on_invalid_label(monkeypatch):
    monkeypatch.setattr(qwen3_judge, "_ollama_generate",
                        lambda p: '{"label":"maybe","confidence":0.9}')
    assert qwen3_judge.qwen3_news_judge("h", "s") is None


def test_returns_none_on_string_confidence(monkeypatch):
    monkeypatch.setattr(qwen3_judge, "_ollama_generate",
                        lambda p: '{"label":"real","confidence":"high"}')
    assert qwen3_judge.qwen3_news_judge("h", "s") is None


def test_prompt_carries_the_news_and_the_untrusted_data_warning(monkeypatch):
    seen = {}
    def capture(prompt):
        seen["p"] = prompt
        return GOOD
    monkeypatch.setattr(qwen3_judge, "_ollama_generate", capture)
    qwen3_judge.qwen3_news_judge("ABCD wins FDA approval", "primary endpoint met")
    assert "ABCD wins FDA approval" in seen["p"]
    assert "primary endpoint met" in seen["p"]
    assert "untrusted" in seen["p"].lower()


def test_summary_is_truncated(monkeypatch):
    seen = {}
    def capture(prompt):
        seen["p"] = prompt
        return GOOD
    monkeypatch.setattr(qwen3_judge, "_ollama_generate", capture)
    qwen3_judge.qwen3_news_judge("h", "x" * 20000)
    assert len(seen["p"]) < 20000


# ---- failure attribution (mirrors haiku_judge: outage vs. bad model output are
# opposite conclusions for a shadow comparison) --------------------------------

def test_take_last_error_reports_an_ollama_failure(monkeypatch):
    def boom(prompt):
        raise RuntimeError("ollama down")
    monkeypatch.setattr(qwen3_judge, "_ollama_generate", boom)
    qwen3_judge.take_last_error()
    assert qwen3_judge.qwen3_news_judge("h", "s") is None
    err = qwen3_judge.take_last_error()
    assert "ollama down" in err
    assert "RuntimeError" in err


def test_take_last_error_reports_a_parse_failure_distinctly(monkeypatch):
    monkeypatch.setattr(qwen3_judge, "_ollama_generate", lambda p: "not json at all")
    qwen3_judge.take_last_error()
    assert qwen3_judge.qwen3_news_judge("h", "s") is None
    assert "unparseable" in qwen3_judge.take_last_error().lower()


def test_take_last_error_is_none_after_success(monkeypatch):
    monkeypatch.setattr(qwen3_judge, "_ollama_generate", lambda p: GOOD)
    qwen3_judge.take_last_error()
    assert qwen3_judge.qwen3_news_judge("h", "s") is not None
    assert qwen3_judge.take_last_error() is None


def test_take_last_error_clears_so_it_cannot_bleed_into_the_next_row(monkeypatch):
    def boom(prompt):
        raise RuntimeError("ollama down")
    monkeypatch.setattr(qwen3_judge, "_ollama_generate", boom)
    qwen3_judge.qwen3_news_judge("h", "s")
    assert qwen3_judge.take_last_error() is not None
    assert qwen3_judge.take_last_error() is None


@pytest.mark.parametrize(
    "detail",
    ["connection refused " + "x" * 1000, "model not found " + "x" * 1000],
    ids=["connection-refused", "model-not-found"],
)
def test_long_ollama_exception_is_bounded_and_traceback_free(monkeypatch, caplog, detail):
    def boom(prompt):
        raise RuntimeError(detail)
    monkeypatch.setattr(qwen3_judge, "_ollama_generate", boom)
    qwen3_judge.take_last_error()
    with caplog.at_level(logging.DEBUG, logger="momentum_scanner.qwen3_judge"):
        assert qwen3_judge.qwen3_news_judge("h", "s") is None
    last_error = qwen3_judge.take_last_error()
    records = [record for record in caplog.records if record.name == "momentum_scanner.qwen3_judge"]
    assert len(last_error) == 500
    assert len(records) == 1
    assert records[0].exc_info is None
    assert records[0].exc_text is None


# ---- the actual Ollama call: SAME num_ctx hardening as catalyst_evaluator, since
# both hit the same local server ------------------------------------------------

def test_ollama_generate_sends_explicit_num_ctx_and_truncate_false(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse("ok")

    monkeypatch.setattr(qwen3_judge.requests, "post", fake_post)
    out = qwen3_judge._ollama_generate("hello")
    assert out == "ok"
    assert captured["json"]["model"] == qwen3_judge.config.QWEN3_MODEL
    assert captured["json"]["truncate"] is False
    assert captured["json"]["options"]["num_ctx"] == qwen3_judge.config.SCANNER_OLLAMA_NUM_CTX
    assert captured["timeout"] == qwen3_judge.config.SHADOW_JUDGE_QWEN3_TIMEOUT

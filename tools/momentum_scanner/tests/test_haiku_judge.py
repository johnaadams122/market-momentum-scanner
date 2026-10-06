# tests/test_haiku_judge.py
"""Haiku news-judge backend (paid Anthropic API) -- the SHADOW half of the A/B.

Contract is deliberately identical to catalyst_evaluator.default_news_judge:
(headline, summary) -> validated dict | None, and NEVER raises. It shares
default_news_judge's parse+validate path on purpose -- giving haiku a stricter or
looser validator than gemma would make the A/B measure the validators rather than
the models.
"""
import logging

import pytest

from momentum_scanner import haiku_judge

GOOD = '{"label":"real","confidence":0.9,"is_fixed_price_buyout":false,"rationale":"FDA approval"}'


def test_returns_validated_dict_on_good_response(monkeypatch):
    monkeypatch.setattr(haiku_judge, "_anthropic_generate", lambda p: GOOD)
    v = haiku_judge.haiku_news_judge("ABCD wins FDA approval", "primary endpoint met")
    assert v == {"label": "real", "confidence": 0.9, "is_fixed_price_buyout": False,
                 "rationale": "FDA approval"}


def test_returns_none_when_api_raises(monkeypatch):
    def boom(prompt):
        raise RuntimeError("anthropic overloaded")
    monkeypatch.setattr(haiku_judge, "_anthropic_generate", boom)
    assert haiku_judge.haiku_news_judge("h", "s") is None


def test_returns_none_on_unparseable_output(monkeypatch):
    monkeypatch.setattr(haiku_judge, "_anthropic_generate", lambda p: "not json at all")
    assert haiku_judge.haiku_news_judge("h", "s") is None


def test_returns_none_on_invalid_label(monkeypatch):
    monkeypatch.setattr(haiku_judge, "_anthropic_generate",
                        lambda p: '{"label":"maybe","confidence":0.9}')
    assert haiku_judge.haiku_news_judge("h", "s") is None


def test_returns_none_on_string_confidence(monkeypatch):
    # Same strictness as gemma's path: a stringy confidence is malformed, not coerced.
    monkeypatch.setattr(haiku_judge, "_anthropic_generate",
                        lambda p: '{"label":"real","confidence":"high"}')
    assert haiku_judge.haiku_news_judge("h", "s") is None


def test_prompt_carries_the_news_and_the_untrusted_data_warning(monkeypatch):
    seen = {}
    def capture(prompt):
        seen["p"] = prompt
        return GOOD
    monkeypatch.setattr(haiku_judge, "_anthropic_generate", capture)
    haiku_judge.haiku_news_judge("ABCD wins FDA approval", "primary endpoint met")
    assert "ABCD wins FDA approval" in seen["p"]
    assert "primary endpoint met" in seen["p"]
    assert "untrusted" in seen["p"].lower()          # prompt-injection guard, same as gemma


def test_summary_is_truncated(monkeypatch):
    seen = {}
    def capture(prompt):
        seen["p"] = prompt
        return GOOD
    monkeypatch.setattr(haiku_judge, "_anthropic_generate", capture)
    haiku_judge.haiku_news_judge("h", "x" * 20000)
    assert len(seen["p"]) < 20000


def test_missing_api_key_returns_none_without_raising(monkeypatch):
    # The launcher may not carry the key yet; that must degrade to "no shadow verdict",
    # never to an exception escaping into the judge batch.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(haiku_judge, "_client", None)
    assert haiku_judge.haiku_news_judge("h", "s") is None


# ---- failure attribution -----------------------------------------------------
# haiku_news_judge swallows every failure to honour the never-raises contract, which
# means a None verdict is ambiguous: API outage, bad key, quota, or a model that
# genuinely emitted junk. For an A/B those are opposite conclusions -- one says "fix
# the plumbing", the other is the measurement itself. take_last_error() disambiguates.

def test_take_last_error_reports_an_api_failure(monkeypatch):
    def boom(prompt):
        raise RuntimeError("anthropic overloaded")
    monkeypatch.setattr(haiku_judge, "_anthropic_generate", boom)
    haiku_judge.take_last_error()                       # clear any residue
    assert haiku_judge.haiku_news_judge("h", "s") is None
    err = haiku_judge.take_last_error()
    assert "anthropic overloaded" in err
    assert "RuntimeError" in err


def test_take_last_error_reports_a_parse_failure_distinctly(monkeypatch):
    monkeypatch.setattr(haiku_judge, "_anthropic_generate", lambda p: "not json at all")
    haiku_judge.take_last_error()
    assert haiku_judge.haiku_news_judge("h", "s") is None
    assert "unparseable" in haiku_judge.take_last_error().lower()


def test_take_last_error_is_none_after_success(monkeypatch):
    monkeypatch.setattr(haiku_judge, "_anthropic_generate", lambda p: GOOD)
    haiku_judge.take_last_error()
    assert haiku_judge.haiku_news_judge("h", "s") is not None
    assert haiku_judge.take_last_error() is None


def test_take_last_error_clears_so_it_cannot_bleed_into_the_next_row(monkeypatch):
    def boom(prompt):
        raise RuntimeError("anthropic overloaded")
    monkeypatch.setattr(haiku_judge, "_anthropic_generate", boom)
    haiku_judge.haiku_news_judge("h", "s")
    assert haiku_judge.take_last_error() is not None
    assert haiku_judge.take_last_error() is None        # second read is empty


def test_last_error_is_redacted_of_key_material(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SECRETVALUE123")
    def boom(prompt):
        raise RuntimeError("401 for key sk-ant-SECRETVALUE123")
    monkeypatch.setattr(haiku_judge, "_anthropic_generate", boom)
    haiku_judge.take_last_error()
    haiku_judge.haiku_news_judge("h", "s")
    assert "SECRETVALUE123" not in haiku_judge.take_last_error()


@pytest.mark.parametrize(
    "detail, secret",
    [
        ("401 for key sk-ant-SYNTHETIC " + "x" * 1000, "sk-ant-SYNTHETIC"),
        ("provider unavailable " + "x" * 1000, None),
    ],
    ids=["key-in-message", "no-key"],
)
def test_long_api_exception_is_bounded_redacted_and_traceback_free(monkeypatch, caplog, detail, secret):
    # Catches an unbounded provider exception being stored, logged, or persisted
    # through the shadow error handoff.  The fake is required because an SDK error
    # is external; the observed behavior remains the real judge boundary.
    def boom(prompt):
        raise RuntimeError(detail)
    monkeypatch.setattr(haiku_judge, "_anthropic_generate", boom)
    haiku_judge.take_last_error()
    with caplog.at_level(logging.DEBUG, logger="momentum_scanner.haiku_judge"):
        assert haiku_judge.haiku_news_judge("h", "s") is None
    last_error = haiku_judge.take_last_error()
    records = [record for record in caplog.records if record.name == "momentum_scanner.haiku_judge"]
    assert len(last_error) == 500
    assert len(records) == 1
    assert len(records[0].getMessage()) <= len("haiku shadow judge call failed: ") + 500
    assert records[0].exc_info is None
    assert records[0].exc_text is None
    if secret:
        assert secret not in caplog.text
        assert secret not in last_error


def test_has_api_key_reports_env_state(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert haiku_judge.has_api_key() is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xxx")
    assert haiku_judge.has_api_key() is True
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    assert haiku_judge.has_api_key() is False        # whitespace-only is not a key

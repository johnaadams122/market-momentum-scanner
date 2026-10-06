# tests/test_shadow_rationale_and_log_dedupe.py
"""Two shadow-judge defects pinned here.

1. RATIONALE TRUNCATION. The [:200] rationale slice in
   catalyst_evaluator._parse_judge_json is SHARED with the deciding gemma path,
   so the more verbose shadow judge's rationales were cut mid-sentence (the cause
   is the slice, not SHADOW_JUDGE_MAX_TOKENS). The rationale IS the deliverable of
   an A/B whose whole purpose is comparing reasoning, so a silent mid-sentence cut
   destroys the thing being measured.

   The fix must not touch what the DECIDING path stores, so the cap became a
   keyword argument that still defaults to 200. Both halves are pinned below.

2. LOG SPAM. resolve_news_judge() is called per candidate from judge_candidate
   (`news_judge or resolve_news_judge()`), not once at startup, so its INFO line
   fired on every candidate and inflated a size-rotated log. Deduping the LOG is
   correct; caching the RESOLUTION is not, because haiku_judge.has_api_key() is
   deliberately read at call time "so a launcher that gains the key mid-life needs
   no code change". Both properties are pinned.
"""
import json
import logging

from momentum_scanner import catalyst_evaluator, haiku_judge as hj, judge
from momentum_scanner.catalyst_evaluator import default_news_judge


def _payload(rationale):
    return json.dumps({"label": "real", "confidence": 0.9,
                       "is_fixed_price_buyout": False, "rationale": rationale})


# --- 1. rationale cap -------------------------------------------------------

def test_deciding_path_rationale_stays_capped_at_200(monkeypatch):
    """The gemma path must be byte-identical to before the fix."""
    monkeypatch.setattr(catalyst_evaluator, "_ollama_generate", lambda p: _payload("y" * 500))
    v = catalyst_evaluator.default_news_judge("h", "s")
    assert len(v["rationale"]) == 200


def test_parse_judge_json_default_cap_is_unchanged():
    v = catalyst_evaluator._parse_judge_json(_payload("z" * 500))
    assert len(v["rationale"]) == 200


def test_shadow_path_keeps_a_longer_rationale(monkeypatch):
    """The observed-only path may keep more, because that text is the A/B's output."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xxx")
    monkeypatch.setattr(hj, "_anthropic_generate", lambda p: _payload("w" * 500))
    v = hj.haiku_news_judge("h", "s")
    assert v is not None
    assert len(v["rationale"]) > 200
    assert v["rationale"].startswith("wwww")


def test_shadow_rationale_is_still_bounded(monkeypatch):
    """Unbounded would put arbitrary model text into an append-only log."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xxx")
    monkeypatch.setattr(hj, "_anthropic_generate", lambda p: _payload("q" * 100_000))
    v = hj.haiku_news_judge("h", "s")
    assert len(v["rationale"]) <= 2000


def test_shadow_label_and_confidence_are_untouched_by_the_wider_cap(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xxx")
    monkeypatch.setattr(hj, "_anthropic_generate", lambda p: _payload("e" * 500))
    v = hj.haiku_news_judge("h", "s")
    assert v["label"] == "real" and v["confidence"] == 0.9
    assert v["is_fixed_price_buyout"] is False


# --- 2. log dedupe ----------------------------------------------------------

def _shadow_on(monkeypatch, tmp_path):
    monkeypatch.setattr(judge.config, "SCANNER_JUDGE_SHADOW", True)
    monkeypatch.setattr(judge.config, "SHADOW_JUDGE_LOG_PATH", str(tmp_path / "s.jsonl"))


def test_active_line_logs_once_across_many_resolves(monkeypatch, tmp_path, caplog):
    _shadow_on(monkeypatch, tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xxx")
    judge._reset_shadow_log_state()
    with caplog.at_level(logging.INFO, logger="momentum_scanner.judge"):
        for _ in range(25):
            judge.resolve_news_judge()
    active = [r for r in caplog.records if "shadow ACTIVE" in r.getMessage()]
    assert len(active) == 1, f"expected 1 ACTIVE line across 25 resolves, got {len(active)}"


def test_missing_key_line_also_logs_once(monkeypatch, tmp_path, caplog):
    _shadow_on(monkeypatch, tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    judge._reset_shadow_log_state()
    with caplog.at_level(logging.INFO, logger="momentum_scanner.judge"):
        for _ in range(25):
            assert judge.resolve_news_judge() is default_news_judge
    unset = [r for r in caplog.records if "unset" in r.getMessage()]
    assert len(unset) == 1


def test_resolution_is_still_per_call_so_a_key_gained_mid_life_upgrades(monkeypatch, tmp_path, caplog):
    """The load-bearing half: dedupe the LOG, never the RESOLUTION. If this fails,
    a launcher that gains the key while running stays gemma-only forever."""
    _shadow_on(monkeypatch, tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    judge._reset_shadow_log_state()
    with caplog.at_level(logging.INFO, logger="momentum_scanner.judge"):
        assert judge.resolve_news_judge() is default_news_judge
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xxx")
        upgraded = judge.resolve_news_judge()
        assert upgraded is not default_news_judge
    msgs = [r.getMessage() for r in caplog.records]
    assert any("unset" in m for m in msgs), "the degraded state should have been announced"
    assert any("ACTIVE" in m for m in msgs), "a STATE CHANGE must re-log, not stay silent"


def test_a_key_lost_mid_life_re_logs_the_degrade(monkeypatch, tmp_path, caplog):
    _shadow_on(monkeypatch, tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xxx")
    judge._reset_shadow_log_state()
    with caplog.at_level(logging.INFO, logger="momentum_scanner.judge"):
        judge.resolve_news_judge()
        judge.resolve_news_judge()
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert judge.resolve_news_judge() is default_news_judge
    assert len([r for r in caplog.records if "shadow ACTIVE" in r.getMessage()]) == 1
    assert len([r for r in caplog.records if "unset" in r.getMessage()]) == 1


def test_shadow_off_logs_nothing_at_all(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(judge.config, "SCANNER_JUDGE_SHADOW", False)
    judge._reset_shadow_log_state()
    with caplog.at_level(logging.INFO, logger="momentum_scanner.judge"):
        for _ in range(10):
            assert judge.resolve_news_judge() is default_news_judge
    assert [r for r in caplog.records if "shadow" in r.getMessage().lower()] == []

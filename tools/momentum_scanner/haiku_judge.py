# haiku_judge.py
"""Haiku news-judge backend -- the SHADOW half of the gemma-vs-haiku A/B.

Contract is deliberately IDENTICAL to catalyst_evaluator.default_news_judge:
(headline, summary) -> validated dict | None, and it NEVER raises. It reuses that
module's prompt block and _parse_judge_json on purpose: giving haiku its own prompt
or its own validator would make the A/B measure the scaffolding instead of the model.

This module is never on the deciding path. resolve_news_judge only ever hands it to
shadow_judge, which logs it beside gemma's verdict and returns GEMMA's. Nothing here
can change a trade.

The API key is read from ANTHROPIC_API_KEY at call time and is never logged, never
written to the shadow log, and never passed anywhere but the SDK constructor.
"""
import logging
import os

from momentum_scanner import config
from momentum_scanner.catalyst_evaluator import _NEWS_JUDGE_INSTRUCTIONS, _parse_judge_json
from momentum_scanner.shadow_judge import MAX_LOGGED_ERROR, redact

_log = logging.getLogger(__name__)

_client = None          # lazily constructed; module-level so tests can reset it

# Why a None verdict happened. The never-raises contract makes None ambiguous --
# outage, bad key, quota, or a model that genuinely emitted junk -- and for an A/B
# those are OPPOSITE conclusions: the first three say "fix the plumbing before
# trusting this window", the last one IS the measurement. Read-and-clear so a stale
# error can never be stamped onto a later, healthy row. Safe as a module global
# because the judge batch is serial (JUDGE_CONCURRENCY=1).
_last_error = None


def take_last_error():
    """Return the reason the last call returned None, and clear it. None if the last
    call succeeded (or nothing has run)."""
    global _last_error
    err, _last_error = _last_error, None
    return err


def has_api_key() -> bool:
    """True when a non-blank ANTHROPIC_API_KEY is present. Read at CALL time, not
    import time, so a launcher that gains the key mid-life needs no code change."""
    return bool((os.environ.get("ANTHROPIC_API_KEY") or "").strip())


def _get_client():
    """Lazy singleton. max_retries=0 is deliberate: this call sits inline in a serial
    judge batch, and an SDK retry storm would stall the DECIDING path behind a
    shadow that is not allowed to matter."""
    global _client
    if _client is None:
        import anthropic                                    # imported late: optional dep
        _client = anthropic.Anthropic(timeout=config.SHADOW_JUDGE_TIMEOUT, max_retries=0)
    return _client


def _anthropic_generate(prompt: str) -> str:
    """Single-shot completion. Mirrors catalyst_evaluator._ollama_generate's shape so
    both backends are monkeypatched the same way in tests."""
    if not has_api_key():
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    resp = _get_client().messages.create(
        model=config.HAIKU_MODEL,
        max_tokens=config.SHADOW_JUDGE_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


def haiku_news_judge(headline, summary):
    """Haiku judge (news path). Returns a validated dict or None on ANY failure.
    On None, take_last_error() explains which kind of failure it was."""
    global _last_error
    _last_error = None
    prompt = _NEWS_JUDGE_INSTRUCTIONS + \
        f"NEWS HEADLINE: {headline}\nNEWS SUMMARY:\n{(summary or '')[:8000]}\n"
    try:
        raw = _anthropic_generate(prompt)
    except Exception as exc:
        # Never surface the exception: the caller is the shadow wrapper, and a shadow
        # outage must read as "no second opinion", not as an error in the trading loop.
        _last_error = redact(f"{type(exc).__name__}: {exc}")[:MAX_LOGGED_ERROR]
        _log.debug("haiku shadow judge call failed: %s", _last_error)
        return None
    # Wider rationale than the deciding path keeps: this text is what the A/B is FOR.
    verdict = _parse_judge_json(raw, rationale_max=config.SHADOW_JUDGE_RATIONALE_MAX)
    if verdict is None:
        # Distinct from the branch above: the API answered, the MODEL was the problem.
        _last_error = "unparseable or invalid judge output"
    return verdict

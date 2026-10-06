# qwen3_judge.py
'Qwen3 shadow news judge: an Ollama-served second opinion that never raises (None plus take_last_error on failure).'
import logging

import requests

from momentum_scanner import config
from momentum_scanner.catalyst_evaluator import _NEWS_JUDGE_INSTRUCTIONS, _parse_judge_json
from momentum_scanner.shadow_judge import MAX_LOGGED_ERROR, redact

_log = logging.getLogger(__name__)

# Why a None verdict happened. Same rationale as haiku_judge: the never-raises contract
# makes None ambiguous (server outage, model-load failure, or a model that genuinely
# emitted junk), and for a shadow comparison those are opposite conclusions. Read-and-clear
# so a stale error can never be stamped onto a later, healthy row. Safe as a module global
# because the judge batch is serial (JUDGE_CONCURRENCY=1).
_last_error = None


def take_last_error():
    """Return the reason the last call returned None, and clear it. None if the last
    call succeeded (or nothing has run)."""
    global _last_error
    err, _last_error = _last_error, None
    return err


def _ollama_generate(prompt: str) -> str:
    """Mirrors catalyst_evaluator._ollama_generate's explicit num_ctx + truncate:false
    hardening (both hit the same Ollama server) so this call fails loud on an oversized
    prompt instead of riding whatever the server's default context happens to be."""
    payload = {
        "model": config.QWEN3_MODEL, "prompt": prompt,
        "stream": False, "think": False,
        "options": {"temperature": 0, "num_ctx": config.SCANNER_OLLAMA_NUM_CTX},
        "truncate": False,
    }
    resp = requests.post(config.OLLAMA_URL, json=payload, timeout=config.SHADOW_JUDGE_QWEN3_TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("response", "")


def qwen3_news_judge(headline, summary):
    """Qwen3:8b judge (news path). Returns a validated dict or None on ANY failure.
    On None, take_last_error() explains which kind of failure it was."""
    global _last_error
    _last_error = None
    prompt = _NEWS_JUDGE_INSTRUCTIONS + \
        f"NEWS HEADLINE: {headline}\nNEWS SUMMARY:\n{(summary or '')[:8000]}\n"
    try:
        raw = _ollama_generate(prompt)
    except Exception as exc:
        # Never surface the exception: the caller is the shadow wrapper, and a shadow
        # outage must read as "no second opinion", not as an error in the trading loop.
        _last_error = redact(f"{type(exc).__name__}: {exc}")[:MAX_LOGGED_ERROR]
        _log.debug("qwen3 shadow judge call failed: %s", _last_error)
        return None
    # Wider rationale than the deciding path keeps, same as haiku: this text is what the
    # comparison is FOR.
    verdict = _parse_judge_json(raw, rationale_max=config.SHADOW_JUDGE_RATIONALE_MAX)
    if verdict is None:
        # Distinct from the branch above: the server answered, the MODEL was the problem.
        _last_error = "unparseable or invalid judge output"
    return verdict

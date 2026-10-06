'Discovery rate gate: load an optional operator-supplied rate limiter and charge quote-chunk budget against it.'
import importlib
import os
import logging
import math

from momentum_scanner import config

_log = logging.getLogger(__name__)


def load_rate_limiter(state_path):
    """Resolve an operator-supplied limiter; absence retains the existing fail-closed gate."""
    module_name = os.environ.get("MARKET_MOMENTUM_RATE_LIMITER", "").strip()
    if not module_name:
        _log.warning("rate_gate: rate limiter not configured; limiter=None")
        return None
    try:
        mod = importlib.import_module(module_name)
        factory = getattr(mod, "SchwabRateLimiter", None)
        if not callable(factory):
            _log.warning("rate_gate: configured limiter lacks SchwabRateLimiter; limiter=None")
            return None
        limiter = factory(state_path)
        if not callable(getattr(limiter, "acquire", None)):
            _log.warning("rate_gate: configured limiter lacks acquire; limiter=None")
            return None
        return limiter
    except ImportError:
        _log.warning("rate_gate: configured limiter unavailable; limiter=None")
        return None

def discovery_chunks(symbol_count):
    return math.ceil(symbol_count / config.SCHWAB_QUOTE_CHUNK)


def acquire_discovery_budget(limiter, chunks, *, now=None, allow_ungated=False):
    if limiter is None:
        return bool(allow_ungated)
    return limiter.acquire(n=chunks, tier="discovery", now=now)

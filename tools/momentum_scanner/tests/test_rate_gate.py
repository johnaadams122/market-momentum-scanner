import importlib
from momentum_scanner.rate_gate import discovery_chunks, acquire_discovery_budget, load_rate_limiter
from momentum_scanner import config


class _FakeLimiter:
    def __init__(self, grant):
        self.grant = grant
        self.calls = []
    def acquire(self, n=1, tier="discovery", now=None):
        self.calls.append((n, tier, now))
        return self.grant


def test_chunks_is_ceil():
    assert discovery_chunks(0) == 0
    assert discovery_chunks(1) == 1
    assert discovery_chunks(config.SCHWAB_QUOTE_CHUNK) == 1
    assert discovery_chunks(config.SCHWAB_QUOTE_CHUNK + 1) == 2


def test_budget_limiter_none_fails_closed_by_default():
    assert acquire_discovery_budget(None, 6) is False


def test_budget_limiter_none_allows_only_with_opt_in():
    assert acquire_discovery_budget(None, 6, allow_ungated=True) is True


def test_budget_uses_discovery_tier_and_forwards_now():
    lim = _FakeLimiter(True)
    assert acquire_discovery_budget(lim, 6, now=123.0) is True
    assert lim.calls == [(6, "discovery", 123.0)]


def test_budget_false_on_deny():
    assert acquire_discovery_budget(_FakeLimiter(False), 6) is False


def test_load_rate_limiter_returns_none_when_module_missing(monkeypatch):
    monkeypatch.setenv("MARKET_MOMENTUM_RATE_LIMITER", "synthetic_rate_fixture")
    real = importlib.import_module

    def _boom(name, *a, **k):
        if name == "synthetic_rate_fixture":
            raise ImportError("synthetic limiter unavailable")
        return real(name, *a, **k)

    monkeypatch.setattr(importlib, "import_module", _boom)
    assert load_rate_limiter("ignored/path.json") is None

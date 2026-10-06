import json
import pytest
from momentum_scanner import fundamentals
from momentum_scanner.fundamentals import fetch_float, load_float_cache, save_float_cache


class _FakeTicker:
    def __init__(self, info=None, exc=None):
        self._info = info or {}
        self._exc = exc
    @property
    def info(self):
        if self._exc:
            raise self._exc
        return self._info


def _patch_yf(monkeypatch, info=None, exc=None):
    import momentum_scanner.fundamentals as f
    monkeypatch.setattr(f.yf, "Ticker", lambda t: _FakeTicker(info=info, exc=exc))


def test_fetch_float_returns_value(monkeypatch):
    _patch_yf(monkeypatch, info={"floatShares": 5_000_000})
    assert fetch_float("AAA") == 5_000_000


def test_fetch_float_none_for_missing_or_nonfinite(monkeypatch):
    _patch_yf(monkeypatch, info={})
    assert fetch_float("AAA") is None
    _patch_yf(monkeypatch, info={"floatShares": float("inf")})
    assert fetch_float("AAA") is None


def test_fetch_float_raises_propagate_for_transient(monkeypatch):
    _patch_yf(monkeypatch, exc=RuntimeError("yfinance http 500"))
    with pytest.raises(RuntimeError):
        fetch_float("AAA")                          # get_float catches this -> NOT cached


def test_load_missing_cache_is_empty(tmp_path):
    assert load_float_cache(tmp_path) == {}


def test_save_then_load_roundtrips(tmp_path):
    cache = {"AAA": {"float": 5_000_000, "as_of": "2026-06-30T00:00:00+00:00"}}
    save_float_cache(tmp_path, cache)
    assert load_float_cache(tmp_path) == cache


def test_corrupt_cache_loads_empty(tmp_path):
    (tmp_path / "fundamentals.json").write_text("{not json")
    assert load_float_cache(tmp_path) == {}

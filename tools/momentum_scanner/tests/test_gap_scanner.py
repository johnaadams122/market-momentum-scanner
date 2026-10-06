"""scan_candidates consumes a SchwabDataProvider (price/gap/relvol);
float stays on yfinance via _fetch_float. Leniency: only missing price/prev_close skip a
ticker; missing relvol -> None and candidate kept (gate decides).
"""
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from momentum_scanner.schwab_data import PremarketData
from momentum_scanner.errors import ScannerNetworkError

ET = ZoneInfo("America/New_York")
NOW = datetime(2026, 6, 29, 8, 0, tzinfo=ET)


class FakeProvider:
    def __init__(self, m):
        self.m = m

    def get_premarket(self, ticker, now_et):
        v = self.m.get(ticker)
        if isinstance(v, Exception):
            raise v
        return v


def _pd(price, prev=4.0, pv=3_000_000, base=500_000):
    return PremarketData("X", price, prev, datetime(2026, 6, 29, 7, 55, tzinfo=ET), pv, base)


def _scan(tickers, provider, monkeypatch, float_val=2_000_000):
    import momentum_scanner.gap_scanner as gs
    monkeypatch.setattr(gs, "_fetch_float", lambda t: float_val)
    return gs.scan_candidates(tickers, now_et=NOW, data_provider=provider)


def test_qualifying_candidate(monkeypatch):
    r = _scan(["TEST"], FakeProvider({"TEST": _pd(5.5)}), monkeypatch)
    assert len(r) == 1
    assert abs(r[0].gap_pct - 0.375) < 1e-9 and r[0].rel_volume == 6.0 and r[0].float_shares == 2_000_000
    assert r[0].premarket_volume == 3_000_000 and r[0].baseline_volume == 500_000


def test_missing_baseline_keeps_candidate_with_none_relvol(monkeypatch):
    r = _scan(["TEST"], FakeProvider({"TEST": _pd(5.5, pv=None, base=None)}), monkeypatch)
    assert len(r) == 1 and r[0].rel_volume is None


def test_zero_baseline_keeps_candidate_with_none_relvol(monkeypatch):
    r = _scan(["TEST"], FakeProvider({"TEST": _pd(5.5, pv=1000, base=0)}), monkeypatch)
    assert len(r) == 1 and r[0].rel_volume is None


def test_provider_none_skips_ticker(monkeypatch):
    assert _scan(["TEST"], FakeProvider({"TEST": None}), monkeypatch) == []


def test_price_below_floor_excluded(monkeypatch):
    assert _scan(["T"], FakeProvider({"T": _pd(0.80, prev=0.50)}), monkeypatch) == []


def test_price_at_one_dollar_included(monkeypatch):
    assert len(_scan(["T"], FakeProvider({"T": _pd(1.20, prev=1.00)}), monkeypatch)) == 1


def test_price_above_max_excluded(monkeypatch):
    assert _scan(["T"], FakeProvider({"T": _pd(25.0)}), monkeypatch) == []


def test_gap_below_min_excluded(monkeypatch):
    assert _scan(["T"], FakeProvider({"T": _pd(4.35, prev=4.0)}), monkeypatch) == []


def test_gap_at_min_included(monkeypatch):
    assert len(_scan(["T"], FakeProvider({"T": _pd(4.40, prev=4.0)}), monkeypatch)) == 1


def test_float_at_max_excluded(monkeypatch):
    assert _scan(["T"], FakeProvider({"T": _pd(5.5)}), monkeypatch, float_val=10_000_000) == []


def test_high_float_excluded(monkeypatch):
    assert _scan(["T"], FakeProvider({"T": _pd(5.5)}), monkeypatch, float_val=20_000_000) == []


def test_none_float_kept(monkeypatch):
    r = _scan(["T"], FakeProvider({"T": _pd(5.5)}), monkeypatch, float_val=None)
    assert len(r) == 1 and r[0].float_shares is None


def test_low_relvol_excluded(monkeypatch):
    assert _scan(["T"], FakeProvider({"T": _pd(5.5, pv=1_500_000, base=500_000)}), monkeypatch) == []


def test_sorts_by_gap_descending(monkeypatch):
    prov = FakeProvider({"AAA": _pd(4.40), "BBB": _pd(6.00), "CCC": _pd(4.80)})
    r = _scan(["AAA", "BBB", "CCC"], prov, monkeypatch)
    assert [x.ticker for x in r] == ["BBB", "CCC", "AAA"]


def test_scanner_network_error_propagates(monkeypatch):
    import momentum_scanner.gap_scanner as gs
    monkeypatch.setattr(gs, "_fetch_float", lambda t: 2_000_000)
    prov = FakeProvider({"T": ScannerNetworkError("auth down")})
    with pytest.raises(ScannerNetworkError):
        gs.scan_candidates(["T"], now_et=NOW, data_provider=prov)


def test_local_exception_skips_ticker(monkeypatch):
    import momentum_scanner.gap_scanner as gs
    monkeypatch.setattr(gs, "_fetch_float", lambda t: 2_000_000)
    prov = FakeProvider({"BAD": RuntimeError("transient"), "GOOD": _pd(5.5)})
    r = gs.scan_candidates(["BAD", "GOOD"], now_et=NOW, data_provider=prov)
    assert [x.ticker for x in r] == ["GOOD"]


def test_now_et_defaults_when_omitted(monkeypatch):
    import momentum_scanner.gap_scanner as gs
    monkeypatch.setattr(gs, "_fetch_float", lambda t: 2_000_000)
    captured = {}

    class P:
        def get_premarket(self, ticker, now_et):
            captured["now_et"] = now_et
            return _pd(5.5)
    gs.scan_candidates(["T"], data_provider=P())   # no now_et -> default
    assert captured["now_et"] is not None and captured["now_et"].tzinfo is not None

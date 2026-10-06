"""SchwabDataProvider premarket price/volume/relvol-baseline.

All offline-mocked (FakeSession), matching the other scanner tests. The fake routes by
request params (daily vs minute, and which session's window) and never uses a closure hack.
"""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

ET = ZoneInfo("America/New_York")


def _ms(dt):
    return int(dt.timestamp() * 1000)


def _et(y, mo, d, h, mi):
    return datetime(y, mo, d, h, mi, tzinfo=ET)


def _min_candle(dt_et, close, vol):
    return {"datetime": _ms(dt_et), "open": close, "high": close, "low": close,
            "close": close, "volume": vol}


def _day_candle(date_obj, close, vol=1):
    return {"datetime": _ms(datetime(date_obj.year, date_obj.month, date_obj.day, 16, 0, tzinfo=ET)),
            "close": close, "volume": vol}


class FakeResp:
    def __init__(self, candles, status=200):
        self._candles = candles
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400 and self.status_code != 401:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return {"candles": self._candles}


class FakeSession:
    def __init__(self, handler):
        self.handler = handler            # params -> (candles, status)
        self.calls = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append(params)
        self.timeouts = getattr(self, "timeouts", [])
        self.timeouts.append(timeout)
        candles, status = self.handler(params)
        return FakeResp(candles, status)


def test_premarket_price_volume_and_pinned_request_params():
    from momentum_scanner.schwab_data import SchwabDataProvider
    now = _et(2026, 6, 29, 8, 0)   # Monday 08:00 ET

    def handler(p):
        if p["frequencyType"] == "daily":
            return ([_day_candle(datetime(2026, 6, 26).date(), 4.00),
                     _day_candle(datetime(2026, 6, 25).date(), 3.90)], 200)
        # single combined minute pull -> today's two candles (no baseline-session candles here)
        return ([_min_candle(_et(2026, 6, 29, 4, 0), 4.10, 1000),
                 _min_candle(_et(2026, 6, 29, 7, 55), 4.50, 2000)], 200)
    sess = FakeSession(handler)
    pd = SchwabDataProvider(token_fn=lambda: "T", session=sess).get_premarket("ABCD", now)

    assert pd is not None
    assert pd.premarket_price == 4.50 and pd.premarket_volume == 3000 and pd.prev_close == 4.00
    assert pd.baseline_volume is None            # no baseline candles -> None, candidate still usable
    # A single minute pull, pinned to extended hours and ending at now
    minute = [c for c in sess.calls if c["frequencyType"] == "minute"]
    assert len(minute) == 1                       # one pull, not one-per-session
    assert minute[0]["needExtendedHoursData"] is True
    assert minute[0]["endDate"] == _ms(now)


def test_baseline_is_median_over_prior_session_dates_with_holiday_gap():
    # Sessions derived from the DAILY candle dates (not calendar subtraction): a holiday gap
    # (06-24 missing) is handled because we only iterate the daily candles Schwab returned.
    from momentum_scanner.schwab_data import SchwabDataProvider
    now = _et(2026, 6, 29, 8, 0)
    sessions = [datetime(2026, 6, 23).date(), datetime(2026, 6, 25).date(), datetime(2026, 6, 26).date()]
    vols = {sessions[0]: 1000, sessions[1]: 3000, sessions[2]: 2000}

    def handler(p):
        if p["frequencyType"] == "daily":
            return ([_day_candle(d, 4.0) for d in sessions], 200)
        # one combined minute series: today's fresh candle + each prior session's premarket candle
        candles = [_min_candle(_et(2026, 6, 29, 7, 55), 4.5, 5000)]
        for d in sessions:
            candles.append(_min_candle(datetime(d.year, d.month, d.day, 7, 0, tzinfo=ET), 4.0, vols[d]))
        return (candles, 200)
    sess = FakeSession(handler)
    pd = SchwabDataProvider(token_fn=lambda: "T", session=sess).get_premarket("ABCD", now)
    assert pd.premarket_volume == 5000
    assert pd.baseline_volume == 2000.0          # median(1000, 3000, 2000)
    assert len([c for c in sess.calls if c["frequencyType"] == "minute"]) == 1   # single pull


def test_single_minute_pull_regardless_of_baseline_count():
    """Perf: all baseline sessions + today come from ONE multi-day minute pull, bucketed
    locally -- not one request per session (the old pattern burst ~N+1 requests per ticker)."""
    from momentum_scanner.schwab_data import SchwabDataProvider
    now = _et(2026, 6, 29, 8, 0)
    sessions = [datetime(2026, 6, d).date() for d in (22, 23, 24, 25, 26)]
    minute_candles = [_min_candle(datetime(d.year, d.month, d.day, 7, 0, tzinfo=ET), 4.0, 1000)
                      for d in sessions]
    minute_candles.append(_min_candle(_et(2026, 6, 29, 7, 55), 4.5, 5000))   # today, fresh

    def handler(p):
        if p["frequencyType"] == "daily":
            return ([_day_candle(d, 4.0) for d in sessions], 200)
        return (minute_candles, 200)            # ONE response holds every day's candles
    sess = FakeSession(handler)
    pd = SchwabDataProvider(token_fn=lambda: "T", session=sess,
                            baseline_sessions=5).get_premarket("ABCD", now)
    assert pd is not None
    minute_calls = [c for c in sess.calls if c["frequencyType"] == "minute"]
    assert len(minute_calls) == 1, f"expected ONE minute pull, got {len(minute_calls)}"
    assert pd.premarket_volume == 5000
    assert pd.baseline_volume == 1000.0         # median of five 1000-vol prior sessions
    # the single pull spans from the earliest baseline session's 04:00 ET window-start to now
    assert minute_calls[0]["startDate"] == _ms(_et(2026, 6, 22, 4, 0))
    assert minute_calls[0]["endDate"] == _ms(now)


def test_freshness_gate_returns_none_when_price_stale():
    from momentum_scanner.schwab_data import SchwabDataProvider
    now = _et(2026, 6, 29, 8, 0)

    def handler(p):
        if p["frequencyType"] == "daily":
            return ([_day_candle(datetime(2026, 6, 26).date(), 4.0)], 200)
        return ([_min_candle(_et(2026, 6, 29, 7, 0), 4.5, 1000)], 200)   # 60 min old > 15
    pd = SchwabDataProvider(token_fn=lambda: "T", session=FakeSession(handler),
                            max_age_minutes=15).get_premarket("ABCD", now)
    assert pd is None


def test_empty_premarket_window_returns_none():
    from momentum_scanner.schwab_data import SchwabDataProvider
    now = _et(2026, 6, 29, 8, 0)

    def handler(p):
        if p["frequencyType"] == "daily":
            return ([_day_candle(datetime(2026, 6, 26).date(), 4.0)], 200)
        return ([], 200)
    assert SchwabDataProvider(token_fn=lambda: "T",
                              session=FakeSession(handler)).get_premarket("ABCD", now) is None


def test_401_escalates_to_scanner_error():
    # 401 = token rejection = SYSTEMIC (affects every ticker) -> escalate, not a per-ticker skip.
    from momentum_scanner.schwab_data import SchwabDataProvider
    from momentum_scanner.errors import ScannerNetworkError
    now = _et(2026, 6, 29, 8, 0)

    def handler(p):
        if p["frequencyType"] == "daily":
            return ([_day_candle(datetime(2026, 6, 26).date(), 4.0)], 200)
        return ([], 401)
    with pytest.raises(ScannerNetworkError):
        SchwabDataProvider(token_fn=lambda: "T", session=FakeSession(handler)).get_premarket("ABCD", now)


def test_connection_error_escalates_to_scanner_error():
    import requests
    from momentum_scanner.schwab_data import SchwabDataProvider
    from momentum_scanner.errors import ScannerNetworkError

    class BoomSession:
        def get(self, *a, **k):
            raise requests.ConnectionError("schwab down")
    with pytest.raises(ScannerNetworkError):
        SchwabDataProvider(token_fn=lambda: "T", session=BoomSession()).get_premarket("ABCD", _et(2026, 6, 29, 8, 0))


def test_404_skips_ticker_not_escalates():
    # 404 = symbol not found = ticker-local -> skip this ticker (None), do NOT abort the scan.
    from momentum_scanner.schwab_data import SchwabDataProvider
    now = _et(2026, 6, 29, 8, 0)

    def handler(p):
        if p["frequencyType"] == "daily":
            return ([_day_candle(datetime(2026, 6, 26).date(), 4.0)], 200)
        return ([], 404)
    assert SchwabDataProvider(token_fn=lambda: "T",
                              session=FakeSession(handler)).get_premarket("ABCD", now) is None


def test_no_prior_session_returns_none():
    from momentum_scanner.schwab_data import SchwabDataProvider
    now = _et(2026, 6, 29, 8, 0)

    def handler(p):
        if p["frequencyType"] == "daily":
            return ([], 200)                      # no daily candles at all
        return ([_min_candle(_et(2026, 6, 29, 7, 0), 4.5, 1000)], 200)
    assert SchwabDataProvider(token_fn=lambda: "T",
                              session=FakeSession(handler)).get_premarket("ABCD", now) is None


def test_guarded_import_raises_scanner_error(monkeypatch):
    import momentum_scanner.schwab_data as sd
    from momentum_scanner.errors import ScannerNetworkError

    def boom(name):
        raise ImportError("token provider module missing")
    monkeypatch.setenv("MARKET_MOMENTUM_TOKEN_PROVIDER", "synthetic_token_provider")
    monkeypatch.setattr(sd.importlib, "import_module", boom)
    p = sd.SchwabDataProvider(session=FakeSession(lambda params: ([], 200)))   # token_fn=None -> default
    with pytest.raises(ScannerNetworkError):
        p.get_premarket("ABCD", _et(2026, 6, 29, 8, 0))

from datetime import datetime
from zoneinfo import ZoneInfo
import pytest
from momentum_scanner import daemon_clock
from momentum_scanner.daemon_clock import session_mode

ET = ZoneInfo("America/New_York")
_never = lambda d: False


def m(y, mo, d, h, mi):
    return datetime(y, mo, d, h, mi, tzinfo=ET)


def test_premarket_window():
    assert session_mode(m(2026, 6, 30, 5, 0), holiday_fn=_never) == "premarket"


def test_premarket_open_boundary_0400():
    assert session_mode(m(2026, 6, 30, 4, 0), holiday_fn=_never) == "premarket"


def test_intraday_window():
    assert session_mode(m(2026, 6, 30, 10, 0), holiday_fn=_never) == "intraday"


def test_closed_after_hours_and_weekend():
    assert session_mode(m(2026, 6, 30, 17, 0), holiday_fn=_never) == "closed"
    assert session_mode(m(2026, 7, 4, 10, 0), holiday_fn=_never) == "closed"


def test_holiday_is_closed():
    assert session_mode(m(2026, 6, 30, 10, 0), holiday_fn=lambda d: True) == "closed"


def test_naive_clock_raises():
    with pytest.raises(ValueError):
        session_mode(datetime(2026, 6, 30, 10, 0), holiday_fn=_never)


def test_open_boundary_is_intraday_close_boundary_is_closed():
    assert session_mode(m(2026, 6, 30, 9, 30), holiday_fn=_never) == "intraday"
    assert session_mode(m(2026, 6, 30, 16, 0), holiday_fn=_never) == "closed"


def test_default_holiday_fn_fails_closed_when_import_fails(monkeypatch, caplog):
    import importlib
    import logging
    monkeypatch.setenv("MARKET_MOMENTUM_CALENDAR", "synthetic_calendar_fixture")
    real = importlib.import_module

    def _boom(name, *a, **k):
        if name == "synthetic_calendar_fixture":
            raise ImportError("synthetic calendar unavailable")
        return real(name, *a, **k)

    monkeypatch.setattr(importlib, "import_module", _boom)
    daemon_clock._HOLIDAY_IMPORT_FAILED = False
    with caplog.at_level(logging.ERROR):
        # Fail-CLOSED: an unknowable calendar means NO session, never a fake-ok run
        assert session_mode(m(2026, 6, 30, 10, 0)) == "closed"
        assert session_mode(m(2026, 6, 30, 11, 0)) == "closed"   # every later call too
    assert any("FAIL-CLOSED" in r.getMessage() for r in caplog.records)


def test_default_holiday_fn_recovers_when_import_succeeds(monkeypatch):
    # A configured calendar that imports cleanly clears the fail-closed latch.
    import importlib
    import types
    fake = types.SimpleNamespace(is_market_holiday=lambda d: False)
    monkeypatch.setenv("MARKET_MOMENTUM_CALENDAR", "synthetic_calendar_fixture")
    real = importlib.import_module

    def _ok(name, *a, **k):
        if name == "synthetic_calendar_fixture":
            return fake
        return real(name, *a, **k)

    monkeypatch.setattr(importlib, "import_module", _ok)
    daemon_clock._HOLIDAY_IMPORT_FAILED = False
    assert session_mode(m(2026, 6, 30, 10, 0)) == "intraday"

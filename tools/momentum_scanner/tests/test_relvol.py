# tests/test_relvol.py
from datetime import datetime
from zoneinfo import ZoneInfo
from momentum_scanner import relvol, config
ET = ZoneInfo("America/New_York")

def test_midday_intraday_relvol():
    now = datetime(2026, 6, 29, 13, 0, tzinfo=ET)        # 3.5h/6.5h ~ 0.538
    r = relvol.relvol_proxy(2_000_000, 5_000_000, now, premarket=False)
    assert r is not None and 0.7 < r < 0.8

def test_opening_guard_returns_none():
    now = datetime(2026, 6, 29, 9, 30, 30, tzinfo=ET)    # inside the 2-min guard
    assert relvol.relvol_proxy(500_000, 5_000_000, now, premarket=False) is None

def test_just_after_guard_is_finite_and_floored():
    now = datetime(2026, 6, 29, 9, 45, tzinfo=ET)        # 15min in, past guard; frac 0.0385 > floor
    r = relvol.relvol_proxy(500_000, 5_000_000, now, premarket=False)
    assert r is not None and r != float("inf") and 2.0 < r < 3.0

def test_premarket_window():
    now = datetime(2026, 6, 29, 8, 0, tzinfo=ET)
    r = relvol.relvol_proxy(300_000, 5_000_000, now, premarket=True)
    assert r is not None and r > 0

def test_none_inputs_return_none():
    now = datetime(2026, 6, 29, 13, 0, tzinfo=ET)
    assert relvol.relvol_proxy(None, 5_000_000, now, premarket=False) is None
    assert relvol.relvol_proxy(1_000_000, None, now, premarket=False) is None
    assert relvol.relvol_proxy(1_000_000, 0, now, premarket=False) is None


def test_naive_datetime_returns_none():
    """A naive datetime (no tzinfo) silently miscomputes -- must fail-closed."""
    now = datetime(2026, 6, 29, 13, 0)   # no tzinfo
    assert relvol.relvol_proxy(1_000_000, 5_000_000, now, premarket=False) is None


def test_pre_session_premarket_returns_none():
    """Before premarket opens (04:00 ET) should return None -- no phantom signal."""
    now = datetime(2026, 6, 29, 3, 0, tzinfo=ET)   # 03:00 ET is before 04:00 premarket start
    assert relvol.relvol_proxy(500_000, 5_000_000, now, premarket=True) is None


def test_rth_floor_active_at_09_33():
    """At 09:33 RTH (3 min in), raw frac ~0.0077 < FRACTION_FLOOR (0.02).
    Result must be finite and equal tv / (adv * FRACTION_FLOOR)."""
    now = datetime(2026, 6, 29, 9, 33, tzinfo=ET)   # past 2-min guard; frac well below floor
    tv, adv = 100_000, 5_000_000
    r = relvol.relvol_proxy(tv, adv, now, premarket=False)
    expected = tv / (adv * config.FRACTION_FLOOR)
    assert r is not None, "expected finite result when floor is active"
    assert abs(r - expected) < 1e-9, f"floor not engaged: got {r}, expected {expected}"

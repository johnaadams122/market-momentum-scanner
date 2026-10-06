'Session clock for the movers daemon: classifies an ET time as premarket, intraday or closed (fail-closed holiday check).'
import importlib
import os
import logging
from datetime import time as _time

from momentum_scanner import config

_log = logging.getLogger(__name__)

_HOLIDAY_IMPORT_FAILED = False     # once-only flag so the degrade WARNING is not logged every cycle


def _hhmm(s):
    h, m = s.split(":")
    return _time(int(h), int(m))


_PREMARKET_START = _hhmm(config.PREMARKET_SESSION_START_ET)
_RTH_START = _hhmm(config.RTH_START_ET)
_RTH_END = _hhmm(config.RTH_END_ET)


def _default_is_holiday(d):
    global _HOLIDAY_IMPORT_FAILED
    module_name = os.environ.get("MARKET_MOMENTUM_CALENDAR", "").strip()
    try:
        if not module_name:
            raise ImportError("MARKET_MOMENTUM_CALENDAR must be explicitly configured")
        mod = importlib.import_module(module_name)
        if not callable(getattr(mod, "is_market_holiday", None)):
            raise ImportError("configured calendar must expose is_market_holiday")
    except ImportError as exc:
        if not _HOLIDAY_IMPORT_FAILED:
            _log.error("daemon_clock: configured calendar unavailable (%s); FAIL-CLOSED -- "
                       "treating every day as a holiday until configuration is fixed", exc)
            _HOLIDAY_IMPORT_FAILED = True
        else:
            _log.error("daemon_clock: configured calendar still unavailable; FAIL-CLOSED (holiday)")
        return True
    _HOLIDAY_IMPORT_FAILED = False
    return bool(mod.is_market_holiday(d))

def session_mode(now_et, *, holiday_fn=None):
    if now_et.tzinfo is None or now_et.tzinfo.utcoffset(now_et) is None:
        raise ValueError("session_mode requires a tz-aware ET datetime")
    holiday_fn = holiday_fn if holiday_fn is not None else _default_is_holiday
    if now_et.weekday() >= 5:
        return "closed"
    if holiday_fn(now_et.date()):
        return "closed"
    t = now_et.timetz().replace(tzinfo=None)
    if _PREMARKET_START <= t < _RTH_START:
        return "premarket"
    if _RTH_START <= t < _RTH_END:
        return "intraday"
    return "closed"

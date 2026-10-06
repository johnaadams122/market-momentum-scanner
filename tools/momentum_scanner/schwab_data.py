'Schwab market-data provider: premarket price, previous close and relative-volume baseline from price history.'
import importlib
import os
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import requests

from momentum_scanner.config import (
    PREMARKET_WINDOW_START_ET,
    RELVOL_BASELINE_SESSIONS,
    SOURCE_MAX_AGE_MINUTES,
    SCANNER_HTTP_TIMEOUT,
)
from momentum_scanner.errors import ScannerNetworkError

ET = ZoneInfo("America/New_York")
BASE_URL = "https://api.schwabapi.com/marketdata/v1"


@dataclass(frozen=True)
class PremarketData:
    ticker: str
    premarket_price: float          # REQUIRED
    prev_close: float               # REQUIRED
    as_of: datetime                 # REQUIRED (UTC, freshest in-window candle)
    premarket_volume: int | None    # OPTIONAL
    baseline_volume: float | None   # OPTIONAL


def _default_token_fn() -> str:
    """Resolve an explicitly configured token-provider module; no bundled credentials."""
    module_name = os.environ.get("MARKET_MOMENTUM_TOKEN_PROVIDER", "").strip()
    if not module_name:
        raise ScannerNetworkError("MARKET_MOMENTUM_TOKEN_PROVIDER must be explicitly configured")
    try:
        mod = importlib.import_module(module_name)
    except ImportError as exc:
        raise ScannerNetworkError("configured token provider unavailable") from exc
    if not callable(getattr(mod, "get_valid_token", None)):
        raise ScannerNetworkError("configured token provider must expose get_valid_token")
    return mod.get_valid_token()

def _hhmm(s: str):
    h, m = s.split(":")
    return int(h), int(m)


def _candle_et(candle):
    return datetime.fromtimestamp(candle["datetime"] / 1000, tz=timezone.utc).astimezone(ET)


class SchwabDataProvider:
    def __init__(self, *, token_fn=None, session=None,
                 window_start_et=PREMARKET_WINDOW_START_ET,
                 baseline_sessions=RELVOL_BASELINE_SESSIONS,
                 max_age_minutes=SOURCE_MAX_AGE_MINUTES):
        self._token_fn = token_fn or _default_token_fn
        self._session = session or requests
        self._wh, self._wm = _hhmm(window_start_et)
        self._baseline_sessions = baseline_sessions
        self._max_age = timedelta(minutes=max_age_minutes)

    def _headers(self):
        return {"Authorization": f"Bearer {self._token_fn()}"}

    def _get(self, params):
        """Source-level fetch. A 404 (symbol not found) is the ONLY ticker-local failure -> None
        (skip that ticker). Everything else -- 401/token rejection, other HTTP status, connection/
        timeout, or a failed token acquisition -- is a SYSTEMIC outage and raises ScannerNetworkError
        so the runner writes a status:"error" watchlist instead of a false "ok, 0 candidates"."""
        try:
            resp = self._session.get(f"{BASE_URL}/pricehistory", headers=self._headers(), params=params,
                                     timeout=SCANNER_HTTP_TIMEOUT)
        except ScannerNetworkError:
            raise                                       # guarded-import / token-fn failure
        except Exception as exc:                        # connection/timeout/etc -> systemic
            # Exception TYPE only, and no chained cause: the original text can echo the bearer token
            # (a malformed header value is quoted verbatim by `requests`).
            raise ScannerNetworkError(f"Schwab request failed: {type(exc).__name__}") from None
        status = getattr(resp, "status_code", 200)
        if status == 404:
            return None                                 # symbol not found -> ticker-local skip
        if status >= 400:
            raise ScannerNetworkError(f"Schwab HTTP {status}")   # 401/403/5xx -> systemic
        return resp.json().get("candles", [])

    def _minute_range(self, ticker, start_et, end_et):
        """ONE minute-frequency pull over [start_et, end_et] (may span several days). The defensive
        filter pins the range so a bar outside it that Schwab may include can never leak in."""
        params = {
            "symbol": ticker, "periodType": "day", "frequencyType": "minute", "frequency": 1,
            "needExtendedHoursData": True,
            "startDate": int(start_et.astimezone(timezone.utc).timestamp() * 1000),
            "endDate": int(end_et.astimezone(timezone.utc).timestamp() * 1000),
        }
        candles = self._get(params)
        if candles is None:
            return None
        return [c for c in candles if start_et <= _candle_et(c) <= end_et]

    def _daily(self, ticker):
        params = {
            "symbol": ticker, "periodType": "month", "period": 2,
            "frequencyType": "daily", "frequency": 1, "needExtendedHoursData": False,
        }
        return self._get(params)

    def get_premarket(self, ticker, now_et) -> "PremarketData | None":
        now_et = now_et.astimezone(ET)
        today = now_et.date()

        # prev_close + the list of completed prior sessions, derived from daily candle dates
        # (NOT calendar subtraction -> weekend/holiday/half-day/DST safe).
        daily = self._daily(ticker)
        if not daily:
            return None
        prior = [c for c in daily if _candle_et(c).date() < today]
        if not prior:
            return None
        prior.sort(key=lambda c: c["datetime"])
        prev_close = float(prior[-1]["close"])

        start_today = now_et.replace(hour=self._wh, minute=self._wm, second=0, microsecond=0)
        baseline_dates = [_candle_et(c).date() for c in prior[-self._baseline_sessions:]]

        # ONE multi-day minute pull from the earliest needed session's window-start to now, then
        # bucket locally per session, instead of one call for today plus one per baseline session.
        earliest = baseline_dates[0] if baseline_dates else today
        range_start = datetime(earliest.year, earliest.month, earliest.day,
                               self._wh, self._wm, tzinfo=ET)
        candles = self._minute_range(ticker, range_start, now_et)
        if candles is None:
            return None                                    # 404 -> ticker-local skip

        # today's premarket window: window_start ET -> now (REQUIRED)
        win = [c for c in candles if start_today <= _candle_et(c) <= now_et]
        if not win:
            return None
        win.sort(key=lambda c: c["datetime"])
        as_of = datetime.fromtimestamp(win[-1]["datetime"] / 1000, tz=timezone.utc)
        if now_et.astimezone(timezone.utc) - as_of > self._max_age:
            return None                                    # stale price -> fail-closed (required)
        premarket_price = round(float(win[-1]["close"]), 4)
        premarket_volume = int(sum(c.get("volume", 0) for c in win))

        # baseline: same-wall-clock window [window_start, now's wall-clock] per prior session date,
        # bucketed from the SAME pull (OPTIONAL -> None ok).
        baseline = []
        for d in baseline_dates:
            s = datetime(d.year, d.month, d.day, self._wh, self._wm, tzinfo=ET)
            e = datetime(d.year, d.month, d.day, now_et.hour, now_et.minute, tzinfo=ET)
            cs = [c for c in candles if s <= _candle_et(c) <= e]
            if cs:
                baseline.append(sum(x.get("volume", 0) for x in cs))
        baseline_volume = float(statistics.median(baseline)) if baseline else None

        return PremarketData(ticker, premarket_price, round(prev_close, 4), as_of,
                             premarket_volume, baseline_volume)

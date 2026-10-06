import logging
import math
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import yfinance as yf

from momentum_scanner.schwab_data import SchwabDataProvider
from momentum_scanner.errors import ScannerNetworkError

_log = logging.getLogger("momentum_scanner.gap_scanner")
ET = ZoneInfo("America/New_York")

PRICE_MIN = 1.0  # default numeric filters; each is overridable per scan_candidates call
PRICE_MAX = 20.0
GAP_MIN = 0.10
FLOAT_MAX = 10_000_000
REL_VOL_MIN = 5.0


@dataclass
class ScanResult:
    ticker: str
    company_name: str
    premarket_price: float
    prev_close: float
    gap_pct: float
    float_shares: int | None
    premarket_volume: int | None     # Schwab cumulative ext-hours volume, 04:00 ET -> scan time
    baseline_volume: float | None    # median same-window volume over the last N sessions
    rel_volume: float | None


def _fetch_float(ticker: str):
    """yfinance floatShares (slow-changing, so it stays on yfinance). None on error/non-finite."""
    try:
        fs = yf.Ticker(ticker).info.get("floatShares")
    except Exception as exc:
        _log.debug("yfinance float failed for %s: %s", ticker, exc)
        return None
    if fs is None or not math.isfinite(fs):
        return None
    return fs


def scan_candidates(
    tickers,
    *,
    now_et=None,
    data_provider=None,
    price_min: float = PRICE_MIN,
    price_max: float = PRICE_MAX,
    gap_min: float = GAP_MIN,
    float_max: int = FLOAT_MAX,
    rel_vol_min: float = REL_VOL_MIN,
):
    """Numeric premarket filter. Price/gap/premarket-volume/relvol come from a SchwabDataProvider
    (injected for tests; defaults to a live one); float stays on yfinance. LENIENT: a ticker is
    skipped ONLY when its required Schwab price/prev_close is missing/stale; missing relvol inputs
    null rel_volume but keep the candidate (the catalyst gate makes the fail-closed call)."""
    if now_et is None:
        now_et = datetime.now(ET)
    provider = data_provider or SchwabDataProvider()
    results = []
    for ticker in tickers:
        try:
            pd = provider.get_premarket(ticker, now_et)
        except ScannerNetworkError:
            raise                                   # global auth/import failure -> runner status:error
        except Exception as exc:                    # local/transient -> skip just this ticker
            _log.debug("provider failed for %s: %s", ticker, exc)
            continue
        if pd is None:
            continue                                # missing/stale required price or prev_close
        if not (price_min <= pd.premarket_price <= price_max):
            continue
        if pd.prev_close == 0 or not math.isfinite(pd.prev_close):
            continue
        gap_pct = (pd.premarket_price - pd.prev_close) / pd.prev_close
        if gap_pct < gap_min:
            continue

        float_shares = _fetch_float(ticker)
        if float_shares is not None and float_shares >= float_max:
            continue

        rel_volume = None
        if (pd.premarket_volume is not None and pd.baseline_volume
                and math.isfinite(pd.baseline_volume) and pd.baseline_volume > 0):
            rv = round(pd.premarket_volume / pd.baseline_volume, 2)
            rel_volume = rv if math.isfinite(rv) else None
        if rel_volume is not None and rel_volume < rel_vol_min:
            continue

        results.append(ScanResult(
            ticker=ticker,
            company_name=ticker,                    # Schwab pricehistory has no name; ticker is fine
            premarket_price=round(pd.premarket_price, 4),
            prev_close=round(pd.prev_close, 4),
            gap_pct=round(gap_pct, 4),
            float_shares=float_shares,
            premarket_volume=pd.premarket_volume,
            baseline_volume=pd.baseline_volume,
            rel_volume=rel_volume,
        ))

    return sorted(results, key=lambda r: r.gap_pct, reverse=True)

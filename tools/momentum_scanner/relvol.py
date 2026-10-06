"""Boundary-safe relative-volume proxy: today's cumulative volume / (avg_daily_volume * elapsed
fraction of the session), fraction FLOORED so the open never divides by ~0. Separate premarket
(04:00-09:30) and RTH (09:30-16:00) windows. Returns None (never inf/NaN) when inputs are unusable
AND during the first OPENING_GUARD_MIN of RTH -- the raw cumulative-volume floor + gap govern those
minutes."""
import math
from datetime import timedelta
from zoneinfo import ZoneInfo

from momentum_scanner import config

ET = ZoneInfo("America/New_York")


def _hhmm(s):
    h, m = s.split(":")
    return int(h), int(m)


def _at(now_et, s):
    h, m = _hhmm(s)
    return now_et.replace(hour=h, minute=m, second=0, microsecond=0)


def relvol_proxy(total_volume, avg_daily_volume, now_et, *, premarket):
    # Fail-closed: a naive datetime silently miscomputes via host-local tz.
    if now_et is None or now_et.tzinfo is None:
        return None
    if total_volume is None or avg_daily_volume is None:
        return None
    try:
        tv, adv = float(total_volume), float(avg_daily_volume)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(tv) and math.isfinite(adv)) or adv <= 0:
        return None
    now_et = now_et.astimezone(ET)
    if premarket:
        start, end = _at(now_et, config.PREMARKET_SESSION_START_ET), _at(now_et, config.RTH_START_ET)
        if now_et < start:
            return None                       # pre-session: no premarket signal yet
    else:
        start, end = _at(now_et, config.RTH_START_ET), _at(now_et, config.RTH_END_ET)
        if now_et < start:
            return None                       # pre-session for RTH
        if now_et < start + timedelta(minutes=config.OPENING_GUARD_MIN):
            return None                       # opening guard: raw-volume floor governs
    total = (end - start).total_seconds()
    if total <= 0:
        return None
    frac = max(min((now_et - start).total_seconds() / total, 1.0), config.FRACTION_FLOOR)
    # denom > 0 is guaranteed: adv > 0 (checked above), frac >= FRACTION_FLOOR > 0
    r = tv / (adv * frac)
    return r if math.isfinite(r) else None

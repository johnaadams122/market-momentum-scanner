"""Two-tier movers scan. Tier-1 is wide and cheap: it filters the whole-universe batch
quote dict by the price / gap / volume-floor / relative-volume pillars (and quote freshness),
keeping a small shortlist. A None relative-volume (avg-vol absent or inside the opening guard) does
NOT reject -- the candidate rides the raw cumulative-volume floor + gap shortlist (raw
fallback) so it still reaches Tier-2. Tier-2 + orchestration live lower in this module."""
import logging
import math
from dataclasses import dataclass
from datetime import timezone

from momentum_scanner import config
from momentum_scanner import movers_watchlist
from momentum_scanner.adapters.schwab_quotes import batch_quotes
from momentum_scanner.errors import ScannerNetworkError
from momentum_scanner.fundamentals import get_float
from momentum_scanner.ranking import rank_candidates
from momentum_scanner.relvol import relvol_proxy

_log = logging.getLogger(__name__)


@dataclass
class Tier1Hit:
    ticker: str
    last: float
    prev_close: float
    gap_pct: float
    total_volume: int | None
    day_high: float | None
    avg_daily_volume: float | None
    rel_volume: float | None
    quote_time: object          # datetime (UTC) | None
    source: str


def _num(v):
    """Coerce to a finite float or None (rejects strings, NaN, inf)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _quote_fresh(quote_time, now_et):
    """True only when quote_time is present and within QUOTE_MAX_AGE_SEC of now (fail-closed).
    A symmetric tolerance also rejects an implausible far-future timestamp; small clock skew is OK."""
    if quote_time is None:
        return False
    try:
        age = abs((now_et.astimezone(timezone.utc) - quote_time).total_seconds())
    except (TypeError, ValueError, AttributeError):
        return False
    return age <= config.QUOTE_MAX_AGE_SEC


NEAR_MISS_GAP_MARGIN = 0.02
NEAR_MISS_FLOAT_MARGIN = 20_000_000


_TIER1_STAT_KEYS = (
    "quotes_seen", "skip_stale_quote", "skip_price_band", "skip_gap",
    "skip_volume_floor", "skip_low_relvol", "tier1_hits",
)
_TIER2_STAT_KEYS = ("tier2_in", "tier2_dropped_highfloat", "tier2_dropped_no_hod", "candidates",
                   "edgar_presence_hit", "edgar_presence_error", "edgar_presence_disabled",
                   "edgar_presence_rate_limited")


def _bump(stats, key):
    if stats is not None:
        stats[key] = stats.get(key, 0) + 1


def _init_stats(stats, keys):
    """Pre-initialize all expected keys to 0 so consumers get a complete schema even on zero counts."""
    if stats is not None:
        for k in keys:
            stats.setdefault(k, 0)


def tier1_filter(quotes, now_et, *, premarket, stats=None, near_miss=None):
    _init_stats(stats, _TIER1_STAT_KEYS)
    vol_floor = config.VOLUME_FLOOR_PREMARKET if premarket else config.VOLUME_FLOOR_INTRADAY
    relvol_min = config.RELVOL_MIN_PREMARKET if premarket else config.RELVOL_MIN_INTRADAY
    hits = []
    for ticker, q in quotes.items():
        _bump(stats, "quotes_seen")
        try:
            if not _quote_fresh(getattr(q, "quote_time", None), now_et):
                _bump(stats, "skip_stale_quote"); continue
            last = _num(getattr(q, "last", None))
            prev = _num(getattr(q, "prev_close", None))
            if last is None or prev is None or prev <= 0:
                _bump(stats, "skip_price_band"); continue
            if not (config.PRICE_MIN <= last <= config.PRICE_MAX):
                _bump(stats, "skip_price_band"); continue
            gap = (last - prev) / prev
            if gap < config.GAP_MIN_PCT:
                if near_miss is not None:
                    try:
                        nm_vol = _num(getattr(q, "total_volume", None))
                        nm_relvol = (_num(relvol_proxy(nm_vol, getattr(q, "avg_daily_volume", None),
                                                       now_et, premarket=premarket))
                                     if nm_vol is not None else None)
                        if (nm_vol is not None and nm_vol >= vol_floor
                                and (nm_relvol is None or nm_relvol >= relvol_min)
                                and config.GAP_MIN_PCT - NEAR_MISS_GAP_MARGIN <= gap < config.GAP_MIN_PCT):
                            near_miss({"ticker": ticker, "reason": "gap", "gap_pct": round(gap, 4),
                                      "rel_volume": (round(nm_relvol, 2) if nm_relvol is not None else None),
                                      "price": last})
                    except Exception:
                        pass
                _bump(stats, "skip_gap"); continue
            vol = _num(getattr(q, "total_volume", None))
            if vol is None or vol < vol_floor:
                _bump(stats, "skip_volume_floor"); continue
            relvol = _num(relvol_proxy(vol, getattr(q, "avg_daily_volume", None), now_et, premarket=premarket))
            if relvol is not None and relvol < relvol_min:
                _bump(stats, "skip_low_relvol"); continue
            hits.append(Tier1Hit(
                ticker=ticker, last=round(last, 4), prev_close=round(prev, 4),
                gap_pct=round(gap, 4), total_volume=int(vol),
                day_high=_num(getattr(q, "day_high", None)),
                avg_daily_volume=_num(getattr(q, "avg_daily_volume", None)),
                rel_volume=(round(relvol, 2) if relvol is not None else None),
                quote_time=getattr(q, "quote_time", None), source=getattr(q, "source", "regular")))
            _bump(stats, "tier1_hits")
        except Exception as exc:                           # per-symbol isolation
            _log.debug("tier1 skip %s: %s", ticker, exc)
            continue
    return hits


@dataclass
class MoverCandidate:
    ticker: str
    company_name: str
    premarket_price: float          # last price (BOTH modes); v2 contract entry-price field
    prev_close: float
    gap_pct: float
    float_shares: float | None
    premarket_volume: int | None    # cumulative volume (v2 contract name)
    baseline_volume: float | None   # avg_daily_volume used (v2 contract name)
    rel_volume: float | None
    day_high: float | None
    new_hod: bool | None            # additive; intraday boolean, premarket None
    scan_mode: str                  # "premarket" | "intraday"
    last_seen_at: str | None        # additive; iso8601 of the quote time
    score: float = 0.0              # set by rank_candidates
    edgar_signal: float = 0.0       # pre-judge net EDGAR term (presence - form dilution); set in tier2
    catalyst_score: float = 0.0     # post-judge catalyst bonus; set by rank_candidates
    # Score-cap provenance: stamped by rank_candidates ONLY when the
    # cap is ON and the raw effective score is finite; None otherwise, and the movers writer
    # DROPS None-valued cap keys so the OFF wire shape is byte-identical to pre-cap output.
    score_uncapped: float | None = None    # raw effective score E (pre-clamp)
    score_cap_value: float | None = None   # the active cap the stamped score was clamped to


def _iso(dt):
    try:
        return dt.isoformat() if dt is not None else None
    except Exception:
        return None


def tier2_enrich(hits, *, float_cache, float_fetch_fn, now_utc, premarket, edgar=None, stats=None,
                 near_miss=None):
    _init_stats(stats, _TIER2_STAT_KEYS)
    out = []
    for h in hits:
        _bump(stats, "tier2_in")
        try:
            fs = get_float(h.ticker, cache=float_cache, fetch_fn=float_fetch_fn, now_utc=now_utc,
                           refresh_days=config.FLOAT_REFRESH_DAYS_SHORTLIST)
            if fs is not None and fs > config.FLOAT_HARD_MAX:
                if near_miss is not None:
                    try:
                        if fs <= config.FLOAT_HARD_MAX + NEAR_MISS_FLOAT_MARGIN:
                            near_miss({"ticker": h.ticker, "reason": "float", "float_shares": fs,
                                      "gap_pct": h.gap_pct, "rel_volume": h.rel_volume, "price": h.last})
                    except Exception:
                        pass
                _bump(stats, "tier2_dropped_highfloat"); continue
            if premarket:
                new_hod = None
            else:
                new_hod = (h.day_high is not None
                           and h.last >= h.day_high * (1 - config.NEW_HOD_TOLERANCE_PCT))
                if not new_hod:
                    _bump(stats, "tier2_dropped_no_hod"); continue
            cand = MoverCandidate(
                ticker=h.ticker, company_name=h.ticker, premarket_price=h.last, prev_close=h.prev_close,
                gap_pct=h.gap_pct, float_shares=fs, premarket_volume=h.total_volume,
                baseline_volume=h.avg_daily_volume, rel_volume=h.rel_volume, day_high=h.day_high,
                new_hod=new_hod, scan_mode=("premarket" if premarket else "intraday"),
                last_seen_at=_iso(h.quote_time))
            cand.edgar_signal = _edgar_signal(h.ticker, edgar, stats)
            out.append(cand)
            _bump(stats, "candidates")
        except Exception as exc:
            _log.debug("tier2 skip %s: %s", getattr(h, "ticker", "?"), exc)
            continue
    return out


def _edgar_signal(ticker, edgar, stats):
    """Pre-judge net EDGAR term. ISOLATED from tier2's candidate-drop
    except: an EdgarContext error (raised, or a status of 'error'/'rate_limited') yields 0.0 and NEVER
    drops the candidate. edgar=None (or config disabled) -> 0.0, no lookup attempted."""
    if edgar is None or not config.EDGAR_PREJUDGE_ENABLED:
        _bump(stats, "edgar_presence_disabled")
        return 0.0
    try:
        presence, form_dilution, status = edgar.presence_and_dilution(ticker)
    except Exception as exc:
        _log.debug("edgar_signal fail-safe for %s: %s", ticker, exc)
        _bump(stats, "edgar_presence_error")
        return 0.0
    if status == "rate_limited":
        _bump(stats, "edgar_presence_rate_limited")
        return 0.0
    if status == "error":
        _bump(stats, "edgar_presence_error")
        return 0.0
    if not presence:
        return 0.0
    _bump(stats, "edgar_presence_hit")
    signal = config.EDGAR_PRESENCE_BONUS - (config.EDGAR_FORM_DILUTION_PENALTY if form_dilution else 0.0)
    return round(signal, 6)


def discover_movers(symbols, *, token_fn, session, float_cache, float_fetch_fn, now_utc, now_et,
                    premarket, edgar=None, stats=None, near_miss=None):
    """Quotes -> Tier-1 -> Tier-2 shortlist. Raises ScannerNetworkError on a whole-scan failure.
    Optional stats dict records per-cycle skip/keep counts (default None -> no-op). `edgar` (an
    EdgarContext) drives the tier-2 pre-judge signal; None -> the signal stays 0.0.
    Optional `near_miss` callback (default None -> no-op) records scored-but-untraded names that
    just missed the gap or float threshold (near-miss instrumentation)."""
    quotes = batch_quotes(symbols, token_fn=token_fn, session=session, now_utc=now_utc,
                          premarket=premarket)
    hits = tier1_filter(quotes, now_et, premarket=premarket, stats=stats, near_miss=near_miss)
    return tier2_enrich(hits, float_cache=float_cache, float_fetch_fn=float_fetch_fn,
                        now_utc=now_utc, premarket=premarket, edgar=edgar, stats=stats,
                        near_miss=near_miss)


def write_from_candidates(candidates, *, verdict_lookup=None, scan_id=None, out_path=None,
                          status="ok", error=None):
    """Rank (with looked-up verdicts) + build the v2 payload ONCE + atomic write (if out_path) + return
    the SAME payload. No discovery/network -- the movers daemon writes from the candidates it already
    discovered + judged (no second Schwab fan-out)."""
    if status != "ok":
        payload = movers_watchlist.build_movers_watchlist_dict([], {}, scan_id=scan_id, status=status,
                                                               error=error)
    else:
        verdicts = {}
        if verdict_lookup is not None:
            for c in candidates:
                try:
                    verdicts[c.ticker] = verdict_lookup(c)
                except Exception as judge_exc:
                    _log.debug("verdict_lookup failed %s: %s", c.ticker, judge_exc)
                    verdicts[c.ticker] = None
        ranked = rank_candidates(candidates, verdicts=verdicts)
        payload = movers_watchlist.build_movers_watchlist_dict(ranked, verdicts, scan_id=scan_id,
                                                               status="ok")
    if out_path:
        from momentum_scanner import watchlist_io
        watchlist_io.atomic_write_json(out_path, payload)
    return payload


def run_movers_scan(symbols, *, token_fn, session, float_cache, float_fetch_fn, now_utc, now_et,
                    premarket, verdict_lookup=None, out_path=None, scan_id=None):
    """Standalone movers scan WRITE pass = discover + write, fail-closed + build-once."""
    try:
        candidates = discover_movers(symbols, token_fn=token_fn, session=session,
                                     float_cache=float_cache, float_fetch_fn=float_fetch_fn,
                                     now_utc=now_utc, now_et=now_et, premarket=premarket)
    except ScannerNetworkError as exc:
        return write_from_candidates([], scan_id=scan_id, out_path=out_path, status="error",
                                     error=str(exc))
    return write_from_candidates(candidates, verdict_lookup=verdict_lookup, scan_id=scan_id,
                                 out_path=out_path)

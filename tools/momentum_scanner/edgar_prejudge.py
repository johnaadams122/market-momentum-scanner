"""Per-cycle EDGAR context: shared, cached CIK-map + filing-detail lookups.
One `EdgarContext` instance per discovery
cycle serves BOTH the tier-2 pre-judge check (`presence_and_dilution`, budgeted)
AND `evaluate_news` (`detail_for`, not budgeted), so a ticker that is both
pre-judged and later judged resolves its CIK and fetches its filing detail
EXACTLY ONCE. Neither public method ever raises: every error path (CIK-map
fetch failure, filing-detail fetch failure, unresolvable ticker, exhausted
budget) resolves to an explicit sentinel so a transient EDGAR outage or a
scarce per-cycle budget can never drop or crash a candidate.
"""
import logging
from datetime import datetime, date, timezone

from momentum_scanner import config, edgar_rss

_log = logging.getLogger(__name__)

_DILUTION_FORMS = config.ACTIVE_OFFERING_FORM_TYPES | config.SHELF_FORM_TYPES


def _coerce_now(now):
    """Normalize the caller-supplied reference time to a datetime so both the
    dilution-window math and edgar_rss.fetch_filing_detail(now=...) see the
    same value. None -> current UTC time (mirrors the rest of the codebase's
    `now or datetime.now(timezone.utc)` convention)."""
    if now is None:
        return datetime.now(timezone.utc)
    if isinstance(now, datetime):
        return now
    if isinstance(now, date):
        return datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    return now


class EdgarContext:
    def __init__(self, *, ticker_to_ciks=edgar_rss.ticker_to_ciks,
                 fetch_detail=edgar_rss.fetch_filing_detail, budget=None, now=None):
        self._ticker_to_ciks = ticker_to_ciks
        self._fetch_detail = fetch_detail
        self._budget = config.EDGAR_PREJUDGE_MAX_PER_CYCLE if budget is None else budget
        self._now = _coerce_now(now)

        self._cmap = None          # ticker_to_ciks() result, fetched lazily ONCE
        self._cmap_failed = False  # remember a CIK-map fetch failure (don't retry this cycle)
        self._detail_cache = {}    # ticker -> (edgar_state, detail, cik)
        self._pd_cache = {}        # ticker -> (presence, form_dilution, status)
        self._spent = 0            # fetch_detail calls charged to the budget

    # -- 1a. evaluate_news path -------------------------------------------------
    def detail_for(self, ticker):
        """Resolve CIK + fetch filing detail for `ticker`, cached for the life of
        this context. NEVER raises. NOT budget-limited (the judge path is already
        small/rate-limited upstream; only presence_and_dilution spends budget)."""
        if ticker in self._detail_cache:
            return self._detail_cache[ticker]
        result = self._detail_for_uncached(ticker)
        self._detail_cache[ticker] = result
        return result

    def _detail_for_uncached(self, ticker):
        cmap = self._resolve_cmap()
        if cmap is None:
            return ("error", None, None)
        ciks = cmap.get(edgar_rss.normalize_ticker(ticker), set())
        if len(ciks) != 1:
            return ("unresolvable", None, None)
        cik = next(iter(ciks))
        try:
            detail = self._fetch_detail(cik, now=self._now)
        except Exception as exc:
            _log.debug("EdgarContext.detail_for(%s): fetch_detail failed: %s", ticker, exc)
            return ("error", None, cik)
        return ("ok", detail, cik)

    # -- 1b. tier-2 pre-judge path -----------------------------------------------
    def presence_and_dilution(self, ticker):
        """Cheap presence/dilution read for the tier-2 pre-judge queue-reorder.
        Budgeted: a NEW fetch beyond `budget` returns rate_limited without
        ever calling fetch_detail. NEVER raises. Served from `_pd_cache` on
        repeat -- EXCEPT a cached `rate_limited` verdict for a ticker whose
        detail has since been resolved (e.g. via a `detail_for` call from the
        judge path, which is not budget-limited) is stale and recomputed for
        free from the now-cached detail rather than replayed forever."""
        cached = self._pd_cache.get(ticker)
        if cached is not None:
            stale_rate_limited = cached[2] == "rate_limited" and ticker in self._detail_cache
            if not stale_rate_limited:
                return cached
        result = self._presence_and_dilution_uncached(ticker)
        self._pd_cache[ticker] = result
        return result

    def _presence_and_dilution_uncached(self, ticker):
        is_new_fetch = ticker not in self._detail_cache
        if is_new_fetch and self._spent >= self._budget:
            return (False, False, "rate_limited")

        edgar_state, detail, cik = self.detail_for(ticker)

        if is_new_fetch and cik is not None and edgar_state in ("ok", "error"):
            self._spent += 1

        if edgar_state == "ok":
            try:
                presence = len(detail.recent_filings) > 0
                form_dilution = any(
                    r["form"] in _DILUTION_FORMS and self._within_dilution_window(r["filing_date"])
                    for r in detail.recent_filings
                )
                return (presence, form_dilution, "ok")
            except Exception as exc:
                # Defense in depth: a malformed detail object must never raise out
                # of this public method, even though a real fetch_detail() result
                # always has this shape.
                _log.debug("EdgarContext: malformed detail for %s: %s", ticker, exc)
                return (False, False, "error")
        if edgar_state == "unresolvable":
            return (False, False, "ok")
        # edgar_state == "error" (cmap fetch failed OR fetch_detail raised)
        return (False, False, "error")

    # -- shared helpers -----------------------------------------------------
    def _resolve_cmap(self):
        if self._cmap is not None:
            return self._cmap
        if self._cmap_failed:
            return None
        try:
            self._cmap = self._ticker_to_ciks()
        except Exception as exc:
            _log.debug("EdgarContext: ticker_to_ciks() failed, edgar unavailable this cycle: %s", exc)
            self._cmap_failed = True
            return None
        return self._cmap

    def _within_dilution_window(self, filing_date):
        """form_dilution counts only filings within
        config.DILUTION_LOOKBACK_CALENDAR_DAYS of the reference date (presence
        itself uses the full recent_filings window, no filtering)."""
        today = self._now.date()
        return (today - filing_date).days <= config.DILUTION_LOOKBACK_CALENDAR_DAYS

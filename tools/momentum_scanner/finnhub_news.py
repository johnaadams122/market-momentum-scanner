"""Finnhub free news-wire client -- the PRIMARY premarket trigger.

Surfaces wire PRs in the ET 04:00-09:30 window that EDGAR's 8-K firehose misses. The API key is
read from env and sent ONLY as the X-Finnhub-Token header (never in a URL/param, never logged).
Finnhub response fields are all OPTIONAL, so items are validated and dropped if unusable. Systemic
failures (missing key / 401 / 429 / HTTP / connection / a body that is not a JSON list) raise
ScannerNetworkError so the runner writes
a status:"error" watchlist. Free tier is non-commercial -> the live-gate guard lives in the runner.
"""
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests

from momentum_scanner.config import (
    FINNHUB_BASE_URL,
    FINNHUB_WINDOW_START_ET,
    FINNHUB_WINDOW_END_ET,
    FINNHUB_MAX_CONFIRM_CALLS,
    SCANNER_HTTP_TIMEOUT,
)
from momentum_scanner.edgar_rss import normalize_ticker
from momentum_scanner.errors import ScannerNetworkError

ET = ZoneInfo("America/New_York")
_log = logging.getLogger("momentum_scanner.finnhub_news")

# The provider literal this client stamps on every NewsItem it builds. Named so the weight-table
# coverage guard (tests/test_catalyst_source_weight_coverage.py) can assert config.SOURCE_WEIGHTS
# carries an entry for it -- a provider with no entry silently scores every catalyst 0.0.
PROVIDER = "finnhub"


@dataclass(frozen=True)
class NewsItem:
    ticker: str
    headline: str
    summary: str
    published: datetime          # UTC
    url: str
    source: str
    id: object                   # Finnhub item id (int|None); used for dedup + trigger confirmation
    provider: str = PROVIDER     # which API client fetched this item ("finnhub" | "alpaca") --
                                  # distinct from `source`, which is the wire/publisher name inside
                                  # that API's response. Lets catalyst_evaluator stamp the verdict's
                                  # content_source with the true provider instead of a hardcoded literal.


def _hhmm(s):
    h, m = s.split(":")
    return int(h), int(m)


_MISSING = object()


def _text(value):
    """A provider text field as a stripped str. Absent/null/empty -> "" (the field is optional); any
    other non-string value is MALFORMED -> the _MISSING sentinel, and the caller drops the item."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return _MISSING


def _opt_str(value):
    """Optional descriptive string (url, source): anything that is not a string becomes ""."""
    return value if isinstance(value, str) else ""


def _opt_id(value):
    """Item id used for dedup: an int or str, else None (unhashable/odd types never reach a set)."""
    return value if isinstance(value, (int, str)) and not isinstance(value, bool) else None


class FinnhubClient:
    def __init__(self, *, api_key=None, session=None, now_fn=None,
                 base_url=FINNHUB_BASE_URL,
                 window_start_et=FINNHUB_WINDOW_START_ET,
                 window_end_et=FINNHUB_WINDOW_END_ET):
        self._api_key = api_key
        self._session = session or requests
        self._base = base_url
        self._ws = _hhmm(window_start_et)
        self._we = _hhmm(window_end_et)

    def _key(self):
        k = self._api_key if self._api_key is not None else os.environ.get("FINNHUB_API_KEY")
        if not k:
            raise ScannerNetworkError("FINNHUB_API_KEY not set")   # fail-closed; no key value in msg
        return k

    def _get(self, path, params):
        try:
            resp = self._session.get(f"{self._base}{path}",
                                     headers={"X-Finnhub-Token": self._key()}, params=params,
                                     timeout=SCANNER_HTTP_TIMEOUT)
        except ScannerNetworkError:
            raise
        except Exception as exc:                       # connection/timeout -> systemic (no key in msg)
            raise ScannerNetworkError(f"Finnhub request failed: {type(exc).__name__}") from None
        status = getattr(resp, "status_code", 200)
        if status >= 400:                              # 401/429/5xx -> systemic
            raise ScannerNetworkError(f"Finnhub HTTP {status}")
        try:
            body = resp.json()
        except Exception as exc:                       # malformed body -> systemic, like a transport error
            raise ScannerNetworkError(f"Finnhub response was not valid JSON: {type(exc).__name__}") from None
        if not isinstance(body, list):                 # null or an error object is not a list of news items
            raise ScannerNetworkError("Finnhub response was not a list of news items")
        return body

    def _window(self, now_et):
        now_et = now_et.astimezone(ET)
        start = now_et.replace(hour=self._ws[0], minute=self._ws[1], second=0, microsecond=0)
        end_cap = now_et.replace(hour=self._we[0], minute=self._we[1], second=0, microsecond=0)
        return start, min(now_et, end_cap)             # [04:00 ET, min(now, 09:30 ET)]

    @staticmethod
    def _item(raw, ticker):
        """Validate one raw Finnhub item for a ticker; None if unusable (optional-field-safe)."""
        if not isinstance(raw, dict):                  # a non-object list member is an unusable item
            return None
        dt = raw.get("datetime")
        headline = _text(raw.get("headline"))
        summary = _text(raw.get("summary"))
        if headline is _MISSING or summary is _MISSING:    # wrong-typed text field -> unusable item
            return None
        if dt is None or not ticker or (not headline and not summary):
            return None
        try:
            published = datetime.fromtimestamp(int(dt), tz=timezone.utc)
        except (ValueError, OSError, TypeError, OverflowError):
            return None
        return NewsItem(ticker=ticker, headline=headline, summary=summary, published=published,
                        url=_opt_str(raw.get("url")), source=_opt_str(raw.get("source")),
                        id=_opt_id(raw.get("id")))

    def fetch_trigger_news(self, now_et):
        """GET /news?category=general -> one validated, in-window NewsItem per related ticker."""
        start, end = self._window(now_et)
        out, seen = [], set()
        for r in self._get("/news", {"category": "general"}):
            if not isinstance(r, dict):                # a non-object list member is an unusable item
                continue
            related = r.get("related")
            if not isinstance(related, str):           # absent -> nothing to explode; wrong type -> unusable
                continue
            for tk in related.split(","):
                item = self._item(r, normalize_ticker(tk))
                if item is None or not (start <= item.published.astimezone(ET) <= end):
                    continue
                key = (item.ticker, item.id if item.id is not None else item.url)
                if key in seen:
                    continue
                seen.add(key)
                out.append(item)
        return out

    def fetch_company_news(self, ticker, frm_date, to_date):
        """GET /company-news?symbol=&from=&to= -> validated NewsItems tagged with the queried symbol."""
        t = normalize_ticker(ticker)
        out, seen = [], set()
        for r in self._get("/company-news", {"symbol": t, "from": frm_date, "to": to_date}):
            item = self._item(r, t)
            if item is None:
                continue
            key = item.id if item.id is not None else item.url
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out


def et_window(now_et):
    """[today 04:00 ET, min(now, today 09:30 ET)] -- the one ET-anchored catalyst window."""
    now_et = now_et.astimezone(ET)
    ws, we = _hhmm(FINNHUB_WINDOW_START_ET), _hhmm(FINNHUB_WINDOW_END_ET)
    start = now_et.replace(hour=ws[0], minute=ws[1], second=0, microsecond=0)
    end_cap = now_et.replace(hour=we[0], minute=we[1], second=0, microsecond=0)
    return start, min(now_et, end_cap)


def _norm_text(s):
    return " ".join((s or "").lower().split())


def _match_key(item):
    return (_norm_text(item.headline), (item.source or "").lower().strip(),
            int(item.published.timestamp()))


def aggregate_news(trigger_items, client, now_et, *, max_confirm=FINNHUB_MAX_CONFIRM_CALLS, stats=None):
    """Per ticker: CONFIRM at least one TRIGGER item exists in /company-news (by id, else url, else a
    normalized (headline, source, datetime) tuple), then MERGE all same-window items. A ticker whose
    trigger item is not confirmed is dropped (confirm the trigger itself, not just 'some news').
    The /company-news fan-out is capped at max_confirm. The dropped count is logged AND, when a `stats`
    dict is passed, written to stats['confirm_overflow'] so the runner can surface it (on busy
    mornings the cap can silently degrade the scan to the first max_confirm tickers)."""
    by_ticker = {}
    for it in trigger_items:
        by_ticker.setdefault(it.ticker, []).append(it)
    tickers = list(by_ticker.keys())
    overflow = max(0, len(tickers) - max_confirm)
    if overflow:
        _log.warning("Finnhub confirm fan-out capped at %d; dropping %d ticker(s): %s",
                     max_confirm, overflow, tickers[max_confirm:])
        tickers = tickers[:max_confirm]
    if stats is not None:
        stats["confirm_overflow"] = overflow

    start, end = et_window(now_et)
    today = now_et.astimezone(ET).date().isoformat()
    confirmed = {}
    for t in tickers:
        trig = by_ticker[t]
        company = client.fetch_company_news(t, today, today)   # ScannerNetworkError propagates (systemic)
        ids = {c.id for c in company if c.id is not None}
        urls = {c.url for c in company if c.url}
        keys = {_match_key(c) for c in company}

        def _matched(item):
            return ((item.id is not None and item.id in ids)
                    or (item.url and item.url in urls)
                    or (_match_key(item) in keys))

        if not any(_matched(it) for it in trig):
            continue                                           # trigger not confirmed -> drop
        merged, seen = [], set()
        for item in list(trig) + list(company):
            if not (start <= item.published.astimezone(ET) <= end):
                continue                                       # same-window only
            key = ("id", item.id) if item.id is not None else ("url", item.url)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
        if merged:
            confirmed[t] = merged
    return confirmed

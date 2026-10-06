'Alpaca news client (article -> NewsItem mapping, pagination, dedup) and a primary/secondary fallback news client.'
import logging
import os
from datetime import date, datetime, timedelta, timezone

import requests

from momentum_scanner.config import (
    ALPACA_NEWS_BASE_URL,
    ALPACA_NEWS_LIMIT,
    ALPACA_NEWS_MAX_PAGES,
    SCANNER_HTTP_TIMEOUT,
)
from momentum_scanner.edgar_rss import normalize_ticker
from momentum_scanner.errors import ScannerNetworkError
from momentum_scanner.finnhub_news import NewsItem, _MISSING, _opt_id, _opt_str, _text

_log = logging.getLogger("momentum_scanner.alpaca_news")

# The provider literal this client stamps on every NewsItem it builds; catalyst_evaluator copies
# it onto the verdict's content_source, which ranking.catalyst_bonus weights by. Named so the
# coverage guard (tests/test_catalyst_source_weight_coverage.py) can assert SOURCE_WEIGHTS has an
# entry -- a missing entry would silently zero every Alpaca catalyst bonus.
PROVIDER = "alpaca"


def _parse_dt(s):
    """Alpaca created_at is RFC3339 (e.g. 2026-07-02T13:05:37Z) -> tz-aware UTC datetime, or None."""
    if not s or not isinstance(s, str):
        return None
    try:
        txt = s.strip()
        if txt.endswith("Z"):
            txt = txt[:-1] + "+00:00"
        dt = datetime.fromisoformat(txt)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _end_exclusive(to_date):
    """[from 00:00Z, to_date+1d 00:00Z): a full-UTC-day-inclusive window for the ET to_date. The judge
    re-trims to the exact ET catalyst window, so over-fetching by a day margin is safe (never under)."""
    d = date.fromisoformat(to_date) + timedelta(days=1)
    return f"{d.isoformat()}T00:00:00Z"


class AlpacaNewsClient:
    def __init__(self, *, api_key=None, api_secret=None, session=None,
                 base_url=ALPACA_NEWS_BASE_URL, limit=ALPACA_NEWS_LIMIT, max_pages=ALPACA_NEWS_MAX_PAGES):
        self._api_key = api_key
        self._api_secret = api_secret
        self._session = session or requests
        self._base = base_url
        self._limit = limit
        self._max_pages = max_pages

    def _creds(self):
        k = self._api_key if self._api_key is not None else os.environ.get("ALPACA_API_KEY_ID")
        s = self._api_secret if self._api_secret is not None else os.environ.get("ALPACA_API_SECRET_KEY")
        if not k or not s:                              # fail-closed; no secret value in the message
            raise ScannerNetworkError("ALPACA_API_KEY_ID/ALPACA_API_SECRET_KEY not set")
        return k, s

    def _get(self, params):
        k, s = self._creds()
        try:
            resp = self._session.get(
                self._base,
                headers={"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": s},
                params=params, timeout=SCANNER_HTTP_TIMEOUT)
        except ScannerNetworkError:
            raise
        except Exception as exc:                        # connection/timeout -> systemic (no secret in msg)
            raise ScannerNetworkError(f"Alpaca request failed: {type(exc).__name__}") from None
        status = getattr(resp, "status_code", 200)
        if status >= 400:                               # 401/429/5xx -> systemic
            raise ScannerNetworkError(f"Alpaca HTTP {status}")
        try:                                            # a 200 with malformed/non-JSON body is systemic
            data = resp.json()
        except Exception as exc:                        # (no body content in msg) -> takes the fallback path
            raise ScannerNetworkError(f"Alpaca bad body: {type(exc).__name__}") from None
        if not isinstance(data, dict):                  # null, or e.g. an error HTML page decoded to a list/str
            raise ScannerNetworkError("Alpaca bad body: non-object")
        if "news" in data:                              # absent -> no news; present -> a list of objects
            news = data["news"]
            if not isinstance(news, list) or not all(isinstance(r, dict) for r in news):
                raise ScannerNetworkError("Alpaca bad body: news is not a list of objects")
        return data

    @staticmethod
    def _item(raw, ticker):
        """Validate + map one Alpaca article to a NewsItem for `ticker`; None if unusable."""
        headline = _text(raw.get("headline"))
        summary = _text(raw.get("summary"))
        content = _text(raw.get("content"))
        if _MISSING in (headline, summary, content):    # wrong-typed text field -> unusable item
            return None
        body = summary or content                       # keep the body; never require it
        published = _parse_dt(raw.get("created_at"))
        if published is None or not ticker or (not headline and not body):
            return None
        return NewsItem(ticker=ticker, headline=headline, summary=body, published=published,
                        url=_opt_str(raw.get("url")), source=_opt_str(raw.get("source")) or "benzinga",
                        id=_opt_id(raw.get("id")), provider=PROVIDER)

    def fetch_company_news(self, ticker, frm_date, to_date):
        """GET /v1beta1/news?symbols=&start=&end=&include_content=true -> NewsItems tagged with the
        queried symbol (Alpaca returns a symbols array per article; we tag with the queried ticker like
        FinnhubClient). Follows next_page_token up to max_pages; dedups by id (else url)."""
        t = normalize_ticker(ticker)
        start = f"{frm_date}T00:00:00Z"
        end = _end_exclusive(to_date)
        out, seen, token = [], set(), None
        for _ in range(self._max_pages):
            params = {"symbols": t, "start": start, "end": end, "limit": self._limit,
                      "include_content": "true", "sort": "desc"}
            if token:
                params["page_token"] = token
            data = self._get(params)
            for r in (data.get("news") or []):
                item = self._item(r, t)
                if item is None:
                    continue
                key = item.id if item.id is not None else item.url
                if key in seen:
                    continue
                seen.add(key)
                out.append(item)
            token = data.get("next_page_token")
            if not token:
                break
        return out


class FallbackNewsClient:
    """Try the primary news client; on a SYSTEMIC outage (ScannerNetworkError) fall back to the
    secondary so a single-feed outage DEGRADES rather than blanks the catalyst signal.
    An empty (non-error) primary result is a valid 'no news' and is returned as-is -- not a reason to
    double-fetch. If BOTH error, the error propagates so the judge leaves the ticker unjudged."""

    def __init__(self, primary, secondary):
        self._primary = primary
        self._secondary = secondary

    def fetch_company_news(self, ticker, frm_date, to_date):
        try:
            return self._primary.fetch_company_news(ticker, frm_date, to_date)
        except ScannerNetworkError as exc:
            _log.warning("primary news feed failed (%s); falling back to secondary for %s",
                         type(exc).__name__, ticker)
            return self._secondary.fetch_company_news(ticker, frm_date, to_date)

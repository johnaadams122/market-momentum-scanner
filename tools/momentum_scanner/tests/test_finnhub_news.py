AUTH_FIXTURE = "synthetic-fixture-value"

"""FinnhubClient (news fetch, ET window, header-token, fail-closed)."""
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

ET = ZoneInfo("America/New_York")


def _et(y, mo, d, h, mi):
    return datetime(y, mo, d, h, mi, tzinfo=ET)


def _raw(id, related, dt, headline="h", summary="s"):
    return {"id": id, "related": related, "headline": headline, "summary": summary,
            "datetime": int(dt.timestamp()), "url": f"http://x/{id}", "source": "wire"}


class FakeResp:
    def __init__(self, data, status=200):
        self._d = data
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400 and self.status_code not in (401, 429):
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._d


class FakeSession:
    def __init__(self, routes):
        self.routes = routes      # {url_substr: data | (data, status)}
        self.calls = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append({"url": url, "headers": headers or {}, "params": params or {}, "timeout": timeout})
        for key, val in self.routes.items():
            if key in url:
                return FakeResp(*val) if isinstance(val, tuple) else FakeResp(val)
        return FakeResp([], 200)


def _client(routes, api_key="K"):
    from momentum_scanner.finnhub_news import FinnhubClient
    return FinnhubClient(api_key=api_key, session=FakeSession(routes))


def test_trigger_news_explodes_related_and_windows():
    now = _et(2026, 6, 29, 8, 0)
    items = [_raw(1, "AAA,BBB", _et(2026, 6, 29, 5, 0)),
             _raw(2, "CCC", _et(2026, 6, 29, 3, 59))]   # before 04:00 -> drop
    out = _client({"/news": items}).fetch_trigger_news(now)
    assert {n.ticker for n in out} == {"AAA", "BBB"}


def test_window_boundaries_and_delayed_scan():
    # scan at 10:00 (after open) -> window end clamps to 09:30
    now = _et(2026, 6, 29, 10, 0)
    items = [_raw(10, "A", _et(2026, 6, 29, 3, 59)),    # < 04:00 -> drop
             _raw(11, "B", _et(2026, 6, 29, 4, 0)),     # 04:00 -> keep
             _raw(12, "C", _et(2026, 6, 29, 9, 30)),    # 09:30 boundary -> keep
             _raw(13, "D", _et(2026, 6, 29, 9, 45))]    # after 09:30 -> drop (delayed scan)
    out = _client({"/news": items}).fetch_trigger_news(now)
    assert {n.ticker for n in out} == {"B", "C"}


def test_drops_malformed_items():
    now = _et(2026, 6, 29, 8, 0)
    items = [_raw(20, "A", _et(2026, 6, 29, 5, 0), headline="", summary=""),   # empty -> drop
             _raw(21, "", _et(2026, 6, 29, 5, 0)),                              # no ticker -> drop
             {"id": 22, "related": "C", "headline": "h", "summary": "s"},       # no datetime -> drop
             _raw(23, "D", _et(2026, 6, 29, 5, 0))]                             # valid
    out = _client({"/news": items}).fetch_trigger_news(now)
    assert {n.ticker for n in out} == {"D"}


def test_dedup_by_ticker_and_id():
    now = _et(2026, 6, 29, 8, 0)
    items = [_raw(1, "A", _et(2026, 6, 29, 5, 0)), _raw(1, "A", _et(2026, 6, 29, 5, 5))]
    out = _client({"/news": items}).fetch_trigger_news(now)
    assert len([n for n in out if n.ticker == "A"]) == 1


def test_token_sent_as_header_never_in_url_or_params():
    now = _et(2026, 6, 29, 8, 0)
    c = _client({"/news": [_raw(1, "A", _et(2026, 6, 29, 5, 0))]}, api_key=AUTH_FIXTURE)
    c.fetch_trigger_news(now)
    call = c._session.calls[0]
    assert call["headers"].get("X-Finnhub-Token") == AUTH_FIXTURE
    blob = str(call["params"]) + call["url"]
    assert AUTH_FIXTURE not in blob and "token" not in blob.lower()
    assert call["timeout"] == 15   # explicit network timeout (never hang the run)


def test_401_escalates_without_leaking_key():
    from momentum_scanner.errors import ScannerNetworkError
    c = _client({"/news": ([], 401)}, api_key=AUTH_FIXTURE)
    with pytest.raises(ScannerNetworkError) as e:
        c.fetch_trigger_news(_et(2026, 6, 29, 8, 0))
    assert AUTH_FIXTURE not in str(e.value)


def test_429_escalates():
    from momentum_scanner.errors import ScannerNetworkError
    c = _client({"/news": ([], 429)})
    with pytest.raises(ScannerNetworkError):
        c.fetch_trigger_news(_et(2026, 6, 29, 8, 0))


def test_connection_error_escalates():
    import requests
    from momentum_scanner.finnhub_news import FinnhubClient
    from momentum_scanner.errors import ScannerNetworkError

    class Boom:
        def get(self, *a, **k):
            raise requests.ConnectionError("finnhub down")
    with pytest.raises(ScannerNetworkError):
        FinnhubClient(api_key="K", session=Boom()).fetch_trigger_news(_et(2026, 6, 29, 8, 0))


def test_missing_api_key_escalates(monkeypatch):
    from momentum_scanner.finnhub_news import FinnhubClient
    from momentum_scanner.errors import ScannerNetworkError
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    c = FinnhubClient(session=FakeSession({"/news": []}))   # api_key=None -> env (absent)
    with pytest.raises(ScannerNetworkError):
        c.fetch_trigger_news(_et(2026, 6, 29, 8, 0))


def test_company_news_sets_ticker_and_params():
    sess_routes = {"/company-news": [_raw(5, "ignored", _et(2026, 6, 29, 5, 0))]}
    c = _client(sess_routes)
    out = c.fetch_company_news("aapl", "2026-06-29", "2026-06-29")
    p = c._session.calls[0]["params"]
    assert p["symbol"] == "AAPL" and p["from"] == "2026-06-29" and p["to"] == "2026-06-29"
    assert out and out[0].ticker == "AAPL"   # company-news items carry the queried symbol


# ---- aggregate_news (confirm the trigger item, then merge; cap fan-out) ----
from datetime import timezone   # noqa: E402


def _ni(ticker, *, id=1, url=None, headline="h", source="wire", dt=None):
    from momentum_scanner.finnhub_news import NewsItem
    d = dt or _et(2026, 6, 29, 5, 0)
    return NewsItem(ticker=ticker, headline=headline, summary="s", published=d.astimezone(timezone.utc),
                    url=url if url is not None else f"http://x/{id}", source=source, id=id)


def test_aggregate_confirms_by_id_and_merges():
    from momentum_scanner.finnhub_news import aggregate_news
    now = _et(2026, 6, 29, 8, 0)
    trig = [_ni("AAA", id=1)]

    class C:
        def fetch_company_news(self, t, f, to):
            return [_ni("AAA", id=1), _ni("AAA", id=2, dt=_et(2026, 6, 29, 6, 0))]
    out = aggregate_news(trig, C(), now)
    assert "AAA" in out and len(out["AAA"]) == 2   # trigger(id1) + company id2; id1 deduped


def test_aggregate_drops_when_trigger_not_confirmed():
    from momentum_scanner.finnhub_news import aggregate_news
    now = _et(2026, 6, 29, 8, 0)
    trig = [_ni("AAA", id=1, headline="Trigger PR")]

    class C:
        def fetch_company_news(self, t, f, to):     # unrelated item: different id/url/headline/time
            return [_ni("AAA", id=99, headline="Other", dt=_et(2026, 6, 29, 6, 30))]
    out = aggregate_news(trig, C(), now)
    assert "AAA" not in out   # the actual trigger item was never confirmed -> dropped


def test_aggregate_confirms_via_normalized_tuple_fallback():
    from momentum_scanner.finnhub_news import aggregate_news
    now = _et(2026, 6, 29, 8, 0)
    dt = _et(2026, 6, 29, 5, 0)
    trig = [_ni("AAA", id=None, url="", headline="Big News", source="Wire", dt=dt)]

    class C:
        def fetch_company_news(self, t, f, to):     # different id+url, same normalized headline/source/time
            return [_ni("AAA", id=7, url="http://y", headline="big   news", source="wire", dt=dt)]
    out = aggregate_news(trig, C(), now)
    assert "AAA" in out   # matched via normalized (headline, source, datetime)


def test_aggregate_caps_confirm_fanout_and_logs(caplog):
    from momentum_scanner.finnhub_news import aggregate_news
    now = _et(2026, 6, 29, 8, 0)
    trig = [_ni(f"T{i}", id=i) for i in range(5)]
    calls = []

    class C:
        def fetch_company_news(self, t, f, to):
            calls.append(t)
            return []
    with caplog.at_level("WARNING"):
        aggregate_news(trig, C(), now, max_confirm=2)
    assert len(calls) == 2   # fan-out capped
    assert any("cap" in r.message.lower() or "drop" in r.message.lower() for r in caplog.records)


def test_aggregate_reports_confirm_overflow_via_stats():
    """The dropped-ticker count must be reported (not just logged) so the runner can
    surface it in the watchlist artifact -- on busy mornings the cap silently hides tickers."""
    from momentum_scanner.finnhub_news import aggregate_news
    now = _et(2026, 6, 29, 8, 0)
    trig = [_ni(f"T{i}", id=i) for i in range(5)]

    class C:
        def fetch_company_news(self, t, f, to):
            return []
    stats = {}
    aggregate_news(trig, C(), now, max_confirm=2, stats=stats)
    assert stats["confirm_overflow"] == 3   # 5 trigger tickers - cap 2


def test_aggregate_no_overflow_reports_zero():
    from momentum_scanner.finnhub_news import aggregate_news
    now = _et(2026, 6, 29, 8, 0)
    trig = [_ni("AAA", id=1)]

    class C:
        def fetch_company_news(self, t, f, to):
            return []
    stats = {}
    aggregate_news(trig, C(), now, stats=stats)
    assert stats["confirm_overflow"] == 0


class _BadJsonResp(FakeResp):
    def json(self):
        raise ValueError("Expecting value: line 1 column 1 (char 0)")


class _BadJsonSession(FakeSession):
    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append({"url": url})
        return _BadJsonResp(None, 200)


def test_malformed_json_body_is_systemic():
    # A 200 response whose body is not JSON takes the same systemic path as a transport error.
    from momentum_scanner.errors import ScannerNetworkError
    from momentum_scanner.finnhub_news import FinnhubClient
    client = FinnhubClient(api_key="K", session=_BadJsonSession({}))
    with pytest.raises(ScannerNetworkError):
        client.fetch_trigger_news(_et(2026, 6, 29, 8, 0))


def test_non_list_json_body_is_systemic():
    # A JSON object (for example an error document) instead of a list of items is not news.
    from momentum_scanner.errors import ScannerNetworkError
    with pytest.raises(ScannerNetworkError):
        _client({"/news": {"error": "synthetic"}}).fetch_trigger_news(_et(2026, 6, 29, 8, 0))


def test_null_json_body_is_systemic():
    # A JSON null body is a malformed response, not "no news": it takes the systemic path too.
    from momentum_scanner.errors import ScannerNetworkError
    with pytest.raises(ScannerNetworkError):
        _client({"/news": None}).fetch_trigger_news(_et(2026, 6, 29, 8, 0))
    with pytest.raises(ScannerNetworkError):
        _client({"/company-news": None}).fetch_company_news("AAA", "2026-06-29", "2026-06-29")


_NON_OBJECT_MEMBERS = [None, "synthetic text", 7, 1.5, True, ["nested"]]


def test_trigger_news_skips_non_object_members():
    # A list member that is not a JSON object is an unusable item: skipped, never an AttributeError.
    now = _et(2026, 6, 29, 8, 0)
    items = list(_NON_OBJECT_MEMBERS) + [_raw(30, "D", _et(2026, 6, 29, 5, 0))]
    out = _client({"/news": items}).fetch_trigger_news(now)
    assert [n.ticker for n in out] == ["D"]


def test_company_news_skips_non_object_members():
    items = list(_NON_OBJECT_MEMBERS) + [_raw(31, "ignored", _et(2026, 6, 29, 5, 0))]
    out = _client({"/company-news": items}).fetch_company_news("aaa", "2026-06-29", "2026-06-29")
    assert [(n.ticker, n.id) for n in out] == [("AAA", 31)]


# ---- malformed field TYPES are dropped items, never an AttributeError out of the client ----

@pytest.mark.parametrize("bad", [
    {"headline": 5}, {"headline": ["a"]}, {"summary": 7.5}, {"summary": {"x": 1}},
    {"headline": True}, {"related": 12}, {"related": ["A"]},
])
def test_non_string_text_fields_drop_the_item_without_raising(bad):
    now = _et(2026, 6, 29, 8, 0)
    broken = _raw(30, "A", _et(2026, 6, 29, 5, 0))
    broken.update(bad)
    items = [broken, _raw(31, "D", _et(2026, 6, 29, 5, 0))]
    out = _client({"/news": items}).fetch_trigger_news(now)
    assert {n.ticker for n in out} == {"D"}


def test_non_string_fields_in_company_news_drop_the_item():
    broken = _raw(40, "A", _et(2026, 6, 29, 5, 0))
    broken["headline"] = 123
    ok = _raw(41, "A", _et(2026, 6, 29, 5, 0))
    out = _client({"/company-news": [broken, ok]}).fetch_company_news("A", "2026-06-29", "2026-06-29")
    assert [i.id for i in out] == [41]


def test_non_string_url_source_and_unhashable_id_do_not_break_the_item():
    now = _et(2026, 6, 29, 8, 0)
    odd = _raw(50, "A", _et(2026, 6, 29, 5, 0))
    odd.update({"url": 5, "source": ["x"], "id": ["not", "hashable"]})
    out = _client({"/news": [odd]}).fetch_trigger_news(now)
    assert len(out) == 1
    assert out[0].url == "" and out[0].source == "" and out[0].id is None

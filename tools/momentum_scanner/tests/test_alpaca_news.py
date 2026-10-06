AUTH_FIXTURE = "synthetic-fixture-value"

"""AlpacaNewsClient -- Benzinga-sourced news CONTENT for the catalyst
judge, drop-in for FinnhubClient.fetch_company_news(ticker, from, to) -> NewsItem. Keys sent only as
APCA-* headers (never leaked); fail-closed (ScannerNetworkError) on missing creds / 401 / 429 / HTTP /
connection so the judge leaves the ticker unjudged rather than caching an outage REJECT.
"""
import pytest


def _raw(id, symbols, created_at, headline="Head", summary="Sum", content="Body",
         url=None, source="benzinga"):
    return {"id": id, "symbols": symbols, "created_at": created_at, "headline": headline,
            "summary": summary, "content": content, "url": url if url is not None else f"http://a/{id}",
            "source": source, "author": "wire"}


class FakeResp:
    def __init__(self, data, status=200):
        self._d = data
        self.status_code = status

    def json(self):
        return self._d


class FakeSession:
    """Returns `pages` in order (one per GET). A page may be a dict (200) or a (dict, status) tuple."""
    def __init__(self, pages):
        self.pages = list(pages)
        self.calls = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append({"url": url, "headers": headers or {}, "params": params or {}, "timeout": timeout})
        data = self.pages.pop(0) if self.pages else {"news": []}
        if isinstance(data, tuple):
            return FakeResp(data[0], data[1])
        return FakeResp(data, 200)


def _client(pages, api_key="K", api_secret="S"):
    from momentum_scanner.alpaca_news import AlpacaNewsClient
    return AlpacaNewsClient(api_key=api_key, api_secret=api_secret, session=FakeSession(pages))


# ---- mapping + contract ----

def test_maps_alpaca_fields_to_newsitem():
    from momentum_scanner.finnhub_news import NewsItem
    pages = [{"news": [_raw(5, ["AAPL"], "2026-07-02T13:05:37Z", headline="H", summary="S")]}]
    out = _client(pages).fetch_company_news("aapl", "2026-07-02", "2026-07-02")
    assert len(out) == 1
    it = out[0]
    assert isinstance(it, NewsItem)
    assert it.ticker == "AAPL"            # tagged with the QUERIED symbol (not Alpaca's symbols array)
    assert it.headline == "H"
    assert it.summary == "S"
    assert it.url == "http://a/5"
    assert it.source == "benzinga"
    assert it.id == 5
    # published parsed to tz-aware UTC
    assert it.published.utcoffset().total_seconds() == 0
    assert (it.published.year, it.published.month, it.published.day, it.published.hour) == (2026, 7, 2, 13)


def test_summary_falls_back_to_content_when_summary_empty():
    # An item with an empty summary but a full content body must NOT lose the body --
    # map content into NewsItem.summary so the Gemma judge still sees the article text.
    pages = [{"news": [_raw(1, ["X"], "2026-07-02T13:00:00Z", summary="", content="Full body here")]}]
    out = _client(pages).fetch_company_news("X", "2026-07-02", "2026-07-02")
    assert out[0].summary == "Full body here"


def test_headline_only_item_is_kept():
    # Real catalysts often arrive headline-only (empty summary AND empty content).
    # They must still be returned -- the judge can classify from the headline alone.
    pages = [{"news": [_raw(2, ["X"], "2026-07-02T13:00:00Z", headline="Acme Corp Joins Example Partner Program",
                             summary="", content="")]}]
    out = _client(pages).fetch_company_news("X", "2026-07-02", "2026-07-02")
    assert len(out) == 1 and out[0].headline == "Acme Corp Joins Example Partner Program" and out[0].summary == ""


def test_drops_item_with_no_headline_and_no_text():
    pages = [{"news": [
        _raw(3, ["X"], "2026-07-02T13:00:00Z", headline="", summary="", content=""),   # nothing -> drop
        _raw(4, ["X"], "not-a-date", headline="H"),                                     # bad ts -> drop
        _raw(5, ["X"], "2026-07-02T13:00:00Z", headline="Keep"),                        # valid
    ]}]
    out = _client(pages).fetch_company_news("X", "2026-07-02", "2026-07-02")
    assert [i.id for i in out] == [5]


def test_dedup_by_id():
    pages = [{"news": [_raw(9, ["X"], "2026-07-02T13:00:00Z"),
                       _raw(9, ["X"], "2026-07-02T14:00:00Z")]}]
    out = _client(pages).fetch_company_news("X", "2026-07-02", "2026-07-02")
    assert len(out) == 1


# ---- query params + window ----

def test_query_params_symbols_window_and_include_content():
    c = _client([{"news": []}])
    c.fetch_company_news("aapl", "2026-07-02", "2026-07-02")
    p = c._session.calls[0]["params"]
    assert p["symbols"] == "AAPL"
    assert p["start"] == "2026-07-02T00:00:00Z"
    assert p["end"] == "2026-07-03T00:00:00Z"          # to_date + 1 day (judge re-trims the ET window)
    assert str(p["include_content"]).lower() == "true"
    assert c._session.calls[0]["timeout"] == 15         # explicit network timeout (never hang the run)


# ---- auth + fail-closed ----

def test_secret_sent_as_header_never_in_url_or_params():
    c = _client([{"news": [_raw(1, ["A"], "2026-07-02T13:00:00Z")]}], api_key="KEYID", api_secret=AUTH_FIXTURE)
    c.fetch_company_news("A", "2026-07-02", "2026-07-02")
    call = c._session.calls[0]
    assert call["headers"].get("APCA-API-KEY-ID") == "KEYID"
    assert call["headers"].get("APCA-API-SECRET-KEY") == AUTH_FIXTURE
    blob = str(call["params"]) + call["url"]
    assert AUTH_FIXTURE not in blob


def test_401_escalates_without_leaking_secret():
    from momentum_scanner.errors import ScannerNetworkError
    c = _client([({"news": []}, 401)], api_secret=AUTH_FIXTURE)
    with pytest.raises(ScannerNetworkError) as e:
        c.fetch_company_news("A", "2026-07-02", "2026-07-02")
    assert AUTH_FIXTURE not in str(e.value)


def test_429_escalates():
    from momentum_scanner.errors import ScannerNetworkError
    c = _client([({"news": []}, 429)])
    with pytest.raises(ScannerNetworkError):
        c.fetch_company_news("A", "2026-07-02", "2026-07-02")


def test_connection_error_escalates():
    import requests
    from momentum_scanner.alpaca_news import AlpacaNewsClient
    from momentum_scanner.errors import ScannerNetworkError

    class Boom:
        def get(self, *a, **k):
            raise requests.ConnectionError("alpaca down")
    with pytest.raises(ScannerNetworkError):
        AlpacaNewsClient(api_key="K", api_secret="S", session=Boom()).fetch_company_news(
            "A", "2026-07-02", "2026-07-02")


def test_missing_credentials_escalates(monkeypatch):
    from momentum_scanner.alpaca_news import AlpacaNewsClient
    from momentum_scanner.errors import ScannerNetworkError
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    c = AlpacaNewsClient(session=FakeSession([{"news": []}]))   # creds=None -> env (absent)
    with pytest.raises(ScannerNetworkError):
        c.fetch_company_news("A", "2026-07-02", "2026-07-02")


# ---- pagination ----

def test_pagination_follows_next_page_token():
    pages = [
        {"news": [_raw(1, ["X"], "2026-07-02T13:00:00Z")], "next_page_token": "tok2"},
        {"news": [_raw(2, ["X"], "2026-07-02T12:00:00Z")], "next_page_token": None},
    ]
    c = _client(pages)
    out = c.fetch_company_news("X", "2026-07-02", "2026-07-02")
    assert {i.id for i in out} == {1, 2}
    assert c._session.calls[1]["params"].get("page_token") == "tok2"   # 2nd call passed the token


def test_pagination_respects_max_pages_cap():
    # every page returns a token -> the client must stop at max_pages, not loop forever
    from momentum_scanner.alpaca_news import AlpacaNewsClient
    pages = [{"news": [_raw(i, ["X"], "2026-07-02T13:00:00Z")], "next_page_token": f"t{i}"} for i in range(10)]
    c = AlpacaNewsClient(api_key="K", api_secret="S", session=FakeSession(pages), max_pages=3)
    c.fetch_company_news("X", "2026-07-02", "2026-07-02")
    assert len(c._session.calls) == 3


# ---- FallbackNewsClient ----

class _Stub:
    def __init__(self, result=None, error=None):
        self.result = result or []
        self.error = error
        self.called = 0

    def fetch_company_news(self, ticker, frm, to):
        self.called += 1
        if self.error is not None:
            raise self.error
        return self.result


def test_fallback_returns_primary_when_ok():
    from momentum_scanner.alpaca_news import FallbackNewsClient
    prim = _Stub(result=["PRIMARY"])
    sec = _Stub(result=["SECONDARY"])
    out = FallbackNewsClient(prim, sec).fetch_company_news("X", "d", "d")
    assert out == ["PRIMARY"] and sec.called == 0


def test_fallback_uses_secondary_on_primary_network_error():
    from momentum_scanner.alpaca_news import FallbackNewsClient
    from momentum_scanner.errors import ScannerNetworkError
    prim = _Stub(error=ScannerNetworkError("alpaca 500"))
    sec = _Stub(result=["SECONDARY"])
    out = FallbackNewsClient(prim, sec).fetch_company_news("X", "d", "d")
    assert out == ["SECONDARY"] and sec.called == 1


def test_fallback_does_not_fallback_on_empty_primary():
    # empty (non-error) is a valid 'no news' -- do not double-fetch the secondary
    from momentum_scanner.alpaca_news import FallbackNewsClient
    prim = _Stub(result=[])
    sec = _Stub(result=["SECONDARY"])
    out = FallbackNewsClient(prim, sec).fetch_company_news("X", "d", "d")
    assert out == [] and sec.called == 0


def test_fallback_propagates_when_both_error():
    from momentum_scanner.alpaca_news import FallbackNewsClient
    from momentum_scanner.errors import ScannerNetworkError
    prim = _Stub(error=ScannerNetworkError("alpaca down"))
    sec = _Stub(error=ScannerNetworkError("finnhub down"))
    with pytest.raises(ScannerNetworkError):
        FallbackNewsClient(prim, sec).fetch_company_news("X", "d", "d")


# ---- malformed 200 bodies must be SYSTEMIC: a bad JSON / non-object body must raise
# ScannerNetworkError so it takes the Finnhub fallback + fail-closed path, not a raw exception. ----

def test_malformed_json_body_escalates():
    from momentum_scanner.alpaca_news import AlpacaNewsClient
    from momentum_scanner.errors import ScannerNetworkError

    class BadJson:
        status_code = 200

        def json(self):
            raise ValueError("not json")

    class Sess:
        def get(self, *a, **k):
            return BadJson()

    c = AlpacaNewsClient(api_key="K", api_secret="S", session=Sess())
    with pytest.raises(ScannerNetworkError):
        c.fetch_company_news("X", "2026-07-02", "2026-07-02")


def test_non_object_body_escalates():
    from momentum_scanner.errors import ScannerNetworkError
    c = _client([[]])   # HTTP 200 but body is a JSON array, not the expected {"news": [...]} object
    with pytest.raises(ScannerNetworkError):
        c.fetch_company_news("X", "2026-07-02", "2026-07-02")


def test_empty_object_body_is_valid_no_news():
    c = _client([{}])   # {} is a well-formed (if empty) object -> no news, NOT an error
    assert c.fetch_company_news("X", "2026-07-02", "2026-07-02") == []


def test_null_json_body_escalates():
    # A JSON null body is malformed, not "no news" -- same systemic path as the Finnhub client.
    from momentum_scanner.errors import ScannerNetworkError
    c = _client([None])
    with pytest.raises(ScannerNetworkError):
        c.fetch_company_news("X", "2026-07-02", "2026-07-02")


@pytest.mark.parametrize("news", [None, {"id": 1}, "synthetic text", 7])
def test_news_collection_not_a_list_escalates(news):
    from momentum_scanner.errors import ScannerNetworkError
    c = _client([{"news": news}])
    with pytest.raises(ScannerNetworkError):
        c.fetch_company_news("X", "2026-07-02", "2026-07-02")


@pytest.mark.parametrize("member", [None, "synthetic text", 7, ["nested"]])
def test_news_collection_with_non_object_member_escalates(member):
    from momentum_scanner.errors import ScannerNetworkError
    c = _client([{"news": [_raw(1, ["X"], "2026-07-02T13:00:00Z"), member]}])
    with pytest.raises(ScannerNetworkError):
        c.fetch_company_news("X", "2026-07-02", "2026-07-02")


def test_malformed_later_page_escalates():
    # A malformed page after a good one is still systemic (no partial result is returned).
    from momentum_scanner.errors import ScannerNetworkError
    pages = [{"news": [_raw(1, ["X"], "2026-07-02T13:00:00Z")], "next_page_token": "tok2"}, None]
    c = _client(pages)
    with pytest.raises(ScannerNetworkError):
        c.fetch_company_news("X", "2026-07-02", "2026-07-02")


# ---- malformed field TYPES are dropped items, never an AttributeError out of the client ----

@pytest.mark.parametrize("bad", [
    {"headline": 5}, {"headline": ["a"]}, {"summary": 7.5}, {"content": {"x": 1}},
    {"created_at": 12345}, {"created_at": ["2026-07-02T13:00:00Z"]},
])
def test_non_string_fields_drop_the_item_without_raising(bad):
    broken = _raw(60, ["X"], "2026-07-02T13:00:00Z")
    broken.update(bad)
    ok = _raw(61, ["X"], "2026-07-02T13:00:00Z")
    out = _client([{"news": [broken, ok]}]).fetch_company_news("X", "2026-07-02", "2026-07-02")
    assert [i.id for i in out] == [61]


def test_non_string_url_source_and_unhashable_id_do_not_break_the_item():
    odd = _raw(70, ["X"], "2026-07-02T13:00:00Z")
    odd.update({"url": 5, "source": ["x"], "id": ["not", "hashable"]})
    out = _client([{"news": [odd]}]).fetch_company_news("X", "2026-07-02", "2026-07-02")
    assert len(out) == 1
    assert out[0].url == "" and out[0].source == "benzinga" and out[0].id is None

"""Tests for edgar_prejudge.EdgarContext: shared per-cycle CIK-map and filing-detail
caching, the budgeted presence/dilution read, and the dilution lookback window.
"""
from datetime import datetime, timedelta, timezone

from momentum_scanner.edgar_prejudge import EdgarContext
from momentum_scanner.errors import ScannerNetworkError
from momentum_scanner.edgar_rss import FilingDetail

# Synthetic SEC CIKs (10-digit zero-padded identifiers).
CIK_1 = "1".zfill(10)

REF_NOW = datetime(2026, 7, 2, tzinfo=timezone.utc)


def _ciks(mapping):
    return lambda: mapping


def _detail(recent_filings, cik=CIK_1):
    return FilingDetail(cik, set(), None, recent_filings)


def _row(form, days_ago, items=None):
    return {
        "form": form,
        "items": set(items or []),
        "filing_date": (REF_NOW - timedelta(days=days_ago)).date(),
        "accession": f"{CIK_1}-26-000001",
        "primary_doc": "doc.htm",
    }


def _fetch_detail_returning(detail):
    calls = []

    def _fetch(cik, **kwargs):
        calls.append(cik)
        return detail

    _fetch.calls = calls
    return _fetch


def test_presence_true_no_dilution_for_recent_8k():
    detail = _detail([_row("8-K", 2)])
    ctx = EdgarContext(
        ticker_to_ciks=_ciks({"AAA": {"1"}}),
        fetch_detail=_fetch_detail_returning(detail),
        now=REF_NOW,
    )
    assert ctx.presence_and_dilution("AAA") == (True, False, "ok")


def test_form_dilution_true_for_offering_form_within_14_days():
    detail = _detail([_row("424B5", 5)])
    ctx = EdgarContext(
        ticker_to_ciks=_ciks({"AAA": {"1"}}),
        fetch_detail=_fetch_detail_returning(detail),
        now=REF_NOW,
    )
    assert ctx.presence_and_dilution("AAA") == (True, True, "ok")


def test_200_day_old_s1_presence_true_form_dilution_false():
    detail = _detail([_row("S-1", 200)])
    ctx = EdgarContext(
        ticker_to_ciks=_ciks({"AAA": {"1"}}),
        fetch_detail=_fetch_detail_returning(detail),
        now=REF_NOW,
    )
    assert ctx.presence_and_dilution("AAA") == (True, False, "ok")


def test_unresolvable_zero_ciks():
    fetch = _fetch_detail_returning(_detail([_row("8-K", 1)]))
    ctx = EdgarContext(ticker_to_ciks=_ciks({}), fetch_detail=fetch, now=REF_NOW)
    assert ctx.presence_and_dilution("AAA") == (False, False, "ok")
    assert ctx.detail_for("BBB") == ("unresolvable", None, None)
    assert fetch.calls == []


def test_ticker_to_ciks_raising_yields_error_status():
    def boom():
        raise ScannerNetworkError("edgar down")

    fetch = _fetch_detail_returning(_detail([_row("8-K", 1)]))
    ctx = EdgarContext(ticker_to_ciks=boom, fetch_detail=fetch, now=REF_NOW)
    assert ctx.presence_and_dilution("AAA") == (False, False, "error")
    assert ctx.detail_for("BBB") == ("error", None, None)
    assert fetch.calls == []


def test_fetch_detail_raising_yields_error_status_with_cik():
    def raising_fetch(cik, **kwargs):
        raise ScannerNetworkError("submissions fetch failed")

    ctx = EdgarContext(
        ticker_to_ciks=_ciks({"AAA": {"1"}}),
        fetch_detail=raising_fetch,
        now=REF_NOW,
    )
    assert ctx.presence_and_dilution("AAA") == (False, False, "error")


def test_budget_exhaustion_returns_rate_limited_without_fetch():
    detail = _detail([_row("8-K", 1)])
    fetch = _fetch_detail_returning(detail)
    ctx = EdgarContext(
        ticker_to_ciks=_ciks({"AAA": {"1"}, "BBB": {"2"}}),
        fetch_detail=fetch,
        budget=1,
        now=REF_NOW,
    )
    ctx.presence_and_dilution("AAA")  # consumes the 1 budget unit
    assert ctx.presence_and_dilution("BBB") == (False, False, "rate_limited")
    assert fetch.calls == ["1"]  # BBB never fetched


def test_cache_dedups_repeat_ticker_one_fetch():
    detail = _detail([_row("8-K", 1)])
    fetch = _fetch_detail_returning(detail)
    ctx = EdgarContext(
        ticker_to_ciks=_ciks({"AAA": {"1"}}),
        fetch_detail=fetch,
        now=REF_NOW,
    )
    ctx.presence_and_dilution("AAA")
    ctx.presence_and_dilution("AAA")
    assert fetch.calls == ["1"]


def test_shared_cache_dedup_across_presence_and_detail_for():
    detail = _detail([_row("8-K", 1)])
    fetch = _fetch_detail_returning(detail)
    cmap_calls = []

    def ticker_to_ciks():
        cmap_calls.append(1)
        return {"AAA": {"1"}}

    ctx = EdgarContext(ticker_to_ciks=ticker_to_ciks, fetch_detail=fetch, now=REF_NOW)

    ctx.presence_and_dilution("AAA")
    ctx.detail_for("AAA")

    assert len(cmap_calls) == 1
    assert fetch.calls == ["1"]


def test_rate_limited_verdict_recovers_once_detail_resolved_via_detail_for():
    # A cached "rate_limited" verdict must not be
    # replayed forever once the ticker's detail becomes available for free via
    # the unbudgeted detail_for() path (e.g. the judge thread reaching the same
    # ticker later in the cycle).
    detail = _detail([_row("8-K", 1)])
    fetch = _fetch_detail_returning(detail)
    ctx = EdgarContext(
        ticker_to_ciks=_ciks({"AAA": {"1"}}),
        fetch_detail=fetch,
        budget=0,  # exhausted from the start
        now=REF_NOW,
    )
    assert ctx.presence_and_dilution("AAA") == (False, False, "rate_limited")
    assert fetch.calls == []

    edgar_state, _, _ = ctx.detail_for("AAA")
    assert edgar_state == "ok"
    assert fetch.calls == ["1"]

    assert ctx.presence_and_dilution("AAA") == (True, False, "ok")
    assert fetch.calls == ["1"]  # still just the one fetch, made by detail_for


def test_detail_for_not_budget_limited():
    detail = _detail([_row("8-K", 1)])
    fetch = _fetch_detail_returning(detail)
    ctx = EdgarContext(
        ticker_to_ciks=_ciks({"AAA": {"1"}, "BBB": {"2"}}),
        fetch_detail=fetch,
        budget=0,
        now=REF_NOW,
    )
    # detail_for is never budget-limited, even with budget exhausted from the start.
    edgar_state, resolved_detail, cik = ctx.detail_for("AAA")
    assert edgar_state == "ok"
    assert cik == "1"
    assert fetch.calls == ["1"]

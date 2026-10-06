# tests/test_movers_scan_run.py
import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from momentum_scanner import movers_scan
from momentum_scanner.adapters.schwab_quotes import QuoteSnapshot
from momentum_scanner.errors import ScannerNetworkError
from momentum_scanner.catalyst_evaluator import CatalystVerdict

ET = ZoneInfo("America/New_York")
NOW_ET = datetime(2026, 6, 29, 13, 0, tzinfo=ET)
NOW_UTC = NOW_ET.astimezone(timezone.utc)
NOW_PM = datetime(2026, 6, 29, 8, 30, tzinfo=ET)
NOW_PM_UTC = NOW_PM.astimezone(timezone.utc)
PROCEED = CatalystVerdict("PROCEED", "real", "breakout", 0.8, "llm")


def _quote(ticker, last, prev, vol, high, avg, qt=NOW_UTC):
    return QuoteSnapshot(ticker, last, prev, vol, high, avg, qt, "regular")


def _patch_quotes(monkeypatch, quotes):
    monkeypatch.setattr(movers_scan, "batch_quotes", lambda symbols, **k: quotes)


def test_end_to_end_ranked_judged_watchlist(monkeypatch, tmp_path):
    quotes = {
        "AAA": _quote("AAA", 5.0, 4.0, 1_500_000, 5.0, 500_000),     # gap .25 in
        "BBB": _quote("BBB", 6.0, 4.0, 2_000_000, 6.0, 500_000),     # gap .50 ranks first
        "DUD": _quote("DUD", 4.05, 4.0, 1_500_000, 4.1, 500_000),    # gap .0125 -> out
    }
    _patch_quotes(monkeypatch, quotes)
    out = tmp_path / "watchlist.json"
    d = movers_scan.run_movers_scan(
        ["AAA", "BBB", "DUD"], token_fn=lambda: "t", session=object(),
        float_cache={}, float_fetch_fn=lambda t: 4_000_000, now_utc=NOW_UTC, now_et=NOW_ET,
        premarket=False, verdict_lookup=lambda c: PROCEED, out_path=str(out), scan_id="sid")
    assert d["status"] == "ok"
    assert [c["ticker"] for c in d["candidates"]] == ["BBB", "AAA"]
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written == d                                    # file == return (build-once)
    assert written["scan_id"] == "sid"


def test_scan_network_error_writes_error_status(monkeypatch, tmp_path):
    def boom(symbols, **k):
        raise ScannerNetworkError("chunk failed")
    monkeypatch.setattr(movers_scan, "batch_quotes", boom)
    out = tmp_path / "watchlist.json"
    d = movers_scan.run_movers_scan(
        ["AAA"], token_fn=lambda: "t", session=object(), float_cache={},
        float_fetch_fn=lambda t: 4_000_000, now_utc=NOW_UTC, now_et=NOW_ET, premarket=False,
        out_path=str(out), scan_id="sid")
    assert d["status"] == "error" and d["candidates"] == [] and "error" in d
    assert json.loads(out.read_text(encoding="utf-8")) == d   # file == return on the error path too


def test_no_verdict_lookup_writes_empty_ok(monkeypatch, tmp_path):
    # unjudged -> absent -> empty ok watchlist (NOT a tradable REJECT row).
    _patch_quotes(monkeypatch, {"AAA": _quote("AAA", 5.0, 4.0, 1_500_000, 5.0, 500_000)})
    out = tmp_path / "watchlist.json"
    d = movers_scan.run_movers_scan(
        ["AAA"], token_fn=lambda: "t", session=object(), float_cache={},
        float_fetch_fn=lambda t: 4_000_000, now_utc=NOW_UTC, now_et=NOW_ET, premarket=False,
        out_path=str(out))
    assert d["status"] == "ok" and d["candidates"] == []


def test_no_verdict_lookup_writes_empty_ok_file_match(monkeypatch, tmp_path):
    # unjudged -> absent -> empty ok watchlist (NOT a tradable REJECT row).
    _patch_quotes(monkeypatch, {"AAA": _quote("AAA", 5.0, 4.0, 1_500_000, 5.0, 500_000)})
    out = tmp_path / "watchlist.json"
    d = movers_scan.run_movers_scan(
        ["AAA"], token_fn=lambda: "t", session=object(), float_cache={},
        float_fetch_fn=lambda t: 4_000_000, now_utc=NOW_UTC, now_et=NOW_ET, premarket=False,
        out_path=str(out))
    assert d["status"] == "ok" and d["candidates"] == []
    assert json.loads(out.read_text(encoding="utf-8")) == d


def test_per_candidate_judge_error_isolated(monkeypatch, tmp_path):
    quotes = {
        "AAA": _quote("AAA", 5.0, 4.0, 1_500_000, 5.0, 500_000),
        "BBB": _quote("BBB", 6.0, 4.0, 2_000_000, 6.0, 500_000),
    }
    _patch_quotes(monkeypatch, quotes)
    def lookup(c):
        if c.ticker == "AAA":
            raise RuntimeError("judge boom")
        return PROCEED
    d = movers_scan.run_movers_scan(
        ["AAA", "BBB"], token_fn=lambda: "t", session=object(), float_cache={},
        float_fetch_fn=lambda t: 4_000_000, now_utc=NOW_UTC, now_et=NOW_ET, premarket=False,
        verdict_lookup=lookup)
    # the scan did not abort, and the erroring candidate (AAA) is absent; only judged BBB remains.
    assert d["status"] == "ok" and [c["ticker"] for c in d["candidates"]] == ["BBB"]


def test_end_to_end_premarket_mode(monkeypatch, tmp_path):
    # premarket: 100k floor, 5.0 relvol, no HOD requirement.
    quotes = {"PM": _quote("PM", 5.0, 4.0, 1_000_000, 4.5, 200_000, qt=NOW_PM_UTC)}  # relvol ~6.1, gap .25
    _patch_quotes(monkeypatch, quotes)
    d = movers_scan.run_movers_scan(
        ["PM"], token_fn=lambda: "t", session=object(), float_cache={},
        float_fetch_fn=lambda t: 4_000_000, now_utc=NOW_PM_UTC, now_et=NOW_PM,
        premarket=True, verdict_lookup=lambda c: PROCEED)
    assert d["status"] == "ok" and d["candidates"][0]["ticker"] == "PM"
    assert d["candidates"][0]["new_hod"] is None and d["candidates"][0]["scan_mode"] == "premarket"

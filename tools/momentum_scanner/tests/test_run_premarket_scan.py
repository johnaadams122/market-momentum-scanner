import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from momentum_scanner.catalyst_evaluator import CatalystVerdict
from momentum_scanner.errors import ScannerNetworkError

FIXED = datetime(2026, 6, 27, 13, 0, tzinfo=timezone.utc)
PROCEED = CatalystVerdict("PROCEED", "real", "FDA", 0.9, "llm")
REJECT = CatalystVerdict("REJECT", "pump", "offering", None, "rule")


@dataclass
class _Cand:
    ticker: str; company_name: str = "X Inc"; premarket_price: float = 5.4
    prev_close: float = 4.1; gap_pct: float = 0.31; float_shares: int = 8_000_000
    premarket_volume: int = 9_000_000; baseline_volume: float = 1_000_000; rel_volume: float = 9.0


@dataclass
class _Filing:
    ticker: str; headline: str


def _run_main(argv, filings, candidates, verdict=REJECT):
    with patch("momentum_scanner.run_premarket_scan._load_ticker_map", return_value={}), \
         patch("momentum_scanner.run_premarket_scan.fetch_recent_8k_filings", return_value=filings), \
         patch("momentum_scanner.run_premarket_scan.scan_candidates", return_value=candidates), \
         patch("momentum_scanner.run_premarket_scan.evaluate", return_value=verdict), \
         patch.object(sys, "argv", argv):
        from momentum_scanner.run_premarket_scan import main
        main()


# ---- build_watchlist_dict (v2) ----

def test_build_watchlist_v2_shape_and_verdict():
    from momentum_scanner.run_premarket_scan import build_watchlist_dict
    d = build_watchlist_dict([_Cand("ABCD")], [_Filing("ABCD", "8-K - FDA approval")],
                             {"ABCD": PROCEED}, scan_id="sid", scanned_at=FIXED)
    assert d["schema_version"] == 2 and d["scan_id"] == "sid" and d["status"] == "ok"
    assert d["expires_at"] > d["scanned_at"]
    c = d["candidates"][0]
    assert c["ticker"] == "ABCD" and c["catalyst"] == "8-K - FDA approval"
    assert c["gate_decision"] == "PROCEED" and c["catalyst_label"] == "real" and c["catalyst_confidence"] == 0.9


def test_build_watchlist_no_verdict_defaults_reject():
    from momentum_scanner.run_premarket_scan import build_watchlist_dict
    d = build_watchlist_dict([_Cand("ZZZZ")], [], {}, scan_id="s", scanned_at=FIXED)
    c = d["candidates"][0]
    assert c["catalyst"] is None
    assert c["gate_decision"] == "REJECT" and c["catalyst_label"] == "no-data"


def test_build_watchlist_empty_candidates():
    from momentum_scanner.run_premarket_scan import build_watchlist_dict
    d = build_watchlist_dict([], [], {}, scan_id="s", scanned_at=FIXED)
    assert d["status"] == "ok" and d["candidates"] == []


def test_build_watchlist_confirm_overflow_defaults_zero():
    """The artifact always carries confirm_overflow so the consumer can trust its absence."""
    from momentum_scanner.run_premarket_scan import build_watchlist_dict
    d = build_watchlist_dict([], [], {}, scan_id="s", scanned_at=FIXED)
    assert d["confirm_overflow"] == 0


def test_build_watchlist_surfaces_confirm_overflow():
    from momentum_scanner.run_premarket_scan import build_watchlist_dict
    d = build_watchlist_dict([_Cand("ABCD")], [], {"ABCD": PROCEED},
                             scan_id="s", scanned_at=FIXED, confirm_overflow=3)
    assert d["confirm_overflow"] == 3


def test_error_artifact_shape():
    from momentum_scanner.run_premarket_scan import build_watchlist_dict
    d = build_watchlist_dict([], [], {}, scan_id="s", scanned_at=FIXED, status="error")
    assert d["status"] == "error" and d["candidates"] == [] and d["schema_version"] == 2


def test_build_watchlist_pins_real_scanresult_contract():
    from momentum_scanner.run_premarket_scan import build_watchlist_dict
    from momentum_scanner.gap_scanner import ScanResult
    sr = ScanResult(ticker="ABCD", company_name="Abcd Inc", premarket_price=5.4, prev_close=4.1,
                    gap_pct=0.31, float_shares=8_000_000, premarket_volume=9_000_000,
                    baseline_volume=1_000_000, rel_volume=9.0)
    d = build_watchlist_dict([sr], [_Filing("ABCD", "8-K - news")], {"ABCD": PROCEED}, scan_id="s", scanned_at=FIXED)
    c = d["candidates"][0]
    for key in ("ticker", "gap_pct", "premarket_price", "float_shares", "rel_volume", "catalyst",
                "gate_decision", "catalyst_label", "gate_reason", "catalyst_confidence"):
        assert key in c, f"field '{key}' missing"
    # Legacy yfinance-only volume fields are gone; the new explicit fields are present.
    assert "avg_volume" not in c and "current_volume" not in c
    assert "premarket_volume" in c and "baseline_volume" in c


# ---- write_watchlist_json (atomic; v2 signature) ----

@dataclass
class _DatedFiling:
    ticker: str; headline: str; filed_at: datetime


@dataclass
class _News:
    ticker: str
    headline: str = "PR headline"
    published: datetime = FIXED


def test_finnhub_live_gate_blocks_paper_only_source(tmp_path, monkeypatch):
    # Free Finnhub is non-commercial -> --source finnhub in live mode -> status:error, before any call.
    import momentum_scanner.run_premarket_scan as r
    out = tmp_path / "w.json"
    monkeypatch.setenv("SCANNER_MODE", "live")
    monkeypatch.setattr(r.config, "FINNHUB_COMMERCIAL_USE", False)
    monkeypatch.setattr(sys, "argv", ["p", "--source", "finnhub", "--json-out", str(out)])
    with pytest.raises(SystemExit) as e:
        r.main()
    assert e.value.code != 0
    assert json.loads(out.read_text(encoding="utf-8"))["status"] == "error"


def test_finnhub_source_error_writes_error_watchlist(tmp_path, monkeypatch):
    import momentum_scanner.run_premarket_scan as r
    import momentum_scanner.finnhub_news as fn
    out = tmp_path / "w.json"
    monkeypatch.delenv("SCANNER_MODE", raising=False)

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def fetch_trigger_news(self, now_et):
            raise ScannerNetworkError("finnhub down")
    monkeypatch.setattr(fn, "FinnhubClient", FakeClient)
    monkeypatch.setattr(sys, "argv", ["p", "--source", "finnhub", "--json-out", str(out)])
    with pytest.raises(SystemExit):
        r.main()
    assert json.loads(out.read_text(encoding="utf-8"))["status"] == "error"


def test_finnhub_happy_path_v2_and_hard_dispatch(tmp_path, monkeypatch):
    import momentum_scanner.run_premarket_scan as r
    import momentum_scanner.finnhub_news as fn
    import momentum_scanner.catalyst_evaluator as ce
    out = tmp_path / "w.json"
    monkeypatch.delenv("SCANNER_MODE", raising=False)
    ev_called, dj_called = [], []
    monkeypatch.setattr(r, "evaluate", lambda *a, **k: ev_called.append(1))          # filing path
    monkeypatch.setattr(ce, "default_judge", lambda *a, **k: dj_called.append(1))

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def fetch_trigger_news(self, now_et):
            return ["t"]
    monkeypatch.setattr(fn, "FinnhubClient", FakeClient)
    monkeypatch.setattr(fn, "aggregate_news", lambda trig, c, now, **k: {"ABCD": [_News("ABCD")]})
    seen = {}

    def fake_scan(tickers, **k):
        seen["now_et"] = k.get("now_et")
        return [_Cand("ABCD")]
    monkeypatch.setattr(r, "scan_candidates", fake_scan)
    monkeypatch.setattr(ce, "evaluate_news", lambda c, items, **k: PROCEED)
    monkeypatch.setattr(sys, "argv", ["p", "--source", "finnhub", "--json-out", str(out)])
    r.main()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["status"] == "ok"
    assert data["candidates"][0]["ticker"] == "ABCD" and data["candidates"][0]["gate_decision"] == "PROCEED"
    assert data["candidates"][0]["catalyst"] == "PR headline"
    assert seen["now_et"] is not None                  # now_et keyword passed
    assert not ev_called and not dj_called             # finnhub never touches the filing path
    assert data["confirm_overflow"] == 0               # no overflow on the happy path


def test_finnhub_overflow_surfaced_in_watchlist(tmp_path, monkeypatch):
    """When aggregate_news drops tickers at the confirm cap, the count reaches the artifact."""
    import momentum_scanner.run_premarket_scan as r
    import momentum_scanner.finnhub_news as fn
    import momentum_scanner.catalyst_evaluator as ce
    out = tmp_path / "w.json"
    monkeypatch.delenv("SCANNER_MODE", raising=False)

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def fetch_trigger_news(self, now_et):
            return ["t"]
    monkeypatch.setattr(fn, "FinnhubClient", FakeClient)

    def fake_agg(trig, c, now, **k):
        st = k.get("stats")
        if st is not None:
            st["confirm_overflow"] = 4
        return {"ABCD": [_News("ABCD")]}
    monkeypatch.setattr(fn, "aggregate_news", fake_agg)
    monkeypatch.setattr(r, "scan_candidates", lambda tickers, **k: [_Cand("ABCD")])
    monkeypatch.setattr(ce, "evaluate_news", lambda c, items, **k: PROCEED)
    monkeypatch.setattr(sys, "argv", ["p", "--source", "finnhub", "--json-out", str(out)])
    r.main()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["confirm_overflow"] == 4


def test_main_writes_error_watchlist_on_schwab_failure(tmp_path, monkeypatch):
    # A Schwab auth/import failure (ScannerNetworkError from scan_candidates) must
    # produce a status:"error" watchlist + nonzero exit, just like an EDGAR outage.
    import momentum_scanner.run_premarket_scan as r
    out = tmp_path / "watchlist.json"
    monkeypatch.setattr(r, "_load_ticker_map", lambda: {})
    monkeypatch.setattr(r, "fetch_recent_8k_filings", lambda **k: [_DatedFiling("ABCD", "8-K", FIXED)])

    def boom(tickers, **k):
        raise ScannerNetworkError("schwab down")
    monkeypatch.setattr(r, "scan_candidates", boom)
    monkeypatch.setattr(sys, "argv", ["prog", "--json-out", str(out)])
    with pytest.raises(SystemExit) as e:
        r.main()
    assert e.value.code != 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["status"] == "error" and data["candidates"] == []


def test_main_unexpected_error_preserves_prior_watchlist(tmp_path, monkeypatch):
    # An UNEXPECTED (non-ScannerNetworkError) failure must crash WITHOUT overwriting the
    # last-good watchlist with an error artifact (crash-keeps-last-good).
    import momentum_scanner.run_premarket_scan as r
    out = tmp_path / "watchlist.json"
    out.write_text('{"schema_version":2,"status":"ok","candidates":[]}', encoding="utf-8")
    monkeypatch.setattr(r, "_load_ticker_map", lambda: {})
    monkeypatch.setattr(r, "fetch_recent_8k_filings", lambda **k: [_DatedFiling("ABCD", "8-K", FIXED)])

    def boom(tickers, **k):
        raise RuntimeError("unexpected")
    monkeypatch.setattr(r, "scan_candidates", boom)
    monkeypatch.setattr(sys, "argv", ["prog", "--json-out", str(out)])
    with pytest.raises(RuntimeError):
        r.main()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["status"] == "ok"   # prior-good preserved, not overwritten


def test_write_watchlist_json_atomic(tmp_path):
    from momentum_scanner.run_premarket_scan import write_watchlist_json
    out = tmp_path / "sub" / "watchlist.json"
    write_watchlist_json(str(out), [_Cand("ABCD")], [_Filing("ABCD", "h")], {"ABCD": PROCEED})
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema_version"] == 2 and data["candidates"][0]["ticker"] == "ABCD"
    assert not any(p.suffix == ".tmp" for p in out.parent.iterdir())


def test_write_watchlist_json_failure_cleans_tmp_and_preserves_last_good(tmp_path):
    import momentum_scanner.run_premarket_scan as rps
    out = tmp_path / "watchlist.json"
    out.write_text('{"schema_version":2,"status":"ok","candidates":[{"ticker":"OLD"}]}', encoding="utf-8")
    with patch("momentum_scanner.run_premarket_scan.os.replace", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            rps.write_watchlist_json(str(out), [_Cand("ABCD")], [_Filing("ABCD", "h")], {"ABCD": PROCEED})
    assert not any(p.suffix == ".tmp" for p in out.parent.iterdir())
    assert json.loads(out.read_text(encoding="utf-8"))["candidates"][0]["ticker"] == "OLD"


# ---- main() integration ----

def test_main_json_out_writes_verdict(tmp_path):
    out = tmp_path / "watchlist.json"
    _run_main(["run.py", "--json-out", str(out)], [_Filing("ABCD", "8-K - news")], [_Cand("ABCD")], verdict=PROCEED)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["candidates"][0]["gate_decision"] == "PROCEED" and data["status"] == "ok"


def test_main_zero_candidates_still_writes_empty(tmp_path):
    out = tmp_path / "watchlist.json"
    _run_main(["run.py", "--json-out", str(out)], [], [])
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["status"] == "ok" and data["candidates"] == []


def test_main_without_json_out_writes_nothing(tmp_path):
    out = tmp_path / "watchlist.json"
    _run_main(["run.py"], [_Filing("ABCD", "h")], [_Cand("ABCD")])
    assert not out.exists()


def test_main_crash_preserves_last_good(tmp_path):
    out = tmp_path / "watchlist.json"
    out.write_text('{"schema_version":2,"status":"ok","candidates":[{"ticker":"OLD"}]}', encoding="utf-8")
    with patch("momentum_scanner.run_premarket_scan._load_ticker_map", return_value={}), \
         patch("momentum_scanner.run_premarket_scan.fetch_recent_8k_filings", return_value=[_Filing("X", "h")]), \
         patch("momentum_scanner.run_premarket_scan.scan_candidates", side_effect=RuntimeError("yfinance down")), \
         patch.object(sys, "argv", ["run.py", "--json-out", str(out)]):
        from momentum_scanner.run_premarket_scan import main
        with pytest.raises(RuntimeError):
            main()
    assert json.loads(out.read_text(encoding="utf-8"))["candidates"][0]["ticker"] == "OLD"


def test_main_fatal_source_writes_error_artifact(tmp_path):
    out = tmp_path / "watchlist.json"
    with patch("momentum_scanner.run_premarket_scan._load_ticker_map", return_value={}), \
         patch("momentum_scanner.run_premarket_scan.fetch_recent_8k_filings",
               side_effect=ScannerNetworkError("EDGAR down")), \
         patch.object(sys, "argv", ["run.py", "--json-out", str(out)]):
        from momentum_scanner.run_premarket_scan import main
        with pytest.raises(SystemExit):
            main()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["status"] == "error" and data["candidates"] == []


def test_main_per_candidate_evaluator_error_isolated(tmp_path):
    out = tmp_path / "watchlist.json"
    with patch("momentum_scanner.run_premarket_scan._load_ticker_map", return_value={}), \
         patch("momentum_scanner.run_premarket_scan.fetch_recent_8k_filings", return_value=[_Filing("ABCD", "h")]), \
         patch("momentum_scanner.run_premarket_scan.scan_candidates", return_value=[_Cand("ABCD")]), \
         patch("momentum_scanner.run_premarket_scan.evaluate", side_effect=RuntimeError("boom")), \
         patch.object(sys, "argv", ["run.py", "--json-out", str(out)]):
        from momentum_scanner.run_premarket_scan import main
        main()  # must NOT raise; the one candidate becomes a REJECT
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["status"] == "ok"
    assert data["candidates"][0]["gate_decision"] == "REJECT"


def test_main_dedups_to_most_recent_filing(tmp_path):
    from momentum_scanner.edgar_rss import FilingResult
    cik = "111".zfill(10)                     # SEC CIKs are 10-digit, zero-padded strings
    old = FilingResult(ticker="ABCD", company_name="A", cik=cik, headline="OLD 8-K",
                       url="http://x/" + cik + "-26-000044-index.htm",
                       filed_at=datetime(2026, 6, 27, 9, 0, tzinfo=timezone.utc))
    new = FilingResult(ticker="ABCD", company_name="A", cik=cik, headline="NEW 8-K",
                       url="http://x/" + cik + "-26-000045-index.htm",
                       filed_at=datetime(2026, 6, 27, 9, 30, tzinfo=timezone.utc))
    captured = {}
    def fake_scan(tickers, **k):
        captured["tickers"] = list(tickers); return [_Cand("ABCD")]
    def fake_eval(c, f):
        captured["filing"] = f; return PROCEED
    out = tmp_path / "watchlist.json"
    with patch("momentum_scanner.run_premarket_scan._load_ticker_map", return_value={}), \
         patch("momentum_scanner.run_premarket_scan.fetch_recent_8k_filings", return_value=[old, new]), \
         patch("momentum_scanner.run_premarket_scan.scan_candidates", side_effect=fake_scan), \
         patch("momentum_scanner.run_premarket_scan.evaluate", side_effect=fake_eval), \
         patch.object(sys, "argv", ["run.py", "--json-out", str(out)]):
        from momentum_scanner.run_premarket_scan import main
        main()
    assert captured["tickers"] == ["ABCD"]            # deduped to one
    assert captured["filing"].headline == "NEW 8-K"   # most-recent used as triggering filing
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["candidates"][0]["catalyst"] == "NEW 8-K"

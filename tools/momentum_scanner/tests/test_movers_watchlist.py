import json
from datetime import datetime, timezone
from momentum_scanner import movers_watchlist as mw
from momentum_scanner.movers_scan import MoverCandidate
from momentum_scanner.catalyst_evaluator import CatalystVerdict

FIXED = datetime(2026, 6, 29, 13, 0, tzinfo=timezone.utc)
PROCEED = CatalystVerdict("PROCEED", "real", "breakout", 0.8, "llm", headline="AAA pops on FDA")

CONTRACT_KEYS = ("ticker", "premarket_price", "gap_pct", "float_shares", "rel_volume",
                 "catalyst", "gate_decision", "catalyst_label", "gate_reason",
                 "catalyst_confidence", "catalyst_source")


def _cand(ticker):
    return MoverCandidate(ticker=ticker, company_name=ticker, premarket_price=5.0, prev_close=4.0,
                          gap_pct=0.25, float_shares=4_000_000, premarket_volume=1_500_000,
                          baseline_volume=500_000, rel_volume=5.0, day_high=5.0, new_hod=True,
                          scan_mode="intraday", last_seen_at="2026-06-29T14:00:00+00:00", score=1.25)


def test_judged_candidate_pins_v2_contract():
    d = mw.build_movers_watchlist_dict([_cand("AAA")], {"AAA": PROCEED}, scan_id="s", scanned_at=FIXED)
    assert d["schema_version"] == 2 and d["status"] == "ok" and d["confirm_overflow"] == 0
    c = d["candidates"][0]
    for key in CONTRACT_KEYS:
        assert key in c, f"missing contract key {key}"
    assert c["gate_decision"] == "PROCEED" and c["catalyst_label"] == "real"
    assert c["catalyst"] == "AAA pops on FDA"
    assert c["score"] == 1.25 and c["new_hod"] is True and c["scan_mode"] == "intraday"
    assert "avg_volume" not in c and "current_volume" not in c


def test_unjudged_candidate_is_absent_not_traded():
    # An unjudged ticker (no verdict) must NOT be written (consumers may act on every written row).
    d = mw.build_movers_watchlist_dict([_cand("JUDGED"), _cand("UNJUDGED")],
                                       {"JUDGED": PROCEED}, scan_id="s", scanned_at=FIXED)
    tickers = [c["ticker"] for c in d["candidates"]]
    assert tickers == ["JUDGED"] and d["status"] == "ok"


def test_all_unjudged_yields_empty_ok():
    d = mw.build_movers_watchlist_dict([_cand("A"), _cand("B")], {}, scan_id="s", scanned_at=FIXED)
    assert d["status"] == "ok" and d["candidates"] == []


def test_movers_error_artifact():
    d = mw.build_movers_watchlist_dict([], {}, scan_id="s", scanned_at=FIXED, status="error",
                                       error="universe incomplete")
    assert d["status"] == "error" and d["candidates"] == [] and d["error"] == "universe incomplete"


def test_write_movers_watchlist_atomic(tmp_path):
    out = tmp_path / "sub" / "watchlist.json"
    mw.write_movers_watchlist_json(str(out), [_cand("AAA")], {"AAA": PROCEED})
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema_version"] == 2 and data["candidates"][0]["ticker"] == "AAA"
    assert not any(p.suffix == ".tmp" for p in out.parent.iterdir())


def test_score_and_catalyst_score_survive_asdict_not_masked_by_setdefault():
    # watchlist_io.stamp_verdict's setdefault("catalyst_score", 0.0)
    # only fires when the KEY IS ABSENT from the dict; the existing additive-field tests call
    # stamp_verdict on a bare {}, which can't distinguish "asdict emitted the real value" from
    # "setdefault silently filled a default". Build a REAL MoverCandidate with non-default
    # score/catalyst_score and prove both survive asdict(c) into the written candidate dict.
    c = MoverCandidate(ticker="AAA", company_name="AAA", premarket_price=5.0, prev_close=4.0,
                       gap_pct=0.25, float_shares=4_000_000, premarket_volume=1_500_000,
                       baseline_volume=500_000, rel_volume=5.0, day_high=5.0, new_hod=True,
                       scan_mode="intraday", last_seen_at="2026-06-29T14:00:00+00:00",
                       score=1.5, catalyst_score=0.2)
    d = mw.build_movers_watchlist_dict([c], {"AAA": PROCEED}, scan_id="s", scanned_at=FIXED)
    cd = d["candidates"][0]
    assert cd["score"] == 1.5
    assert cd["catalyst_score"] == 0.2

# tests/test_judge_runtime.py
from datetime import datetime, timezone
from dataclasses import replace
from zoneinfo import ZoneInfo
from momentum_scanner import judge_runtime, movers_scan
from momentum_scanner.verdict_cache import VerdictCache
from momentum_scanner.adapters.schwab_quotes import QuoteSnapshot
from momentum_scanner.catalyst_evaluator import CatalystVerdict

ET = ZoneInfo("America/New_York")
NOW_ET = datetime(2026, 6, 29, 13, 0, tzinfo=ET)
NOW_UTC = NOW_ET.astimezone(timezone.utc)
PROCEED = CatalystVerdict("PROCEED", "real", "breakout", 0.8, "llm", headline="ABCD soars on FDA news")


def _quote(t, last, prev, vol, high, avg):
    return QuoteSnapshot(t, last, prev, vol, high, avg, NOW_UTC, "regular")


def _kwargs():
    return dict(token_fn=lambda: "t", session=object(), float_cache={},
                float_fetch_fn=lambda t: 4_000_000, now_utc=NOW_UTC, now_et=NOW_ET, premarket=False)


def test_make_verdict_lookup_reads_cache():
    cache = VerdictCache(); cache.put("AAA", NOW_ET, replace(PROCEED, judged_at=NOW_UTC))
    lookup = judge_runtime.make_verdict_lookup(cache, now_et=NOW_ET, premarket=False)
    class C: ticker = "AAA"
    class Z: ticker = "ZZZ"
    assert lookup(C()).decision == "PROCEED" and lookup(Z()) is None


def test_discover_then_write_from_candidates_no_rediscovery(monkeypatch, tmp_path):
    calls = {"n": 0}
    def fake_quotes(symbols, **k):
        calls["n"] += 1
        return {"AAA": _quote("AAA", 5.0, 4.0, 1_500_000, 5.0, 500_000)}
    monkeypatch.setattr(movers_scan, "batch_quotes", fake_quotes)
    cache = VerdictCache()
    lookup = judge_runtime.make_verdict_lookup(cache, now_et=NOW_ET, premarket=False)
    cands = movers_scan.discover_movers(["AAA"], **_kwargs())          # 1 discovery (1 quote call)
    assert [c.ticker for c in cands] == ["AAA"] and calls["n"] == 1
    # judge into the cache (decoupled worker)
    def judge_fn(c, *, now_et, premarket, edgar=None):
        return replace(PROCEED, judged_at=now_et.astimezone(timezone.utc)), None
    judge_runtime.make_judge_batch(cache, news_client=None, now_et=NOW_ET, premarket=False,
                                   judge_fn=judge_fn)(cands)
    # write from the SAME candidates -- NO second quote call
    out = tmp_path / "w.json"
    d = movers_scan.write_from_candidates(cands, verdict_lookup=lookup, out_path=str(out), scan_id="sid")
    assert calls["n"] == 1                                             # no re-discovery
    assert [c["ticker"] for c in d["candidates"]] == ["AAA"]
    assert d["candidates"][0]["gate_decision"] == "PROCEED"
    assert d["candidates"][0]["catalyst"] == "ABCD soars on FDA news"  # headline stamped


def test_decoupled_two_cycle_latency_via_run_movers_scan(monkeypatch, tmp_path):
    monkeypatch.setattr(movers_scan, "batch_quotes",
                        lambda symbols, **k: {"AAA": _quote("AAA", 5.0, 4.0, 1_500_000, 5.0, 500_000)})
    cache = VerdictCache()
    lookup = judge_runtime.make_verdict_lookup(cache, now_et=NOW_ET, premarket=False)
    out = tmp_path / "w.json"
    d1 = movers_scan.run_movers_scan(["AAA"], verdict_lookup=lookup, out_path=str(out), **_kwargs())
    assert d1["status"] == "ok" and d1["candidates"] == []             # cycle 1: unjudged -> absent
    cands = movers_scan.discover_movers(["AAA"], **_kwargs())
    def judge_fn(c, *, now_et, premarket, edgar=None):
        return replace(PROCEED, judged_at=now_et.astimezone(timezone.utc)), None
    judge_runtime.make_judge_batch(cache, news_client=None, now_et=NOW_ET, premarket=False,
                                   judge_fn=judge_fn)(cands)
    d2 = movers_scan.run_movers_scan(["AAA"], verdict_lookup=lookup, out_path=str(out), **_kwargs())
    assert [c["ticker"] for c in d2["candidates"]] == ["AAA"]          # cycle 2: appears

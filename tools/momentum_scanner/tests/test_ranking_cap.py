"""Score-cap test suite.

Test classes (lettered by test-name prefix):
  A  upper bound: ON, every finite stamped score <= cap; extreme rows stamp exactly the cap.
  B  OFF-identity: (i) golden-byte artifact comparison against the pre-cap bytes;
     (ii) Python-level score/effective/rank equality to 6dp + pinned wire key sets.
  C  ON values: stamped == min(E, cap); sub-cap rows unchanged exactly; provenance stamps;
     top-level envelope metadata including the EMPTY-candidate envelope.
  D  ordering invariance ON vs OFF (rank_candidates, judge_batch selection set AND order,
     stable re-sort of capped dicts) + the hump guard.
  E  config tri-state: token sets, unset, junk -> INVALID (daemon abort, no publish), value
     parsing/range, never silent-OFF, never silent-default.
  F  finite-input contract: every non-finite input case pinned.
  G  durable marker: match/mismatch/absent/malformed on the discovery side + the consumer-side
     watchlist_cap_mismatch contract.
"""
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

from momentum_scanner import config
from momentum_scanner import judge_batch
from momentum_scanner import movers_watchlist as mw
from momentum_scanner import ranking
from momentum_scanner import scanner_movers_daemon as smd
from momentum_scanner import watchlist_io
from momentum_scanner.catalyst_evaluator import CatalystVerdict
from momentum_scanner.movers_scan import MoverCandidate
from momentum_scanner.tests import cap_fixtures
from momentum_scanner.verdict_cache import VerdictCache

from momentum_scanner.tests.golden.watchlist_off_golden import GOLDEN_BYTES
CAP_FLAG = "SCANNER_SCORE_CAP"
CAP_VALUE = "SCANNER_SCORE_CAP_VALUE"
FIXED = datetime(2026, 8, 11, 13, 30, 0, tzinfo=timezone.utc)

# The full OFF wire shape, pinned (B: "key sets identical"). Any additive key leaking into the
# OFF artifact -- including a None-valued score_uncapped/score_cap_value -- breaks this set.
OFF_ROW_KEYS = {
    "ticker", "company_name", "premarket_price", "prev_close", "gap_pct", "float_shares",
    "premarket_volume", "baseline_volume", "rel_volume", "day_high", "new_hod", "scan_mode",
    "last_seen_at", "score", "edgar_signal", "catalyst_score",
    "catalyst", "gate_decision", "catalyst_label", "gate_reason", "catalyst_confidence",
    "catalyst_source", "dilution_flag", "catalyst_content_source",
}
OFF_ENVELOPE_KEYS = {"schema_version", "scan_id", "scanned_at", "expires_at", "status",
                     "confirm_overflow", "candidates"}


@pytest.fixture
def cap_off(monkeypatch):
    monkeypatch.delenv(CAP_FLAG, raising=False)
    monkeypatch.delenv(CAP_VALUE, raising=False)


@pytest.fixture
def cap_on(monkeypatch):
    monkeypatch.setenv(CAP_FLAG, "1")
    monkeypatch.delenv(CAP_VALUE, raising=False)


def _cand(ticker="AAA", gap=0.25, relvol=5.0, fs=5_000_000, edgar=0.0, **kw):
    c = MoverCandidate(ticker=ticker, company_name=ticker, premarket_price=5.0, prev_close=4.0,
                       gap_pct=gap, float_shares=fs, premarket_volume=1_500_000,
                       baseline_volume=500_000, rel_volume=relvol, day_high=5.0, new_hod=True,
                       scan_mode="intraday", last_seen_at=cap_fixtures.FIXED_SEEN_AT, **kw)
    c.edgar_signal = edgar
    return c


def _v(decision="PROCEED", label="real", conf=0.8, src="finnhub", dil=None):
    return CatalystVerdict(decision, label, "r", conf, "llm", headline="h",
                           dilution_flag=dil, content_source=src)


def _rank_cohort():
    candidates, verdicts = cap_fixtures.build_cohort()
    return ranking.rank_candidates(candidates, verdicts=verdicts), verdicts


# --- A: upper bound --------------------------------------------------------------------------

def test_a_on_extreme_rows_stamp_exactly_the_cap(cap_on):
    ranked, _ = _rank_cohort()
    by_ticker = {c.ticker: c for c in ranked}
    for t in cap_fixtures.TAIL_TICKERS:
        assert by_ticker[t].score == config.SCORE_CAP_DEFAULT, t


def test_a_on_every_finite_stamped_score_at_most_cap(cap_on):
    ranked, _ = _rank_cohort()
    for c in ranked:
        assert math.isfinite(c.score)
        assert c.score <= config.SCORE_CAP_DEFAULT


# --- B: OFF-identity -------------------------------------------------------------------------

def test_b_off_golden_byte_identity(cap_off, tmp_path):
    out = tmp_path / "watchlist.json"
    cap_fixtures.write_off_watchlist(out)
    assert out.read_bytes() == GOLDEN_BYTES


def test_b_off_python_level_identity_to_6dp(cap_off):
    ranked, verdicts = _rank_cohort()
    for c in ranked:
        base = ranking.score(c)
        bonus = ranking.catalyst_bonus(c, verdicts.get(c.ticker))
        dil = ranking.dilution_factor(getattr(verdicts.get(c.ticker), "dilution_flag", None))
        expected = round((base * ranking.float_factor(c.float_shares) + bonus) * dil, 6)
        assert c.score == expected, c.ticker
    # three rows fully manual from primitive constants (not via the module's own helpers)
    by_ticker = {c.ticker: c for c in ranked}
    assert by_ticker["T1"].score == round(round(5.0 * 400.0, 6) * 1.0
                                          + round(0.2 * 0.90 * 1.0 * 1.0, 6), 6) == 2000.18
    assert by_ticker["H8"].score == round((5.0 + round(0.2 * 0.74 * 0.7, 6)) * 0.7, 6) == 3.57252
    assert by_ticker["H9"].score == round(3.0 + round(0.2 * 0.90 * 0.7, 6), 6) == 3.126


def test_b_off_wire_key_sets_pinned(cap_off):
    ranked, verdicts = _rank_cohort()
    d = mw.build_movers_watchlist_dict(ranked, verdicts, scan_id="s", scanned_at=FIXED)
    assert set(d.keys()) == OFF_ENVELOPE_KEYS
    for row in d["candidates"]:
        assert set(row.keys()) == OFF_ROW_KEYS, row["ticker"]


# --- C: ON values ----------------------------------------------------------------------------

def test_c_on_stamped_is_min_of_e_and_cap_with_provenance(cap_on):
    ranked, verdicts = _rank_cohort()
    # recompute raw E per row via an OFF-config rank of a FRESH cohort (same deterministic build)
    off_scores = {}
    import os
    saved = os.environ.pop(CAP_FLAG)
    try:
        off_ranked, _ = _rank_cohort()
        off_scores = {c.ticker: c.score for c in off_ranked}
    finally:
        os.environ[CAP_FLAG] = saved
    for c in ranked:
        e = off_scores[c.ticker]
        assert c.score == min(e, config.SCORE_CAP_DEFAULT), c.ticker
        assert c.score_uncapped == e, c.ticker
        assert c.score_cap_value == config.SCORE_CAP_DEFAULT, c.ticker
        if e <= config.SCORE_CAP_DEFAULT:
            assert c.score == e, "sub-cap row must be unchanged exactly"


def test_c_on_wire_rows_carry_cap_keys_off_rows_do_not(cap_on, monkeypatch):
    ranked, verdicts = _rank_cohort()
    d = mw.build_movers_watchlist_dict(ranked, verdicts, scan_id="s", scanned_at=FIXED)
    for row in d["candidates"]:
        assert set(row.keys()) == OFF_ROW_KEYS | {"score_uncapped", "score_cap_value"}
        assert row["score_cap_value"] == config.SCORE_CAP_DEFAULT
    monkeypatch.delenv(CAP_FLAG)
    ranked_off, verdicts_off = _rank_cohort()
    d_off = mw.build_movers_watchlist_dict(ranked_off, verdicts_off, scan_id="s", scanned_at=FIXED)
    for row in d_off["candidates"]:
        assert "score_uncapped" not in row and "score_cap_value" not in row


def test_c_on_top_level_metadata_present_including_empty_envelope(cap_on):
    ranked, verdicts = _rank_cohort()
    d = mw.build_movers_watchlist_dict(ranked, verdicts, scan_id="s", scanned_at=FIXED)
    assert d["score_cap"] == {"active": True, "value": config.SCORE_CAP_DEFAULT}
    empty = mw.build_movers_watchlist_dict([], {}, scan_id="s", scanned_at=FIXED)
    assert empty["candidates"] == []
    assert empty["score_cap"] == {"active": True, "value": config.SCORE_CAP_DEFAULT}
    err = mw.build_movers_watchlist_dict([], {}, scan_id="s", scanned_at=FIXED,
                                         status="error", error="outage")
    assert err["score_cap"] == {"active": True, "value": config.SCORE_CAP_DEFAULT}


def test_c_off_top_level_metadata_absent(cap_off):
    empty = mw.build_movers_watchlist_dict([], {}, scan_id="s", scanned_at=FIXED)
    assert "score_cap" not in empty


def test_c_on_custom_cap_value_stamped(monkeypatch):
    monkeypatch.setenv(CAP_FLAG, "true")
    monkeypatch.setenv(CAP_VALUE, "12.5")
    c = _cand(gap=5.0, relvol=400.0)          # raw E 2000.0
    ranked = ranking.rank_candidates([c], verdicts={"AAA": _v(conf=None, src=None)})
    assert ranked[0].score == 12.5
    assert ranked[0].score_uncapped == 2000.0
    assert ranked[0].score_cap_value == 12.5
    d = mw.build_movers_watchlist_dict(ranked, {"AAA": _v(conf=None, src=None)},
                                       scan_id="s", scanned_at=FIXED)
    assert d["score_cap"] == {"active": True, "value": 12.5}


def test_c_on_e_exactly_at_cap_stamps_and_keeps_value(cap_on):
    c = _cand(gap=0.40, relvol=20.0)          # raw E exactly 8.0 with a no-bonus verdict
    ranked = ranking.rank_candidates([c], verdicts={"AAA": _v(conf=None, src=None)})
    assert ranked[0].score == 8.0
    assert ranked[0].score_uncapped == 8.0
    assert ranked[0].score_cap_value == config.SCORE_CAP_DEFAULT


# --- D: ordering invariance + hump guard -----------------------------------------------------

def test_d_rank_order_identical_on_vs_off(monkeypatch):
    monkeypatch.delenv(CAP_FLAG, raising=False)
    monkeypatch.delenv(CAP_VALUE, raising=False)
    off_ranked, _ = _rank_cohort()
    monkeypatch.setenv(CAP_FLAG, "1")
    on_ranked, _ = _rank_cohort()
    assert [c.ticker for c in on_ranked] == [c.ticker for c in off_ranked]
    assert len(on_ranked) > 20
    n_tail = sum(1 for c in on_ranked if c.ticker in cap_fixtures.TAIL_TICKERS)
    assert n_tail >= 3, "fixture must carry multiple tail rows"


def test_d_judge_batch_selection_set_and_order_identical_on_vs_off(monkeypatch):
    et_now = datetime(2026, 8, 11, 10, 0, tzinfo=timezone.utc)

    def run():
        candidates, _ = cap_fixtures.build_cohort()
        order = []

        def jf(c, *, now_et, premarket, edgar=None):
            order.append(c.ticker)
            return None, None                     # systemic -> nothing cached; order is the evidence
        judge_batch.process_judge_batch(candidates, VerdictCache(), now_et=et_now,
                                        premarket=False, judge_fn=jf)
        return order

    monkeypatch.delenv(CAP_FLAG, raising=False)
    monkeypatch.delenv(CAP_VALUE, raising=False)
    off_order = run()
    monkeypatch.setenv(CAP_FLAG, "1")
    on_order = run()
    assert on_order == off_order                  # order identical
    assert set(on_order) == set(off_order)        # selection set identical
    assert len(on_order) == config.MAX_JUDGE_PER_CYCLE   # the cap binds on this >20-row cohort


def test_d_stable_resort_of_capped_dicts_reproduces_producer_order(cap_on):
    ranked, verdicts = _rank_cohort()
    d = mw.build_movers_watchlist_dict(ranked, verdicts, scan_id="s", scanned_at=FIXED)
    rows = d["candidates"]
    tied_at_cap = [r for r in rows if r["score"] == config.SCORE_CAP_DEFAULT]
    assert len(tied_at_cap) >= 3, "re-sort test needs multiple rows tied at the cap"
    resorted = sorted(rows, key=lambda r: r["score"], reverse=True)   # stable sort: ties at the cap keep producer order
    assert [r["ticker"] for r in resorted] == [c.ticker for c in ranked]


def test_d_hump_guard_and_hump_rows_unchanged(monkeypatch):
    assert config.SCORE_CAP_MIN_ALLOWED > 7.7     # above the fixture hump band checked below
    monkeypatch.delenv(CAP_FLAG, raising=False)
    monkeypatch.delenv(CAP_VALUE, raising=False)
    off_ranked, _ = _rank_cohort()
    off_scores = {c.ticker: c.score for c in off_ranked}
    monkeypatch.setenv(CAP_FLAG, "1")
    on_ranked, _ = _rank_cohort()
    on_scores = {c.ticker: c.score for c in on_ranked}
    for t in cap_fixtures.HUMP_TICKERS:
        assert 1.0 <= off_scores[t] <= 7.7, "fixture drift: %s left the hump band" % t
        assert on_scores[t] == off_scores[t], t


# --- E: config tri-state ---------------------------------------------------------------------

def test_e_each_on_token_enables_with_default_value():
    for tok in ("1", "true", "yes", "on", " TRUE ", "On"):
        res = config.resolve_score_cap(env={CAP_FLAG: tok})
        assert res.state == "on", tok
        assert res.value == 8.0, tok


def test_e_each_off_token_and_unset_resolve_off():
    for env in ({}, {CAP_FLAG: "0"}, {CAP_FLAG: "false"}, {CAP_FLAG: "no"},
                {CAP_FLAG: "off"}, {CAP_FLAG: ""}, {CAP_FLAG: " OFF "}):
        res = config.resolve_score_cap(env=env)
        assert res.state == "off", env
        assert res.value is None


def test_e_junk_token_present_is_invalid_never_silent_off():
    for tok in ("maybe", "2", "enable", "tru"):
        res = config.resolve_score_cap(env={CAP_FLAG: tok})
        assert res.state == "invalid", tok
        assert tok.strip() in res.reason


def test_e_value_unset_defaults_to_8():
    res = config.resolve_score_cap(env={CAP_FLAG: "1"})
    assert res.state == "on" and res.value == 8.0
    assert config.SCORE_CAP_DEFAULT == 8.0
    assert config.SCORE_CAP_MIN_ALLOWED == 8.0
    assert config.SCORE_CAP_MAX_ALLOWED == 1500.0


def test_e_bad_values_are_invalid_not_silent_off_not_silent_default():
    for bad in ("0.5", "1e9", "abc", "inf", "-inf", "nan", "7.99", "1500.01", ""):
        res = config.resolve_score_cap(env={CAP_FLAG: "1", CAP_VALUE: bad})
        assert res.state == "invalid", bad
        assert res.value is None, bad
        assert bad.strip() in res.reason or bad == "", bad


def test_e_bad_value_with_flag_off_is_still_invalid():
    # Fail-closed: a present-but-garbage value aborts even when the flag says OFF -- a broken
    # launcher env is a config error, never something to run past silently.
    res = config.resolve_score_cap(env={CAP_FLAG: "0", CAP_VALUE: "abc"})
    assert res.state == "invalid"
    res = config.resolve_score_cap(env={CAP_VALUE: "abc"})
    assert res.state == "invalid"


def test_e_in_range_values_exact():
    for raw, want in (("8", 8.0), ("8.0", 8.0), ("12.25", 12.25), ("1500", 1500.0),
                      (" 9.5 ", 9.5)):
        res = config.resolve_score_cap(env={CAP_FLAG: "1", CAP_VALUE: raw})
        assert res.state == "on" and res.value == want, raw


def test_e_rank_candidates_raises_loudly_on_invalid_config(monkeypatch):
    monkeypatch.setenv(CAP_FLAG, "maybe")
    with pytest.raises(config.ScoreCapConfigError):
        ranking.rank_candidates([_cand()], verdicts={"AAA": _v()})


def test_e_writer_raises_loudly_on_invalid_config(monkeypatch):
    monkeypatch.setenv(CAP_FLAG, "maybe")
    with pytest.raises(config.ScoreCapConfigError):
        mw.build_movers_watchlist_dict([], {}, scan_id="s", scanned_at=FIXED)


def test_e_daemon_startup_abort_on_invalid_no_publish(tmp_path, monkeypatch):
    # Guard: without the startup seam, main() would fall through into the REAL daemon loop
    # (network + infinite cycle). Fail fast instead of hanging the suite.
    assert hasattr(smd, "score_cap_startup"), "score_cap_startup not implemented"
    monkeypatch.setenv(CAP_FLAG, "maybe")
    rc = smd.main(["--state-dir", str(tmp_path)])
    assert rc == 2
    assert not (tmp_path / "watchlist.json").exists()
    assert not (tmp_path / config.DAEMON_HEARTBEAT_FILE).exists()


# --- F: finite-input contract (non-finite input cases) --------------------------------------

def _publish_rejected(tmp_path, candidates, verdicts):
    """Rank + build + attempt the atomic publish; assert the 'publish rejected' outcome:
    raises, prior artifact byte-identical, no temp file left."""
    out = tmp_path / "watchlist.json"
    watchlist_io.atomic_write_json(str(out), {"prior": True})
    prior = out.read_bytes()
    ranked = ranking.rank_candidates(candidates, verdicts=verdicts)
    payload = mw.build_movers_watchlist_dict(ranked, verdicts, scan_id="s", scanned_at=FIXED)
    with pytest.raises(ValueError):
        watchlist_io.atomic_write_json(str(out), payload)
    assert out.read_bytes() == prior
    assert not [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    return ranked


@pytest.mark.parametrize("bad_gap", [float("nan"), float("inf")])
def test_f_gap_nonfinite_rejected_no_new_keys_both_states(tmp_path, monkeypatch, bad_gap):
    for state in ("off", "on"):
        if state == "on":
            monkeypatch.setenv(CAP_FLAG, "1")
        else:
            monkeypatch.delenv(CAP_FLAG, raising=False)
        monkeypatch.delenv(CAP_VALUE, raising=False)
        verdicts = {"BAD": _v(), "OK": _v()}
        cands = [_cand("BAD", gap=bad_gap), _cand("OK", gap=0.25)]
        sub = tmp_path / state
        sub.mkdir()
        ranked = _publish_rejected(sub, cands, verdicts)
        bad = next(c for c in ranked if c.ticker == "BAD")
        assert not math.isfinite(bad.score)                       # E non-finite, stamped raw
        assert getattr(bad, "score_uncapped", None) is None       # clamp SKIPPED, no new keys
        assert getattr(bad, "score_cap_value", None) is None
        assert len(ranked) == 2                                   # crash-free ordering


def test_f_relvol_none_neutral_multiplier_published_identical(tmp_path, monkeypatch):
    scores = {}
    for state in ("off", "on"):
        if state == "on":
            monkeypatch.setenv(CAP_FLAG, "1")
        else:
            monkeypatch.delenv(CAP_FLAG, raising=False)
        c = _cand("NR", gap=0.30, relvol=None)
        ranked = ranking.rank_candidates([c], verdicts={"NR": _v(conf=None, src=None)})
        assert ranked[0].score == round(0.30 * 1.0, 6)            # mult-1.0 fallback, finite
        scores[state] = ranked[0].score
        out = tmp_path / ("wl_%s.json" % state)
        watchlist_io.atomic_write_json(str(out), mw.build_movers_watchlist_dict(
            ranked, {"NR": _v(conf=None, src=None)}, scan_id="s", scanned_at=FIXED))
        assert out.exists()                                       # published
    assert scores["on"] == scores["off"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_f_relvol_nonfinite_rejected(tmp_path, cap_on, bad):
    _publish_rejected(tmp_path, [_cand("BAD", relvol=bad)], {"BAD": _v()})


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_f_edgar_nonfinite_passes_or_guard_and_is_rejected(tmp_path, cap_on, bad):
    # pin the mechanism: `or 0.0` does NOT neutralize a non-finite edgar_signal (truthy)
    c = _cand("BAD", edgar=bad)
    assert not math.isfinite(ranking.score(c))
    _publish_rejected(tmp_path, [_cand("BAD", edgar=bad)], {"BAD": _v()})


def test_f_float_factor_nonfinite_pins():
    assert ranking.float_factor(float("inf")) == 0.0    # 1 - K*inf^2 -> -inf; max(0, -inf) = 0
    assert ranking.float_factor(float("nan")) == 0.0    # NaN compares False; max keeps 0.0
    assert ranking.float_factor(float("-inf")) == 1.0   # -inf <= FLOAT_SOFT_MAX


@pytest.mark.parametrize("fs,ff", [(float("inf"), 0.0), (float("nan"), 0.0), (float("-inf"), 1.0)])
def test_f_float_nonfinite_pinned_factor_e_finite_identical(tmp_path, monkeypatch, fs, ff):
    # Not a "published" case: the injected non-finite float_shares OPERAND itself is a v2
    # wire field, so allow_nan=False rejects the artifact regardless of the (finite) score.
    # The load-bearing claims still hold
    # and are pinned here: float_factor pinned (inf/nan -> 0.0, -inf -> 1.0), E finite, ranking
    # normal, and behavior IDENTICAL OFF vs ON (the cap adds no divergence and no emission path).
    scores = {}
    for state in ("off", "on"):
        if state == "on":
            monkeypatch.setenv(CAP_FLAG, "1")
        else:
            monkeypatch.delenv(CAP_FLAG, raising=False)
        c = _cand("F", gap=0.30, relvol=10.0, fs=fs)              # base 3.0
        ranked = ranking.rank_candidates([c], verdicts={"F": _v(conf=None, src=None)})
        assert ranked[0].score == round(3.0 * ff, 6)
        assert math.isfinite(ranked[0].score)
        scores[state] = ranked[0].score
        sub = tmp_path / ("%s_%s" % (state, ff))
        sub.mkdir()
        _publish_rejected(sub, [_cand("F", gap=0.30, relvol=10.0, fs=fs)],
                          {"F": _v(conf=None, src=None)})
    assert scores["on"] == scores["off"]


@pytest.mark.parametrize("bad_conf", [float("nan"), float("inf")])
def test_f_confidence_nonfinite_rejected(tmp_path, cap_on, bad_conf):
    v = _v(conf=bad_conf, src="finnhub")
    c = _cand("BAD")
    assert not math.isfinite(ranking.effective_score(c, {"BAD": v}))
    _publish_rejected(tmp_path, [_cand("BAD")], {"BAD": v})


# --- G: durable marker -----------------------------------------------------------------------

def _write_marker(state_dir, content):
    p = Path(state_dir) / "score_cap_state.json"
    if isinstance(content, str):
        p.write_text(content, encoding="utf-8")
    else:
        p.write_text(json.dumps(content), encoding="utf-8")
    return p


def _marker(active=True, value=8.0):
    return {"cap_active": active, "cap_value": value,
            "declared_at": "2026-08-11T13:00:00+00:00", "declared_by": "runbook"}


def test_g_marker_absent_both_sides_behave_as_today(tmp_path, cap_off):
    from momentum_scanner import score_cap_state
    assert score_cap_state.read_marker(str(tmp_path)) is None
    res, banner = smd.score_cap_startup(str(tmp_path))
    assert banner == "score cap: OFF"
    assert score_cap_state.watchlist_cap_mismatch(None, {"scan_id": "s"}) is None


def test_g_marker_match_normal_startup_banner(tmp_path, cap_on):
    _write_marker(tmp_path, _marker(active=True, value=8.0))
    res, banner = smd.score_cap_startup(str(tmp_path))
    assert banner == "score cap: ENABLED value=8.0"


def test_g_inactive_marker_matches_off_config(tmp_path, cap_off):
    _write_marker(tmp_path, _marker(active=False, value=8.0))
    res, banner = smd.score_cap_startup(str(tmp_path))
    assert banner == "score cap: OFF"


def test_g_env_lost_mismatch_aborts_discovery(tmp_path, cap_off):
    # wrong-launcher restart: marker declares the cap active but the env resolved OFF
    _write_marker(tmp_path, _marker(active=True, value=8.0))
    with pytest.raises(smd.ScoreCapStartupAbort):
        smd.score_cap_startup(str(tmp_path))


def test_g_value_mismatch_aborts_discovery(tmp_path, monkeypatch):
    monkeypatch.setenv(CAP_FLAG, "1")
    monkeypatch.setenv(CAP_VALUE, "9.0")
    _write_marker(tmp_path, _marker(active=True, value=8.0))
    with pytest.raises(smd.ScoreCapStartupAbort):
        smd.score_cap_startup(str(tmp_path))


@pytest.mark.parametrize("content", [
    "{not json",                                        # unparseable
    json.dumps(["not", "a", "dict"]),                   # wrong shape
    json.dumps({"cap_value": 8.0}),                     # cap_active missing
    json.dumps({"cap_active": "yes", "cap_value": 8.0}),  # cap_active wrong type
    json.dumps({"cap_active": True}),                   # cap_value missing
    json.dumps({"cap_active": True, "cap_value": "8"}),   # cap_value wrong type
    json.dumps({"cap_active": True, "cap_value": True}),  # bool is not a numeric value
])
def test_g_malformed_marker_read_raises_and_startup_aborts(tmp_path, cap_off, content):
    from momentum_scanner import score_cap_state
    _write_marker(tmp_path, content)
    with pytest.raises(score_cap_state.MalformedMarkerError):
        score_cap_state.read_marker(str(tmp_path))
    with pytest.raises(smd.ScoreCapStartupAbort):
        smd.score_cap_startup(str(tmp_path))


def test_g_consumer_refusal_contract():
    from momentum_scanner import score_cap_state
    active = _marker(active=True, value=8.0)
    inactive = _marker(active=False, value=8.0)
    good = {"scan_id": "s", "score_cap": {"active": True, "value": 8.0}}
    # active marker: missing key -> refusal named watchlist_cap_mismatch
    r = score_cap_state.watchlist_cap_mismatch(active, {"scan_id": "s"})
    assert r is not None and "watchlist_cap_mismatch" in r
    # active marker: wrong value -> refusal
    r = score_cap_state.watchlist_cap_mismatch(
        active, {"scan_id": "s", "score_cap": {"active": True, "value": 9.0}})
    assert r is not None and "watchlist_cap_mismatch" in r
    # active marker: matching artifact -> accepted
    assert score_cap_state.watchlist_cap_mismatch(active, good) is None
    # inactive marker: key must be ABSENT
    assert score_cap_state.watchlist_cap_mismatch(inactive, {"scan_id": "s"}) is None
    r = score_cap_state.watchlist_cap_mismatch(inactive, good)
    assert r is not None and "watchlist_cap_mismatch" in r


def test_g_daemon_main_marker_mismatch_aborts_no_publish(tmp_path, monkeypatch):
    assert hasattr(smd, "score_cap_startup"), "score_cap_startup not implemented"
    monkeypatch.delenv(CAP_FLAG, raising=False)
    monkeypatch.delenv(CAP_VALUE, raising=False)
    _write_marker(tmp_path, _marker(active=True, value=8.0))
    rc = smd.main(["--state-dir", str(tmp_path)])
    assert rc == 2
    assert not (tmp_path / "watchlist.json").exists()
    assert not (tmp_path / config.DAEMON_HEARTBEAT_FILE).exists()


def test_c_on_then_off_rerank_of_same_objects_clears_stale_cap_provenance(monkeypatch):
    # Re-ranking the SAME candidate objects after an ON pass
    # with the cap OFF must publish pure OFF rows -- raw score, no score_uncapped /
    # score_cap_value, no top-level score_cap -- never an uncapped score wearing stale cap
    # provenance. Unreachable in the daemon today (fresh candidates per cycle) but load-bearing
    # for analyzer/test reuse and for the cohort contract's integrity.
    monkeypatch.setenv(CAP_FLAG, "1")
    monkeypatch.delenv(CAP_VALUE, raising=False)
    candidates, verdicts = cap_fixtures.build_cohort()
    ranking.rank_candidates(candidates, verdicts=verdicts)          # ON pass stamps provenance
    assert any(c.score_uncapped is not None for c in candidates)    # precondition: stamps exist
    monkeypatch.delenv(CAP_FLAG)
    ranked_off = ranking.rank_candidates(candidates, verdicts=verdicts)   # SAME objects, OFF
    for c in ranked_off:
        assert c.score_uncapped is None, c.ticker
        assert c.score_cap_value is None, c.ticker
    d = mw.build_movers_watchlist_dict(ranked_off, verdicts, scan_id="s", scanned_at=FIXED)
    assert "score_cap" not in d
    for row in d["candidates"]:
        assert set(row.keys()) == OFF_ROW_KEYS, row["ticker"]
    by_ticker = {c.ticker: c for c in ranked_off}
    assert by_ticker["T1"].score == 2000.18                         # raw score restored exactly

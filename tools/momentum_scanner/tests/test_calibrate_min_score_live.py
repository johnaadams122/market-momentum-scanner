"""Tests for the min_score_live calibration harness (read-only outcome analysis).

The harness ingests the trade outcome journal (closed rows carrying the producer `score` +
realized `pnl`), sweeps candidate floor cutoffs, and recommends the LOWEST floor whose EV
lower-confidence bound is positive at a minimum sample size.
"""
import math

from momentum_scanner import calibrate_min_score_live as cal


def _row(**kw):
    """A minimal closed trade-journal row; overridable per test."""
    base = {
        "action": "TRADE_OPEN",
        "id": kw.get("id", "t-AAA-1"),
        "status": "closed",
        "ticker": "AAA",
        "pnl": 10.0,
        "score": 0.5,
        "catalyst_score": 0.1,
        "gate_decision": "PROCEED",
        "outcome": "target",
        "paper": True,
    }
    base.update(kw)
    return base


# --- extract_scored_trades: which rows count -------------------------------------

def test_extract_keeps_closed_proceed_rows_with_numeric_score_and_pnl():
    rows = [_row(id="t-AAA-1", score=0.5, pnl=10.0)]
    # basis pinned to the legacy gross field: this test is about WHICH ROWS COUNT,
    # not about costing. The default basis is exercised in the cost-basis section.
    trades = cal.extract_scored_trades(rows, basis="gross_pnl")
    assert len(trades) == 1
    assert trades[0].score == 0.5
    assert trades[0].pnl == 10.0


def test_extract_skips_non_closed_rows():
    rows = [_row(id="a", status="pending"), _row(id="b", status="open"),
            _row(id="c", status="orphaned")]
    assert cal.extract_scored_trades(rows) == []


def test_extract_skips_rows_with_null_or_nonnumeric_pnl():
    rows = [_row(id="a", pnl=None), _row(id="b", pnl="x")]
    assert cal.extract_scored_trades(rows) == []


def test_extract_skips_rows_with_missing_or_nonfinite_score():
    rows = [_row(id="a", score=None), _row(id="b", score="0.5"),
            _row(id="c", score=float("inf"))]
    assert cal.extract_scored_trades(rows) == []


def test_extract_excludes_bool_score_and_pnl():
    # bool is an int subclass -- must not be treated as a numeric score/pnl
    rows = [_row(id="a", score=True), _row(id="b", pnl=True)]
    assert cal.extract_scored_trades(rows) == []


def test_extract_gate_proceed_excludes_reject_when_asked():
    # PROCEED is not the default gate (see test_default_gate_is_all_not_proceed); the
    # filter itself still works when selected explicitly, which is what this test pins.
    rows = [_row(id="a", gate_decision="PROCEED"), _row(id="b", gate_decision="REJECT")]
    trades = cal.extract_scored_trades(rows, gate="PROCEED", basis="gross_pnl")
    assert [t.ticker for t in trades] == ["AAA"]
    assert len(trades) == 1


def test_extract_gate_all_includes_reject():
    rows = [_row(id="a", gate_decision="PROCEED"), _row(id="b", gate_decision="REJECT")]
    assert len(cal.extract_scored_trades(rows, gate="all", basis="gross_pnl")) == 2


def test_extract_dedup_by_id_terminal_wins():
    # a pending then a closed for the same id -> the closed terminal row is the one counted
    rows = [_row(id="t-AAA-1", status="pending", pnl=None, score=0.5),
            _row(id="t-AAA-1", status="closed", pnl=12.0, score=0.5)]
    trades = cal.extract_scored_trades(rows, basis="gross_pnl")
    assert len(trades) == 1
    assert trades[0].pnl == 12.0


def test_extract_dedup_stray_later_pending_does_not_regress_closed():
    rows = [_row(id="t-AAA-1", status="closed", pnl=12.0, score=0.5),
            _row(id="t-AAA-1", status="pending", pnl=None, score=0.5)]
    trades = cal.extract_scored_trades(rows, basis="gross_pnl")
    assert len(trades) == 1 and trades[0].pnl == 12.0


# --- EV lower-confidence bounds --------------------------------------------------

def test_ev_lcb_normal_zero_variance_equals_mean():
    assert cal.ev_lcb_normal([10.0, 10.0, 10.0, 10.0]) == 10.0


def test_ev_lcb_normal_known_value():
    # mean 10, sample stdev sqrt(200)=14.142..., n=2 -> 10 - 1.645*14.142/sqrt(2) = -6.45
    got = cal.ev_lcb_normal([0.0, 20.0])
    assert math.isclose(got, 10.0 - 1.645 * math.sqrt(200.0) / math.sqrt(2), rel_tol=1e-9)


def test_ev_lcb_normal_undefined_below_two_samples():
    assert cal.ev_lcb_normal([]) == float("-inf")
    assert cal.ev_lcb_normal([5.0]) == float("-inf")


def test_ev_lcb_bootstrap_deterministic_for_fixed_seed():
    pnls = [-10.0, 20.0, -5.0, 15.0, 3.0, -8.0, 11.0]
    a = cal.ev_lcb_bootstrap(pnls, iters=1000, seed=42)
    b = cal.ev_lcb_bootstrap(pnls, iters=1000, seed=42)
    assert a == b


def test_ev_lcb_bootstrap_zero_variance_equals_value():
    assert cal.ev_lcb_bootstrap([7.0] * 10, iters=500, seed=1) == 7.0


def test_ev_lcb_bootstrap_not_above_mean():
    pnls = [-10.0, 20.0, -5.0, 15.0, 3.0, -8.0, 11.0, -2.0]
    mean = sum(pnls) / len(pnls)
    assert cal.ev_lcb_bootstrap(pnls, iters=2000, seed=7) <= mean


# --- cost basis + exit model -----------------------------------------------------
# The raw `pnl` field is always GROSS (cost is never folded into it) and reflects the
# scale_trail exit. The default basis is the fixed-2R exit, net of the row's own
# measured spread; gross_pnl is the uncosted legacy basis.

def _costed_row(**kw):
    """A closed row carrying BOTH exits plus the fields empirical costing needs."""
    base = _row(pnl=100.0, pnl_fixed_2r=50.0, shares=1000,
                features={"spread_at_signal": 0.02}, exit_spread_at_close=0.02)
    base.update(kw)
    return base


def _fixed_cost(amount):
    """Inject a deterministic cost so these tests need no external cost model."""
    return lambda row: amount


def test_default_basis_is_net_fixed_2r_not_gross_pnl():
    # gross pnl 100 must NOT be what the sweep sees; fixed-2R 50 minus cost 20 = 30
    trades = cal.extract_scored_trades([_costed_row()], cost_fn=_fixed_cost(20.0))
    assert len(trades) == 1
    assert trades[0].pnl == 30.0


def test_gross_pnl_basis_still_available_and_uncosted():
    trades = cal.extract_scored_trades([_costed_row()], basis="gross_pnl",
                                       cost_fn=_fixed_cost(20.0))
    assert trades[0].pnl == 100.0        # legacy basis: raw field, no cost subtracted


def test_net_pnl_basis_costs_the_scale_trail_exit():
    trades = cal.extract_scored_trades([_costed_row()], basis="net_pnl",
                                       cost_fn=_fixed_cost(20.0))
    assert trades[0].pnl == 80.0         # 100 gross - 20 cost


def test_default_gate_is_all_not_proceed():
    # When the catalyst gate does not separate outcomes, calibrating a floor on the PROCEED
    # cohort alone tunes a non-discriminating subset, so the default gate keeps every verdict.
    rows = [_costed_row(id="a", gate_decision="PROCEED"),
            _costed_row(id="b", gate_decision="REJECT")]
    assert len(cal.extract_scored_trades(rows, cost_fn=_fixed_cost(0.0))) == 2


def test_gate_proceed_still_selectable_explicitly():
    rows = [_costed_row(id="a", gate_decision="PROCEED"),
            _costed_row(id="b", gate_decision="REJECT")]
    trades = cal.extract_scored_trades(rows, gate="PROCEED", cost_fn=_fixed_cost(0.0))
    assert len(trades) == 1 and trades[0].gate_decision == "PROCEED"


def test_unresolved_counterfactual_is_excluded_never_treated_as_zero():
    # A pending_replay row has no fixed-2R outcome yet.
    # Counting it as 0.0 would silently drag EV toward zero and fake precision.
    rows = [_costed_row(id="a", pnl_fixed_2r=None), _costed_row(id="b", pnl_fixed_2r=50.0)]
    trades = cal.extract_scored_trades(rows, cost_fn=_fixed_cost(20.0))
    assert len(trades) == 1 and trades[0].pnl == 30.0


def test_unresolved_rows_are_counted_not_silently_dropped():
    rows = [_costed_row(id="a", pnl_fixed_2r=None), _costed_row(id="b", pnl_fixed_2r=50.0)]
    res = cal.calibrate(rows, min_n=1, cost_fn=_fixed_cost(20.0))
    assert res["n_unresolved"] == 1
    assert res["n_trades"] == 1
    assert res["basis"] == "net_fixed_2r"


def test_nonfinite_cost_excludes_the_row_rather_than_poisoning_ev():
    trades = cal.extract_scored_trades([_costed_row()], cost_fn=_fixed_cost(float("nan")))
    assert trades == []


def test_default_cost_prefers_measured_spread_over_flat_anchor(monkeypatch):
    """The measured-spread cost wins over the flat trade_cost fallback when it is available."""
    from types import SimpleNamespace
    synthetic = SimpleNamespace(scanner_spread_cost=lambda row: 60.0,
                                trade_cost=lambda row: 20.0)
    monkeypatch.setattr(cal, "_COST_MODEL", synthetic)
    cm = cal.cost_model()
    row = _costed_row(features={"spread_at_signal": 0.06}, exit_spread_at_close=0.06)
    assert cm.scanner_spread_cost(row) == 60.0     # (0.06/2 + 0.06/2) * 1000 shares
    assert cm.trade_cost(row) == 20.0              # flat anchor: 0.02 * 1000 shares
    assert cal.default_cost(row) == 60.0           # empirical wins when it is measurable


def test_default_cost_falls_back_to_flat_anchor_when_spread_unmeasurable(monkeypatch):
    from types import SimpleNamespace
    synthetic = SimpleNamespace(scanner_spread_cost=lambda row: None,
                                trade_cost=lambda row: 20.0)
    monkeypatch.setattr(cal, "_COST_MODEL", synthetic)
    row = _costed_row(features={"spread_at_signal": None}, exit_spread_at_close=None)
    assert cal.cost_model().scanner_spread_cost(row) is None
    assert cal.default_cost(row) == 20.0          # flat anchor: 0.02 * 1000 shares


def test_unknown_basis_is_a_hard_error_not_a_silent_fallback():
    try:
        cal.extract_scored_trades([_costed_row()], basis="bogus", cost_fn=_fixed_cost(0.0))
    except ValueError:
        return
    raise AssertionError("unknown basis must raise, never silently pick a default")


# --- sweep -----------------------------------------------------------------------

def test_sweep_thresholds_are_unique_scores_ascending():
    trades = [cal.Trade(0.2, -5.0, "A", "stop", "PROCEED"),
              cal.Trade(0.5, 10.0, "B", "target", "PROCEED"),
              cal.Trade(0.8, 10.0, "C", "target", "PROCEED")]
    rows = cal.sweep(trades, min_n=1, seed=1)
    assert [r["threshold"] for r in rows] == [0.2, 0.5, 0.8]


def test_sweep_counts_and_ev_for_cutoff():
    trades = [cal.Trade(0.2, -5.0, "A", "stop", "PROCEED"),
              cal.Trade(0.5, 10.0, "B", "target", "PROCEED"),
              cal.Trade(0.8, 10.0, "C", "target", "PROCEED")]
    rows = {r["threshold"]: r for r in cal.sweep(trades, min_n=1, seed=1)}
    assert rows[0.2]["n"] == 3 and math.isclose(rows[0.2]["ev"], 5.0)
    assert rows[0.5]["n"] == 2 and math.isclose(rows[0.5]["ev"], 10.0)
    assert rows[0.8]["n"] == 1 and math.isclose(rows[0.8]["ev"], 10.0)
    assert math.isclose(rows[0.2]["win_rate"], 2 / 3)
    assert math.isclose(rows[0.2]["total_pnl"], 15.0)


# --- recommend -------------------------------------------------------------------

def _bucketed(low_pnl, low_n, high_pnl, high_n, low_score=0.2, high_score=0.6):
    return ([cal.Trade(low_score, low_pnl, "L", "stop", "PROCEED")] * low_n
            + [cal.Trade(high_score, high_pnl, "H", "target", "PROCEED")] * high_n)


def test_recommend_lowest_cutoff_that_turns_ev_lcb_positive():
    trades = _bucketed(-10.0, 30, 10.0, 30)      # low bucket loses, high bucket wins
    rows = cal.sweep(trades, min_n=20, seed=1)
    rec = cal.recommend(rows, min_n=20)
    assert rec["floor"] == 0.6


def test_recommend_none_when_no_cutoff_is_positive():
    trades = _bucketed(-10.0, 30, -3.0, 30)      # everything loses
    rows = cal.sweep(trades, min_n=20, seed=1)
    rec = cal.recommend(rows, min_n=20)
    assert rec["floor"] is None
    assert "no" in rec["reason"].lower()


def test_recommend_respects_min_sample_size():
    trades = _bucketed(-10.0, 60, 10.0, 30)      # only the +EV bucket is small (n=30)
    rows = cal.sweep(trades, min_n=20, seed=1)
    rec = cal.recommend(rows, min_n=40)          # require n>=40 -> the +EV cutoff is too small
    assert rec["floor"] is None


def test_recommend_empty_returns_none_with_reason():
    rec = cal.recommend([], min_n=20)
    assert rec["floor"] is None
    assert rec["reason"]


# --- end-to-end on raw rows ------------------------------------------------------

def test_calibrate_from_rows_no_scored_data_is_clean_not_error():
    # closed rows exist but none carry a score
    rows = [_row(id="a", score=None, pnl=5.0), _row(id="b", status="pending")]
    result = cal.calibrate(rows, min_n=20, seed=1, basis="gross_pnl")
    assert result["floor"] is None
    assert result["n_trades"] == 0
    assert result["reason"]


def test_calibrate_from_rows_recommends_floor():
    rows = []
    for i in range(30):
        rows.append(_row(id="lo-%d" % i, score=0.2, pnl=-10.0, gate_decision="PROCEED"))
    for i in range(30):
        rows.append(_row(id="hi-%d" % i, score=0.6, pnl=10.0, gate_decision="PROCEED"))
    result = cal.calibrate(rows, min_n=20, seed=1, basis="gross_pnl")
    assert result["floor"] == 0.6
    assert result["n_trades"] == 60


# --- driver-bucketing analysis: which driver actually predicts PnL -----------------

def _t(score, pnl, feats, gate="PROCEED"):
    return cal.Trade(score, pnl, "X", "target", gate, feats)


def test_spearman_perfect_monotonic():
    assert math.isclose(cal._spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
    assert math.isclose(cal._spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)


def test_spearman_zero_variance_or_too_few():
    assert cal._spearman([1, 2, 3], [5, 5, 5]) == 0.0
    assert cal._spearman([1, 2], [1, 2]) == 0.0   # < 3 points


def test_bucket_numeric_quartiles():
    trades = [_t(1.0, float(v), {"gap_pct": v}) for v in [1, 2, 3, 4, 5, 6, 7, 8]]
    res = cal.bucket_numeric(trades, "gap_pct", bins=4, seed=1)
    assert [b["n"] for b in res["buckets"]] == [2, 2, 2, 2]
    assert math.isclose(res["buckets"][0]["ev"], 1.5)
    assert math.isclose(res["buckets"][-1]["ev"], 7.5)
    assert math.isclose(res["spearman"], 1.0)


def test_bucket_numeric_excludes_missing_values():
    trades = [_t(1.0, 5.0, {"gap_pct": 0.2}), _t(1.0, 5.0, {"gap_pct": None}), _t(1.0, 5.0, None)]
    res = cal.bucket_numeric(trades, "gap_pct", bins=4, seed=1)
    assert res["excluded"] == 2
    assert sum(b["n"] for b in res["buckets"]) == 1


def test_group_categorical_by_value():
    trades = ([_t(1.0, -5.0, {"dilution_flag": "active_offering"}) for _ in range(3)]
              + [_t(1.0, 10.0, {"dilution_flag": None}) for _ in range(3)])
    res = cal.group_categorical(trades, "dilution_flag", min_n=1, seed=1)
    evs = {g["value"]: g["ev"] for g in res["groups"]}
    assert math.isclose(evs["active_offering"], -5.0)
    assert math.isclose(evs[None], 10.0)
    assert res["groups"][0]["value"] is None          # sorted EV desc -> clean group first
    assert math.isclose(res["spread"], 15.0)


def test_group_categorical_excludes_featureless_rows():
    trades = [_t(1.0, 5.0, {"dilution_flag": "shelf_risk"}), _t(1.0, 5.0, None)]
    res = cal.group_categorical(trades, "dilution_flag", min_n=1, seed=1)
    assert res["excluded"] == 1


def test_analyze_drivers_ranks_most_predictive_numeric_first():
    # gap_pct rises with pnl (predictive); rel_volume constant (no signal). analyze_drivers takes
    # raw journal rows (not Trade objects) -- it runs them through extract_scored_trades itself.
    rows = [_row(id="r%d" % v, score=1.0, pnl=float(v), gate_decision="PROCEED",
                 features={"gap_pct": v, "rel_volume": 5.0}) for v in range(1, 13)]
    res = cal.analyze_drivers(rows, min_n=1, bins=4, seed=1, basis="gross_pnl")
    assert res["numeric"][0]["key"] == "gap_pct"
    assert abs(res["numeric"][0]["spearman"]) > abs(res["numeric"][1]["spearman"])


def test_analyze_drivers_empty_is_clean():
    res = cal.analyze_drivers([], min_n=1, seed=1)
    assert res["n_trades"] == 0
    assert res["numeric"] and all(d["buckets"] == [] for d in res["numeric"])


def test_analyze_drivers_from_rows_end_to_end():
    rows = []
    for i, v in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]):
        rows.append(_row(id="r%d" % i, score=1.0, pnl=float(i), gate_decision="PROCEED",
                         features={"gap_pct": v, "rel_volume": 5.0,
                                   "dilution_flag": ("active_offering" if i < 3 else None)}))
    res = cal.analyze_drivers(rows, min_n=1, bins=3, seed=1, basis="gross_pnl")
    assert res["n_trades"] == 6
    dil = next(d for d in res["categorical"] if d["key"] == "dilution_flag")
    evs = {g["value"]: g["ev"] for g in dil["groups"]}
    assert evs[None] > evs["active_offering"]


def test_extract_scored_trades_ignores_live_and_unmarked_rows():
    rows = [
        {"action": cal.ACTION, "id": "a", "status": "closed", "score": 1.0, "pnl": 5.0,
         "paper": True, "gate_decision": "PROCEED"},
        {"action": cal.ACTION, "id": "b", "status": "closed", "score": 1.0, "pnl": 500.0,
         "paper": False, "gate_decision": "PROCEED"},          # non-paper row -> excluded
        {"action": cal.ACTION, "id": "c", "status": "closed", "score": 1.0, "pnl": 500.0,
         "gate_decision": "PROCEED"},                          # no paper key -> excluded (fail-closed)
    ]
    trades = cal.extract_scored_trades(rows, basis="gross_pnl")
    assert [t.pnl for t in trades] == [5.0]


# --- score-cap cohort contract -------------------------------------------------------------
# Partition the calibration population (deduped CLOSED PAPER rows) by `score_cap_value`.
# Six pinned cases, one test per case; every refusal names its rule and the cohort sizes.

import pytest


def test_cohort_rule1_key_absent_on_all_rows_proceeds_as_uncapped():
    rows = [_row(id="a"), _row(id="b", score=0.7)]
    result = cal.calibrate(rows, min_n=1, basis="gross_pnl")
    assert result["cohort"] == "uncapped"
    assert result["cap_value"] is None
    assert result["n_trades"] == 2


def test_cohort_rule2_single_cap_labels_cohort_and_refuses_floor_above_cap():
    # (a) the label half: one identical finite stamp on every row -> proceed, capped@V
    rows = [_row(id="a", score=0.5, score_cap_value=8.0),
            _row(id="b", score=0.7, score_cap_value=8.0)]
    result = cal.calibrate(rows, min_n=1, basis="gross_pnl")
    assert result["cohort"] == "capped@8"
    assert result["cap_value"] == 8.0
    # (b) the ENFORCED half: rows whose scores sit
    # above their own stamped cap (a broken producer) would recommend a floor > cap; that
    # recommendation must be REFUSED, not printed as advisory.
    bad = [_row(id="x", score=9.0, pnl=10.0, score_cap_value=8.0),
           _row(id="y", score=9.0, pnl=10.0, score_cap_value=8.0)]
    with pytest.raises(cal.CohortContractError) as ei:
        cal.calibrate(bad, min_n=1, basis="gross_pnl")
    msg = str(ei.value)
    assert "rule 2" in msg and "8" in msg


def test_cohort_rule3_mixed_stamped_and_unstamped_refused_with_both_counts():
    rows = [_row(id="a", score_cap_value=8.0), _row(id="b"), _row(id="c")]
    with pytest.raises(cal.CohortContractError) as ei:
        cal.calibrate(rows, min_n=1, basis="gross_pnl")
    msg = str(ei.value)
    assert "rule 3" in msg
    assert "1" in msg and "2" in msg          # both cohort sizes reported


def test_cohort_rule4_null_cap_value_is_malformed_and_names_rows():
    rows = [_row(id="nullrow", score_cap_value=None), _row(id="b", score_cap_value=8.0)]
    with pytest.raises(cal.CohortContractError) as ei:
        cal.calibrate(rows, min_n=1, basis="gross_pnl")
    msg = str(ei.value)
    assert "rule 4" in msg and "nullrow" in msg


def test_cohort_rule5_nonnumeric_nonfinite_or_nonpositive_cap_value_is_malformed():
    for bad in ("8", float("nan"), float("inf"), 0.0, -3.0, True):
        rows = [_row(id="badrow", score_cap_value=bad)]
        with pytest.raises(cal.CohortContractError) as ei:
            cal.calibrate(rows, min_n=1, basis="gross_pnl")
        msg = str(ei.value)
        assert "rule 5" in msg and "badrow" in msg, bad


def test_cohort_rule6_multiple_distinct_caps_refused_reporting_all_values_with_counts():
    rows = [_row(id="a1", score_cap_value=8.0), _row(id="a2", score_cap_value=8.0),
            _row(id="b1", score_cap_value=9.0),
            _row(id="c1", score_cap_value=12.5)]
    with pytest.raises(cal.CohortContractError) as ei:
        cal.calibrate(rows, min_n=1, basis="gross_pnl")
    msg = str(ei.value)
    assert "rule 6" in msg
    for frag in ("8", "9", "12.5"):           # ALL distinct values, not just "both"
        assert frag in msg, frag
    assert "2" in msg                         # per-value counts present (8.0 appears twice)


def test_cohort_contract_applies_to_drivers_mode_too():
    rows = [_row(id="a", score_cap_value=8.0, features={"gap_pct": 0.2}),
            _row(id="b", features={"gap_pct": 0.3})]
    with pytest.raises(cal.CohortContractError):
        cal.analyze_drivers(rows, min_n=1, basis="gross_pnl")


def test_cohort_partition_ignores_non_closed_and_non_paper_rows():
    # the partition population is the CALIBRATION population: deduped closed paper rows.
    # A non-paper row carrying a stamp must not poison an otherwise-uncapped paper cohort.
    rows = [_row(id="a"), _row(id="live", paper=False, score_cap_value=8.0),
            _row(id="open", status="open", score_cap_value=8.0)]
    result = cal.calibrate(rows, min_n=1, basis="gross_pnl")
    assert result["cohort"] == "uncapped"


# --- CLI: --basis is honoured in drivers mode; --json is strict JSON -------------------

import json  # noqa: E402
from types import SimpleNamespace  # noqa: E402


def _write_journal(tmp_path, rows):
    p = tmp_path / "journal.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(p)


def _no_cost_adapter(monkeypatch):
    monkeypatch.delenv("MARKET_MOMENTUM_COST_MODEL", raising=False)
    monkeypatch.setattr(cal, "_COST_MODEL", None)


def _driver_rows(n=8):
    # gross pnl 100 on every row; NO fixed-2R value, so a silently-used net_fixed_2r basis finds
    # nothing to score and reports zero trades.
    return [_row(id="d%d" % i, score=1.0 + i, pnl=100.0, features={"gap_pct": 0.1 * (i + 1),
                                                                   "rel_volume": 5.0})
            for i in range(n)]


def test_cli_drivers_gross_pnl_uses_gross_basis_and_needs_no_cost_adapter(tmp_path, monkeypatch, capsys):
    _no_cost_adapter(monkeypatch)
    path = _write_journal(tmp_path, _driver_rows())
    rc = cal.main(["--journal", path, "--mode", "drivers", "--basis", "gross_pnl", "--json",
                   "--min-n", "1"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["basis"] == "gross_pnl"
    assert out["n_trades"] == 8                       # scored on the gross field


def test_cli_drivers_net_pnl_uses_the_net_pnl_basis(tmp_path, monkeypatch, capsys):
    # net_pnl = gross pnl minus the configured cost (here a fixed synthetic 20), so 100 -> 80.
    monkeypatch.setattr(cal, "_COST_MODEL", SimpleNamespace(scanner_spread_cost=lambda row: 20.0,
                                                            trade_cost=lambda row: 20.0))
    rows = [dict(r, pnl_fixed_2r=50.0) for r in _driver_rows()]
    path = _write_journal(tmp_path, rows)
    rc = cal.main(["--journal", path, "--mode", "drivers", "--basis", "net_pnl", "--json",
                   "--min-n", "1"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["basis"] == "net_pnl"
    gap = next(d for d in out["numeric"] if d["key"] == "gap_pct")
    assert all(b["ev"] == 80.0 for b in gap["buckets"])


def test_cli_drivers_default_basis_is_net_fixed_2r(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cal, "_COST_MODEL", SimpleNamespace(scanner_spread_cost=lambda row: 20.0,
                                                            trade_cost=lambda row: 20.0))
    rows = [dict(r, pnl_fixed_2r=50.0) for r in _driver_rows()]
    path = _write_journal(tmp_path, rows)
    assert cal.main(["--journal", path, "--mode", "drivers", "--json", "--min-n", "1"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["basis"] == "net_fixed_2r"
    gap = next(d for d in out["numeric"] if d["key"] == "gap_pct")
    assert all(b["ev"] == 30.0 for b in gap["buckets"])


def _reject_constant(name):
    raise ValueError("non-standard JSON constant: %s" % name)


def _unique_top_score_rows():
    # The highest score is unique, so the top cutoff holds ONE trade and its normal-approximation
    # lower bound is undefined (negative infinity internally).
    return [_row(id="u%d" % i, score=float(i + 1), pnl=float(10 + i)) for i in range(6)]


def test_cli_calibrate_json_is_strict_json_with_null_for_undefined_values(tmp_path, monkeypatch, capsys):
    _no_cost_adapter(monkeypatch)
    path = _write_journal(tmp_path, _unique_top_score_rows())
    assert cal.main(["--journal", path, "--basis", "gross_pnl", "--json", "--min-n", "1"]) == 0
    text = capsys.readouterr().out
    out = json.loads(text, parse_constant=_reject_constant)        # no Infinity / NaN tokens
    assert out["sweep"][-1]["n"] == 1
    assert out["sweep"][-1]["ev_lcb_normal"] is None


def test_cli_drivers_json_is_strict_json(tmp_path, monkeypatch, capsys):
    _no_cost_adapter(monkeypatch)
    path = _write_journal(tmp_path, _driver_rows())
    assert cal.main(["--journal", path, "--mode", "drivers", "--basis", "gross_pnl", "--json",
                     "--min-n", "1"]) == 0
    json.loads(capsys.readouterr().out, parse_constant=_reject_constant)


def test_cli_human_report_still_shows_the_undefined_bound(tmp_path, monkeypatch, capsys):
    _no_cost_adapter(monkeypatch)
    path = _write_journal(tmp_path, _unique_top_score_rows())
    assert cal.main(["--journal", path, "--basis", "gross_pnl", "--min-n", "1"]) == 0
    assert "-inf" in capsys.readouterr().out

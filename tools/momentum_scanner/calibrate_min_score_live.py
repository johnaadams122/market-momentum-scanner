"""Calibration harness for a `min_score_live` score floor (read-only analysis).

METHOD
    Ingest closed PAPER rows (paper==true only) carrying (score, pnl), sweep candidate cutoffs,
    and pick the LOWEST cutoff whose EV lower-confidence bound is positive at a minimum sample
    size -- "positive ev_lcb, not raw EV". The lowest cutoff that clears maximizes trade volume
    subject to a *confidently* positive EV. Any floor must be <= the active score cap when one
    is enabled (stamped scores are clamped to min(E, cap)); the score-cap cohort contract below
    (partition_cap_cohort, rule 2) refuses such a floor.

COST BASIS AND EXIT MODEL
    The default basis is `net_fixed_2r`: the fixed-2R exit, net of the row's own measured
    spread. `net_pnl` costs the scale_trail exit; `gross_pnl` is the uncosted legacy basis.
    Costing uses an importable module named by MARKET_MOMENTUM_COST_MODEL that exposes
    scanner_spread_cost and trade_cost; the gross basis needs no cost model. The trade outcome
    journal path comes from --journal or MARKET_MOMENTUM_JOURNAL; the action value that marks an
    opened trade row comes from MARKET_MOMENTUM_JOURNAL_OPEN_ACTION (default TRADE_OPEN).

SAFETY
    Read-only. Never writes the journal or any configuration; the printed floor is advisory.
    Bootstrap is seeded -> reproducible.

USAGE
    python -m momentum_scanner.calibrate_min_score_live [--journal PATH] [--gate PROCEED|all]
        [--basis net_fixed_2r|net_pnl|gross_pnl] [--min-n N] [--seed S] [--iters K] [--json]
    # driver-predictiveness mode -- which of gap/relvol/float/dilution actually moves PnL:
    python -m momentum_scanner.calibrate_min_score_live --mode drivers [--bins N]
"""
import argparse
import json
import math
import os
import random
import statistics
import sys
from collections import namedtuple
from pathlib import Path

# --- constants -------------------------------------------------------------------
# One-sided 95% normal quantile: the lower confidence bound is mean - z*se, mirroring the
# ev_lcb gate semantics (we only care that the DOWNSIDE of the EV estimate clears zero).
Z_95_ONE_SIDED = 1.645
DEFAULT_MIN_N = 20          # min trades in a cutoff bucket before its EV_LCB is trusted
DEFAULT_SEED = 12345        # fixed -> bootstrap is reproducible run-to-run
DEFAULT_ITERS = 2000        # bootstrap resamples
DEFAULT_BOOT_PCT = 5.0      # 5th percentile of resample means = one-sided 95% lower bound
DEFAULT_GATE = "all"        # calibrating on PROCEED alone can tune a non-discriminating subset
                            # when the catalyst gate does not separate outcomes; PROCEED stays
                            # selectable via --gate.
DEFAULT_BASIS = "net_fixed_2r"
BASES = ("net_fixed_2r", "net_pnl", "gross_pnl")

# Memoized cost-model module, loaded on first use from MARKET_MOMENTUM_COST_MODEL.
_COST_MODEL = None

# Trade outcome journal path; --journal overrides it. No default location is bundled.
DEFAULT_JOURNAL = os.environ.get("MARKET_MOMENTUM_JOURNAL")

# Action value that marks an opened trade row in the outcome journal.
ACTION = os.environ.get("MARKET_MOMENTUM_JOURNAL_OPEN_ACTION", "TRADE_OPEN")
# Terminal (closed/orphaned) rows outrank active ones; ties -> last wins. So a stray late
# pending never regresses a closed trade out of the analysis.
_STATUS_RANK = {"pending": 0, "open": 1, "closed": 2, "orphaned": 2}

Trade = namedtuple("Trade", ["score", "pnl", "ticker", "outcome", "gate_decision", "features"],
                   defaults=(None,))   # features (score-driver telemetry dict) defaults None for
                                       # pre-enrichment rows / positional callers


# --- ingest ----------------------------------------------------------------------

def read_journal(path):
    """Parse the trade outcome journal (JSONL) into row dicts. Missing file -> []; bad lines skipped.
    utf-8-sig tolerates a stray BOM without crashing (defensive; the writer emits plain utf-8)."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for ln in p.read_text(encoding="utf-8-sig").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


def _is_num(v):
    """Real, finite numeric -- rejects None, strings, bool (an int subclass), NaN, +/-inf."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def dedup_terminal(rows):
    """Collapse the append-only journal to one row per id, terminal-status-wins (last on ties)."""
    state = {}
    for t in rows:
        if t.get("action") != ACTION:
            continue
        rid = t.get("id")
        if not rid:
            continue
        prev = state.get(rid)
        if prev is None or _STATUS_RANK.get(t.get("status"), 0) >= _STATUS_RANK.get(prev.get("status"), 0):
            state[rid] = t
    return state


def cost_model():
    """Load an explicitly configured cost-model module; no bundled production algorithm."""
    global _COST_MODEL
    if _COST_MODEL is None:
        module_name = os.environ.get("MARKET_MOMENTUM_COST_MODEL", "").strip()
        if not module_name:
            raise RuntimeError("MARKET_MOMENTUM_COST_MODEL must be explicitly configured")
        import importlib
        model = importlib.import_module(module_name)
        if not all(callable(getattr(model, name, None))
                   for name in ("scanner_spread_cost", "trade_cost")):
            raise RuntimeError("configured cost model must provide both cost callbacks")
        _COST_MODEL = model
    return _COST_MODEL


def default_cost(row):
    """Round-trip cost for one scanner row from the configured cost model: the row's measured
    spread cost (scanner_spread_cost) first, else the flat trade_cost fallback."""
    cm = cost_model()
    cost = cm.scanner_spread_cost(row)
    if cost is None:
        cost = cm.trade_cost(row)
    return float(cost)


def basis_value(row, basis=DEFAULT_BASIS, cost_fn=None):
    """The per-trade number the sweep should optimize, or None when this row cannot supply it.

    None is reserved for UNRESOLVED/UNKNOWN and is never coerced to 0.0. A pending_replay
    row (no fixed-2R outcome yet) counted as zero would drag EV toward zero while inflating
    n, i.e. manufacture false precision."""
    if basis not in BASES:
        raise ValueError("unknown basis %r; expected one of %s" % (basis, ", ".join(BASES)))
    if basis == "gross_pnl":
        raw = row.get("pnl")
        return float(raw) if _is_num(raw) else None
    raw = row.get("pnl_fixed_2r") if basis == "net_fixed_2r" else row.get("pnl")
    if not _is_num(raw):
        return None
    cost = (cost_fn or default_cost)(row)
    if not _is_num(cost):
        return None                                     # unknowable cost -> unknowable net
    return round(float(raw) - float(cost), 2)


def extract_scored_trades(rows, *, gate=DEFAULT_GATE, basis=DEFAULT_BASIS, cost_fn=None):
    """Closed rows carrying a numeric score and a resolvable `basis` value, deduped by id.
    gate='all' (default) keeps every verdict; gate='PROCEED' restricts to the catalyst-passing
    cohort. Rows whose basis value is None are dropped here and counted by count_unresolved()."""
    out = []
    for t in dedup_terminal(rows).values():
        if t.get("status") != "closed":
            continue
        if t.get("paper") is not True:
            continue  # fail-CLOSED: only provably-PAPER rows calibrate the floor
                      # (non-paper rows and unmarked rows are both excluded)
        if not _is_num(t.get("score")):
            continue
        if gate != "all" and t.get("gate_decision") != gate:
            continue
        v = basis_value(t, basis=basis, cost_fn=cost_fn)
        if v is None:
            continue
        out.append(Trade(score=float(t["score"]), pnl=v, ticker=t.get("ticker"),
                         outcome=t.get("outcome"), gate_decision=t.get("gate_decision"),
                         features=t.get("features")))
    return out


def count_unresolved(rows, *, gate=DEFAULT_GATE, basis=DEFAULT_BASIS, cost_fn=None):
    """Closed, scored, in-gate rows whose basis value could not be resolved. Surfaced beside
    n_trades so a thin sample can never be mistaken for a complete one."""
    n = 0
    for t in dedup_terminal(rows).values():
        if t.get("status") != "closed" or t.get("paper") is not True:
            continue
        if not _is_num(t.get("score")):
            continue
        if gate != "all" and t.get("gate_decision") != gate:
            continue
        if basis_value(t, basis=basis, cost_fn=cost_fn) is None:
            n += 1
    return n


# --- score-cap cohort contract ---------------------------------------------------

class CohortContractError(ValueError):
    """The journal's `score_cap_value` cohort labeling REFUSES this analysis.

    A floor swept across a cap boundary (or over malformed cap stamps) is calibrated on a
    measurement that changed meaning mid-sample. Every refusal names its rule and the
    offending cohort sizes; the CLI surfaces it as REFUSED, exit 2 -- never as an advisory
    floor."""


def partition_cap_cohort(rows):
    """Partition the CALIBRATION POPULATION (deduped closed paper row dicts) by
    `score_cap_value` (six pinned cases):

      1. key absent on ALL rows      -> proceed, ('uncapped', None) -- e.g. a pre-cap journal
      2. ONE identical finite value  -> proceed, ('capped@V', V); the floor<=V refusal is
                                        enforced in calibrate() where the floor exists
      3. mixed stamped + unstamped   -> REFUSE, both counts
      4. key present with null       -> MALFORMED (the producer never writes null), rows named
      5. non-numeric/non-finite/<=0  -> MALFORMED, rows named
      6. multiple distinct caps      -> REFUSE, ALL distinct values with per-value counts
    """
    absent = 0
    stamped = 0
    null_rows, bad_rows = [], []
    value_counts = {}
    for t in rows:
        if "score_cap_value" not in t:
            absent += 1
            continue
        v = t["score_cap_value"]
        if v is None:
            null_rows.append(t.get("id"))
            continue
        if not _is_num(v) or v <= 0:
            bad_rows.append(t.get("id"))
            continue
        stamped += 1
        key = float(v)
        value_counts[key] = value_counts.get(key, 0) + 1
    if null_rows:
        raise CohortContractError(
            "rule 4 (score_cap_value=null is MALFORMED; the producer never stamps null): "
            "%d row(s): %s" % (len(null_rows), null_rows))
    if bad_rows:
        raise CohortContractError(
            "rule 5 (non-numeric/non-finite/non-positive score_cap_value is MALFORMED): "
            "%d row(s): %s" % (len(bad_rows), bad_rows))
    if len(value_counts) > 1:
        detail = ", ".join("%g x%d" % (v, n) for v, n in sorted(value_counts.items()))
        raise CohortContractError(
            "rule 6 (multiple distinct caps in one pool): %s -- refusing to sweep across "
            "cap eras" % detail)
    if stamped and absent:
        raise CohortContractError(
            "rule 3 (mixed capped/uncapped cohort): %d stamped, %d unstamped -- refusing to "
            "calibrate across the cap activation boundary" % (stamped, absent))
    if not stamped:
        return "uncapped", None
    cap = next(iter(value_counts))
    return "capped@%g" % cap, cap


def _calibration_population(rows):
    """The rows the floor analysis draws from: deduped, terminal-status closed, provably paper."""
    return [t for t in dedup_terminal(rows).values()
            if t.get("status") == "closed" and t.get("paper") is True]


def _enforce_floor_below_cap(floor, cap, n_capped):
    """Rule 2's ENFORCED half: a recommended floor above the active cap can never admit a row
    (min(E, cap) <= cap < floor) -- refuse it rather than print it as advisory."""
    if cap is not None and floor is not None and floor > cap:
        raise CohortContractError(
            "rule 2 (floor above the active cap): recommended floor %.4f > cap %g on a "
            "capped cohort of %d row(s) -- such a floor rejects every row; refusing"
            % (floor, cap, n_capped))


# --- EV lower-confidence bounds --------------------------------------------------

def ev_lcb_normal(pnls, *, z=Z_95_ONE_SIDED):
    """Normal-approx one-sided lower bound on mean PnL: mean - z*sd/sqrt(n). Needs n>=2 for a
    sample stdev; below that the bound is undefined -> -inf (never 'confident')."""
    n = len(pnls)
    if n < 2:
        return float("-inf")
    mean = sum(pnls) / n
    sd = statistics.stdev(pnls)            # ddof=1
    return mean - z * sd / math.sqrt(n)


def _percentile(sorted_vals, pct):
    """Linear-interpolated percentile of an already-sorted list."""
    if not sorted_vals:
        return float("-inf")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return sorted_vals[int(k)]
    return sorted_vals[lo] * (hi - k) + sorted_vals[hi] * (k - lo)


def ev_lcb_bootstrap(pnls, *, iters=DEFAULT_ITERS, seed=DEFAULT_SEED, pct=DEFAULT_BOOT_PCT):
    """Bootstrap one-sided lower bound: the `pct`-th percentile of resampled mean PnL. More honest
    than the normal approx for small, skewed per-trade PnL samples (capped upside, stopped downside).
    Seeded -> deterministic. n<2 is degenerate (returned as-is; recommendation's min_n guards it)."""
    n = len(pnls)
    if n == 0:
        return float("-inf")
    if n == 1:
        return float(pnls[0])
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(pnls, k=n)) / n for _ in range(iters))
    return _percentile(means, pct)


# --- sweep + recommend -----------------------------------------------------------

def sweep(trades, *, min_n=DEFAULT_MIN_N, seed=DEFAULT_SEED, iters=DEFAULT_ITERS):
    """One row per distinct observed score (ascending). Each row summarizes the subset
    {trades with score >= threshold}: n, win_rate, EV/trade, both EV_LCBs, total PnL. Thresholds
    are the observed scores themselves, so `score >= threshold` compares exact values (no grid snap).
    `min_n` is accepted for API symmetry; it is applied in recommend(), not here."""
    thresholds = sorted({t.score for t in trades})
    rows = []
    for f in thresholds:
        pnls = [t.pnl for t in trades if t.score >= f]
        n = len(pnls)
        wins = sum(1 for x in pnls if x > 0)
        rows.append({
            "threshold": f,
            "n": n,
            "win_rate": (wins / n) if n else float("nan"),
            "ev": (sum(pnls) / n) if n else float("nan"),
            "ev_lcb_normal": ev_lcb_normal(pnls),
            "ev_lcb_boot": ev_lcb_bootstrap(pnls, iters=iters, seed=seed),
            "total_pnl": sum(pnls),
        })
    return rows


def recommend(sweep_rows, *, min_n=DEFAULT_MIN_N, metric="ev_lcb_boot"):
    """Lowest-threshold row (rows are ascending) whose `metric` > 0 at n >= min_n. Maximizes
    trade volume subject to a confidently-positive EV. None + reason if no cutoff qualifies."""
    if not sweep_rows:
        return {"floor": None, "n": None, "ev": None, "ev_lcb": None,
                "reason": "no scored trades to sweep -- accrue closed paper-trade rows first"}
    for r in sweep_rows:
        if r["n"] >= min_n and r[metric] > 0:
            return {"floor": r["threshold"], "n": r["n"], "ev": r["ev"], "ev_lcb": r[metric],
                    "reason": "lowest cutoff with %s>0 at n>=%d" % (metric, min_n)}
    eligible = [r for r in sweep_rows if r["n"] >= min_n]
    best = max(eligible, key=lambda r: r[metric]) if eligible else None
    detail = ("best achievable %s=%.4f at threshold %.4f (n=%d)"
              % (metric, best[metric], best["threshold"], best["n"])) if best else \
             ("no cutoff reaches n>=%d" % min_n)
    return {"floor": None, "n": None, "ev": None, "ev_lcb": None,
            "reason": "no cutoff has positive %s at n>=%d -- %s" % (metric, min_n, detail)}


def calibrate(rows_or_path, *, min_n=DEFAULT_MIN_N, seed=DEFAULT_SEED, gate=DEFAULT_GATE,
              iters=DEFAULT_ITERS, basis=DEFAULT_BASIS, cost_fn=None):
    """End-to-end: rows (or a journal path) -> recommended floor + full sweep + score distribution.
    Zero scored trades is an EXPECTED state (returns floor=None, n_trades=0), never an error.
    `basis` and `n_unresolved` are always reported so a reader can never mistake which exit
    and which cost regime a floor was derived under."""
    rows = read_journal(rows_or_path) if isinstance(rows_or_path, (str, Path)) else list(rows_or_path)
    population = _calibration_population(rows)
    cohort, cap_value = partition_cap_cohort(population)      # refuses impure pools
    trades = extract_scored_trades(rows, gate=gate, basis=basis, cost_fn=cost_fn)
    unresolved = count_unresolved(rows, gate=gate, basis=basis, cost_fn=cost_fn)
    if not trades:
        return {"floor": None, "n_trades": 0, "gate": gate, "basis": basis,
                "cohort": cohort, "cap_value": cap_value,
                "n_unresolved": unresolved,
                "reason": ("no closed rows carry a resolvable %s value yet -- accrue "
                           "closed paper-trade rows first (min_score_live=None does not "
                           "block paper accrual)" % basis),
                "recommendation": None, "sweep": [], "scores": {}}
    rows_sweep = sweep(trades, min_n=min_n, seed=seed, iters=iters)
    rec = recommend(rows_sweep, min_n=min_n)
    _enforce_floor_below_cap(rec["floor"], cap_value, len(population))   # rule 2, enforced half
    scores = [t.score for t in trades]
    return {
        "floor": rec["floor"], "n_trades": len(trades), "gate": gate, "basis": basis,
        "cohort": cohort, "cap_value": cap_value,
        "n_unresolved": unresolved, "reason": rec["reason"],
        "recommendation": rec, "sweep": rows_sweep,
        "scores": {"min": min(scores), "max": max(scores), "median": statistics.median(scores)},
    }


# --- driver-bucketing analysis: which score-driver actually predicts PnL ---------
# Numeric drivers -> quantile buckets + Spearman rank-correlation with PnL. Categorical drivers ->
# per-value EV groups + best-vs-worst EV spread. Answers "should the producer score weight
# gap/relvol/float/dilution more?" -- reads the `features` telemetry stamped on every trade row.
_NUMERIC_FEATURES = ("gap_pct", "rel_volume", "float_shares", "premarket_volume",
                     "baseline_volume", "edgar_signal")
_CATEGORICAL_FEATURES = ("dilution_flag", "catalyst_content_source")


def _rank(vals):
    """Fractional (tie-averaged, 1-based) ranks -- the rank transform behind Spearman."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0                    # average 1-based rank across the tie run
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx == 0 or sy == 0:
        return 0.0
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    return cov / math.sqrt(sx * sy)


def _spearman(xs, ys):
    """Spearman rank correlation in [-1, 1]. 0.0 if <3 points or either variable is constant --
    a nonparametric 'does this driver move monotonically with PnL' signal, robust to fat tails."""
    if len(xs) != len(ys) or len(xs) < 3:
        return 0.0
    return _pearson(_rank(xs), _rank(ys))


def _feat(trade, key):
    return trade.features.get(key) if trade.features else None


def bucket_numeric(trades, key, *, bins=4, seed=DEFAULT_SEED, iters=DEFAULT_ITERS):
    """Quantile-bucket trades by a numeric driver; per-bucket n/win_rate/EV/EV_LCB + value range,
    plus the Spearman rho of driver-vs-PnL over all trades carrying that driver. Trades missing a
    numeric value for `key` are excluded (counted)."""
    pairs = [(_feat(t, key), t.pnl) for t in trades if _is_num(_feat(t, key))]
    excluded = len(trades) - len(pairs)
    pairs.sort(key=lambda p: p[0])
    n = len(pairs)
    buckets = []
    k = min(bins, n) if n else 0
    for b in range(k):
        chunk = pairs[b * n // k:(b + 1) * n // k]
        if not chunk:
            continue
        pnls = [p for _, p in chunk]
        buckets.append({"lo": chunk[0][0], "hi": chunk[-1][0], "n": len(pnls),
                        "win_rate": sum(1 for x in pnls if x > 0) / len(pnls),
                        "ev": sum(pnls) / len(pnls),
                        "ev_lcb": ev_lcb_bootstrap(pnls, iters=iters, seed=seed)})
    rho = _spearman([v for v, _ in pairs], [p for _, p in pairs])
    return {"key": key, "kind": "numeric", "buckets": buckets, "excluded": excluded, "spearman": rho}


def group_categorical(trades, key, *, min_n=DEFAULT_MIN_N, seed=DEFAULT_SEED, iters=DEFAULT_ITERS):
    """Group trades by a categorical driver value (None is a real 'clean' category); per-group
    n/win_rate/EV/EV_LCB, sorted EV-desc. `spread` = best-minus-worst EV over groups with n>=min_n
    (a robust effect size). Rows with NO features dict at all are excluded (counted)."""
    groups, excluded = {}, 0
    for t in trades:
        if t.features is None:
            excluded += 1
            continue
        groups.setdefault(t.features.get(key), []).append(t.pnl)
    out = []
    for v, pnls in groups.items():
        out.append({"value": v, "n": len(pnls),
                    "win_rate": sum(1 for x in pnls if x > 0) / len(pnls),
                    "ev": sum(pnls) / len(pnls),
                    "ev_lcb": ev_lcb_bootstrap(pnls, iters=iters, seed=seed)})
    out.sort(key=lambda g: g["ev"], reverse=True)
    evs = [g["ev"] for g in out if g["n"] >= min_n]
    spread = (max(evs) - min(evs)) if len(evs) >= 2 else 0.0
    return {"key": key, "kind": "categorical", "groups": out, "excluded": excluded, "spread": spread}


def analyze_drivers(rows_or_path, *, gate=DEFAULT_GATE, bins=4, min_n=DEFAULT_MIN_N,
                    seed=DEFAULT_SEED, iters=DEFAULT_ITERS, basis=DEFAULT_BASIS, cost_fn=None):
    """Per-driver predictiveness: numeric drivers ranked by |Spearman rho|, categorical by |EV
    spread|. Reads the `features` telemetry off closed rows. Zero trades -> clean empty result.
    Shares `basis`/`cost_fn` with calibrate() so a driver is never ranked against one exit model
    while the floor is chosen under another."""
    rows = read_journal(rows_or_path) if isinstance(rows_or_path, (str, Path)) else list(rows_or_path)
    cohort, cap_value = partition_cap_cohort(_calibration_population(rows))   # same ingest contract
    trades = extract_scored_trades(rows, gate=gate, basis=basis, cost_fn=cost_fn)
    numeric = [bucket_numeric(trades, k, bins=bins, seed=seed, iters=iters) for k in _NUMERIC_FEATURES]
    numeric.sort(key=lambda d: abs(d["spearman"]), reverse=True)
    categorical = [group_categorical(trades, k, min_n=min_n, seed=seed, iters=iters)
                   for k in _CATEGORICAL_FEATURES]
    categorical.sort(key=lambda d: abs(d["spread"]), reverse=True)
    return {"n_trades": len(trades), "gate": gate, "basis": basis, "cohort": cohort,
            "cap_value": cap_value, "numeric": numeric, "categorical": categorical}


def format_drivers_report(result):
    lines = ["driver predictiveness -- gate=%s, basis=%s, %d scored closed trade(s)"
             % (result["gate"], result.get("basis", "?"), result["n_trades"])]
    if not result["n_trades"]:
        lines.append("  no scored closed trades yet -- accrue closed paper-trade rows first")
        return "\n".join(lines)
    lines.append("")
    lines.append("NUMERIC drivers (ranked by |Spearman rho| of driver vs PnL):")
    for d in result["numeric"]:
        lines.append("  %-22s rho=%+.3f  (excluded %d)" % (d["key"], d["spearman"], d["excluded"]))
        for b in d["buckets"]:
            lines.append("      [%10.4g .. %-10.4g] n=%3d  win%%=%5.1f  EV=%8.2f  EV_LCB=%8.2f"
                         % (b["lo"], b["hi"], b["n"], 100 * b["win_rate"], b["ev"], b["ev_lcb"]))
    lines.append("")
    lines.append("CATEGORICAL drivers (ranked by best-vs-worst EV spread):")
    for d in result["categorical"]:
        lines.append("  %-22s spread=%8.2f  (excluded %d)" % (d["key"], d["spread"], d["excluded"]))
        for g in d["groups"]:
            label = str(g["value"]) if g["value"] is not None else "(none)"
            lines.append("      %-18s n=%3d  win%%=%5.1f  EV=%8.2f  EV_LCB=%8.2f"
                         % (label, g["n"], 100 * g["win_rate"], g["ev"], g["ev_lcb"]))
    lines.append("")
    lines.append("NOTE: read-only. High |rho| / large EV spread => the driver separates winners from"
                 " losers => consider weighting it more in the producer score.")
    return "\n".join(lines)


# --- CLI -------------------------------------------------------------------------

def _json_safe(obj):
    """Strict-JSON copy of a result: every non-finite float (negative infinity for an undefined lower
    bound, NaN) becomes null, because JSON has no such numbers. The human report still shows them."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _dump_json(result):
    return json.dumps(_json_safe(result), indent=2, allow_nan=False)


def _fmt(x):
    if x == float("-inf"):
        return "  -inf"
    if isinstance(x, float) and math.isnan(x):
        return "   nan"
    return "%7.3f" % x


def format_report(result):
    lines = []
    lines.append("min_score_live calibration -- gate=%s, basis=%s, cohort=%s, %d scored closed trade(s)"
                 % (result["gate"], result.get("basis", "?"), result.get("cohort", "?"),
                    result["n_trades"]))
    if result.get("n_unresolved"):
        lines.append("  %d in-gate row(s) EXCLUDED: no resolvable %s value (never counted as zero)"
                     % (result["n_unresolved"], result.get("basis", "?")))
    if result.get("basis") == "gross_pnl":
        lines.append("  WARNING: gross_pnl is the LEGACY basis -- uncosted, on the scale_trail "
                     "exit. Do not set a floor from it.")
    if not result["sweep"]:
        lines.append("  " + result["reason"])
        return "\n".join(lines)
    sc = result["scores"]
    lines.append("score distribution: min=%.4f  median=%.4f  max=%.4f"
                 % (sc["min"], sc["median"], sc["max"]))
    lines.append("")
    lines.append("  cutoff       n   win%     EV/trade  EV_LCB(boot)  EV_LCB(norm)   total_pnl")
    lines.append("  " + "-" * 74)
    for r in result["sweep"]:
        lines.append("  %6.3f  %6d  %5.1f  %s  %s  %s  %10.2f"
                     % (r["threshold"], r["n"], 100.0 * r["win_rate"], _fmt(r["ev"]),
                        _fmt(r["ev_lcb_boot"]), _fmt(r["ev_lcb_normal"]), r["total_pnl"]))
    lines.append("")
    if result["floor"] is not None:
        rec = result["recommendation"]
        lines.append("RECOMMENDED min_score_live = %.4f  (n=%d, EV/trade=%.2f, EV_LCB=%.3f)"
                     % (result["floor"], rec["n"], rec["ev"], rec["ev_lcb"]))
        lines.append("  %s" % result["reason"])
    else:
        lines.append("RECOMMENDED min_score_live = (none yet)")
        lines.append("  %s" % result["reason"])
    lines.append("")
    lines.append("NOTE: advisory only. Apply configuration changes only after review.")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Recommend a min_score_live floor from closed paper-trade outcomes.")
    ap.add_argument("--journal", default=DEFAULT_JOURNAL,
                    help="path to the trade outcome journal (JSONL)")
    ap.add_argument("--gate", default=DEFAULT_GATE, choices=["PROCEED", "all"],
                    help="verdict population to analyze (default all)")
    ap.add_argument("--basis", default=DEFAULT_BASIS, choices=list(BASES),
                    help="per-trade value to optimize (default net_fixed_2r = the fixed-2R "
                         "exit net of measured spread; gross_pnl is the uncosted legacy basis)")
    ap.add_argument("--min-n", type=int, default=DEFAULT_MIN_N, dest="min_n")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable JSON instead of a table (strict JSON: a value that "
                         "is undefined, such as an EV lower bound from one trade, is null)")
    ap.add_argument("--mode", choices=["calibrate", "drivers"], default="calibrate",
                    help="calibrate = recommend a min_score_live floor; drivers = per-driver "
                         "predictiveness (which of gap/relvol/float/dilution actually moves PnL)")
    ap.add_argument("--bins", type=int, default=4, help="quantile buckets per numeric driver (drivers mode)")
    args = ap.parse_args(argv)
    if not args.journal:
        ap.error("--journal or MARKET_MOMENTUM_JOURNAL must be explicitly configured")
    if args.basis != "gross_pnl":
        cost_model()  # Resolve adapter before journal file I/O.
    try:
        if args.mode == "drivers":
            result = analyze_drivers(args.journal, gate=args.gate, bins=args.bins, min_n=args.min_n,
                                     seed=args.seed, iters=args.iters, basis=args.basis)
            print(_dump_json(result) if args.json else format_drivers_report(result))
            return 0
        result = calibrate(args.journal, min_n=args.min_n, seed=args.seed, gate=args.gate,
                           iters=args.iters, basis=args.basis)
    except CohortContractError as exc:
        # An impure/malformed score_cap_value cohort is a REFUSAL, never an advisory
        # floor. Loud, named rule, nonzero exit -- the operator fixes the pool, not the floor.
        print("REFUSED (score-cap cohort contract): %s" % exc)
        return 2
    if args.json:
        print(_dump_json(result))
    else:
        print(format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

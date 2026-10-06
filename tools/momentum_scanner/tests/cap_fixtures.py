"""Deterministic fixture cohort for the score-cap test suite.

One shared cohort builder used by BOTH:
  - the committed golden-byte OFF-identity artifact (tests/golden/watchlist_off_golden.json),
    generated from the pre-cap code and compared byte-for-byte forever after, and
  - the ON-vs-OFF ordering/value tests.

The cohort is >20 rows with multiple tail rows (raw effective score far above the default cap),
several hump-band rows, sub-1 rows, an exact-tie pair (confidence tiebreak), a pump-label row,
and two unknown-float rows whose raw scores sit BELOW every known row's score so a stable global
re-sort by stamped score reproduces producer order exactly.

Everything here is pinned: fixed scan_id, fixed scanned_at, fixed quote strings. No randomness,
no clocks. All builders return FRESH objects each call (rank_candidates mutates in place).
"""
from datetime import datetime, timezone

from momentum_scanner.catalyst_evaluator import CatalystVerdict
from momentum_scanner.movers_scan import MoverCandidate

FIXED_SCANNED_AT = datetime(2026, 8, 11, 13, 30, 0, tzinfo=timezone.utc)
FIXED_SCAN_ID = "cap-golden-0001"
FIXED_SEEN_AT = "2026-08-11T13:29:50+00:00"

# (ticker, gap_pct, rel_volume, float_shares, edgar_signal,
#  decision, label, confidence, content_source, dilution_flag)
# Raw effective scores as computed by the pre-change formula (verified at golden generation):
#   tail rows:  T1 2000.18, T2 500.17, T3 60.112, T4 12.0, T5 9.105, T6 8.3, T7 8.072
#               (all strictly above the 8.0 default cap; catalyst bonuses included)
#   hump band:  H1 7.176, H2 5.0, H8 3.57252 (shelf_risk 0.7), H9 3.126, H3 3.0994,
#               H7 2.654 (float 30M -> factor 0.625), H4 2.0, H5 1.57, H6 1.15 (edgar)
#   sub-1:      L1 0.5, L2 0.4, P1 0.35, L3 0.3, L4/L5 0.2 tie (confidence 0.9 vs 0.5 breaks it)
#   unknown:    U1 0.06, U2 0.05 (below every known row -> re-sort keeps them last)
_SPEC = [
    ("T1", 5.00, 400.0, 5_000_000, 0.0, "PROCEED", "real", 0.90, "edgar", None),
    ("T2", 2.00, 250.0, 5_000_000, 0.0, "PROCEED", "real", 0.85, "edgar", None),
    ("T3", 1.50, 40.0, 5_000_000, 0.0, "PROCEED", "real", 0.80, "finnhub", None),
    ("T4", 0.60, 20.0, 5_000_000, 0.0, "REJECT", "no-data", 0.30, None, None),
    ("T5", 0.45, 20.0, 5_000_000, 0.0, "PROCEED", "real", 0.75, "finnhub", None),
    ("T6", 0.415, 20.0, 5_000_000, 0.0, "REJECT", "no-data", 0.20, None, None),
    ("T7", 0.40, 20.0, 5_000_000, 0.0, "PROCEED", "real", 0.72, "web", None),
    ("H1", 0.35, 20.0, 5_000_000, 0.0, "PROCEED", "real", 0.88, "edgar", None),
    ("H2", 0.25, 20.0, 5_000_000, 0.0, "REJECT", "avoid", 0.60, None, None),
    ("H3", 0.30, 10.0, 5_000_000, 0.0, "PROCEED", "real", 0.71, "finnhub", None),
    ("H4", 0.20, 10.0, 5_000_000, 0.0, "REJECT", "no-data", 0.10, None, None),
    ("H5", 0.15, 10.0, 5_000_000, 0.0, "PROCEED", "real", 0.70, "web", None),
    ("H6", 0.10, 10.0, 5_000_000, 0.5, "REJECT", "no-data", 0.15, None, None),
    ("H7", 0.40, 10.0, 30_000_000, 0.0, "PROCEED", "real", 0.77, "edgar", None),
    ("H8", 0.50, 10.0, 5_000_000, 0.0, "PROCEED", "real", 0.74, "finnhub", "shelf_risk"),
    ("H9", 0.30, 10.0, 5_000_000, 0.0, "PROCEED", "real", 0.90, "finnhub", None),
    ("L1", 0.10, 5.0, 5_000_000, 0.0, "REJECT", "no-data", 0.25, None, None),
    ("L2", 0.08, 5.0, 5_000_000, 0.0, "REJECT", "no-data", 0.22, None, None),
    ("L3", 0.06, 5.0, 5_000_000, 0.0, "REJECT", "no-data", 0.21, None, None),
    ("L4", 0.05, 4.0, 5_000_000, 0.0, "REJECT", "no-data", 0.90, None, None),
    ("L5", 0.05, 4.0, 5_000_000, 0.0, "REJECT", "no-data", 0.50, None, None),
    ("P1", 0.07, 5.0, 5_000_000, 0.0, "REJECT", "pump", 0.95, "finnhub", None),
    ("U1", 0.05, 1.2, None, 0.0, "REJECT", "no-data", 0.12, None, None),
    ("U2", 0.05, 1.0, None, 0.0, "REJECT", "no-data", 0.11, None, None),
]

# Note on catalyst_bonus: it needs BOTH confidence and content_source non-None, a "real" label,
# and a nonzero SOURCE_WEIGHT to be nonzero. H9 (0.90 * finnhub 0.7 * W_CATALYST 0.2 = 0.126) is
# the only row designed to carry a bonus that moves its effective score; P1 pins the pump-label
# zero-bonus rule inside the cohort.

HUMP_TICKERS = ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9")
TAIL_TICKERS = ("T1", "T2", "T3", "T4", "T5", "T6", "T7")   # raw score strictly above the 8.0 default cap


def build_cohort():
    """Return (candidates, verdicts): 24 fresh MoverCandidates + a full verdict map (every
    candidate judged, so every candidate is written -- unjudged-absent is pinned elsewhere)."""
    candidates = []
    verdicts = {}
    for (ticker, gap, relvol, fs, edgar, decision, label, conf, src, dil) in _SPEC:
        c = MoverCandidate(
            ticker=ticker, company_name=ticker, premarket_price=5.0, prev_close=4.0,
            gap_pct=gap, float_shares=fs, premarket_volume=1_500_000, baseline_volume=500_000,
            rel_volume=relvol, day_high=5.0, new_hod=True, scan_mode="intraday",
            last_seen_at=FIXED_SEEN_AT)
        c.edgar_signal = edgar
        candidates.append(c)
        verdicts[ticker] = CatalystVerdict(
            decision, label, "fixture reason %s" % ticker, conf, "llm",
            headline="fixture headline %s" % ticker, dilution_flag=dil, content_source=src)
    return candidates, verdicts


def write_off_watchlist(out_path):
    """Rank + serialize the fixture cohort through the real writer chain and atomically write it
    to out_path. Callers are responsible for the score-cap env state (the golden was generated
    with both SCANNER_SCORE_CAP vars UNSET, i.e. cap OFF)."""
    from momentum_scanner import movers_watchlist, watchlist_io
    from momentum_scanner.ranking import rank_candidates

    candidates, verdicts = build_cohort()
    ranked = rank_candidates(candidates, verdicts=verdicts)
    payload = movers_watchlist.build_movers_watchlist_dict(
        ranked, verdicts, scan_id=FIXED_SCAN_ID, scanned_at=FIXED_SCANNED_AT)
    watchlist_io.atomic_write_json(str(out_path), payload)
    return payload

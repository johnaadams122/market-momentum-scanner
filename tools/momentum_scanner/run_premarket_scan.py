#!/usr/bin/env python3
"""
Premarket momentum scanner for micro-pullback candidates.

Usage:
    python -m momentum_scanner
    python -m momentum_scanner --lookback 30
    python -m momentum_scanner --json-out path/to/watchlist.json
    (legacy) python run_premarket_scan.py --json-out path/to/watchlist.json

For each numeric-qualified candidate it runs the fail-closed catalyst gate
(real/pump/no-data -> PROCEED/REJECT) and, with --json-out, writes an atomic
watchlist.json (schema v2) consumed by downstream tools.
"""
import argparse
import json
import os
import sys
import tempfile
import uuid
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from momentum_scanner import config
from momentum_scanner import watchlist_io
from momentum_scanner.edgar_rss import _load_ticker_map, fetch_recent_8k_filings
from momentum_scanner.gap_scanner import scan_candidates
from momentum_scanner.catalyst_evaluator import evaluate, CatalystVerdict
from momentum_scanner.config import WATCHLIST_TTL_MINUTES
from momentum_scanner.errors import ScannerNetworkError


def build_watchlist_dict(candidates, filings, verdicts, *, scan_id=None, scanned_at=None, status="ok",
                         confirm_overflow=0):
    """Assemble the watchlist.json v2 payload. `verdicts` maps ticker -> CatalystVerdict;
    a missing verdict defaults to a fail-closed REJECT/no-data. `confirm_overflow` is the count of
    trigger tickers dropped at the Finnhub confirm-fan-out cap (0 on the EDGAR path)."""
    by_ticker = {}
    for f in filings:
        if f.ticker and f.ticker not in by_ticker:
            by_ticker[f.ticker] = f.headline
    cands = []
    for c in candidates:
        d = asdict(c)
        d["catalyst"] = by_ticker.get(c.ticker)
        watchlist_io.stamp_verdict(d, verdicts.get(c.ticker))
        cands.append(d)
    return watchlist_io.v2_envelope(cands, scan_id=scan_id, scanned_at=scanned_at, status=status,
                                    confirm_overflow=confirm_overflow)


def write_watchlist_json(path, candidates, filings, verdicts, *, scan_id=None, scanned_at=None, status="ok",
                         confirm_overflow=0):
    """Atomically write watchlist.json (temp file in the same dir + os.replace)."""
    payload = build_watchlist_dict(candidates, filings, verdicts,
                                   scan_id=scan_id, scanned_at=scanned_at, status=status,
                                   confirm_overflow=confirm_overflow)
    watchlist_io.atomic_write_json(path, payload)


def _gate_candidates(candidates, filings):
    """Run the catalyst gate per candidate, isolating per-candidate failures into
    a fail-closed REJECT so one bad ticker never aborts the batch."""
    filing_by_ticker = {f.ticker: f for f in filings if f.ticker}
    verdicts = {}
    for c in candidates:
        f = filing_by_ticker.get(c.ticker)
        if f is None:
            verdicts[c.ticker] = CatalystVerdict("REJECT", "no-data", "no matching filing", None, "rule")
            continue
        try:
            verdicts[c.ticker] = evaluate(c, f)
        except Exception as exc:  # last-resort isolation; evaluate is already fail-closed
            verdicts[c.ticker] = CatalystVerdict("REJECT", "no-data", f"evaluator error: {exc}", None, "rule")
    return verdicts


def _error_exit(json_out, msg) -> None:
    print(f"FATAL: {msg}")
    if json_out:
        write_watchlist_json(json_out, [], [], {}, status="error")
    sys.exit(1)


def _gate_news(candidates, news_by_ticker, now_utc):
    """Gate the Finnhub path -- per candidate, evaluate_news over its confirmed same-window items.
    Per-candidate isolation -> a single bad candidate becomes a REJECT, never aborts the batch."""
    from momentum_scanner.catalyst_evaluator import evaluate_news
    verdicts = {}
    for c in candidates:
        try:
            verdicts[c.ticker] = evaluate_news(c, news_by_ticker.get(c.ticker, []), now=now_utc)
        except Exception as exc:
            verdicts[c.ticker] = CatalystVerdict("REJECT", "no-data", f"evaluator error: {exc}", None, "rule")
    return verdicts


def _print_and_write(args, candidates, catalyst_filings, verdicts, *, confirm_overflow=0) -> None:
    if confirm_overflow:
        print(f"  WARNING: {confirm_overflow} trigger ticker(s) dropped at the Finnhub confirm cap "
              f"(surfaced as confirm_overflow in the watchlist).")
    if candidates:
        print(f"\n{'='*60}\nCANDIDATES ({len(candidates)}) -- gate verdict\n{'='*60}")
        for c in candidates:
            v = verdicts.get(c.ticker)
            float_str = f"{c.float_shares:,}" if c.float_shares else "N/A"
            relvol_str = f"{c.rel_volume}x" if c.rel_volume else "N/A"
            decision = v.decision if v else "REJECT"
            label = v.label if v else "no-data"
            print(f"  {c.ticker:6s} | {decision:7s} [{label}] | gap={c.gap_pct:.1%} | "
                  f"price=${c.premarket_price:.2f} | float={float_str} | relvol={relvol_str}")
            if v:
                print(f"         reason: {v.reason}")
    else:
        print("No candidates after numeric filters.")
    if args.json_out:
        write_watchlist_json(args.json_out, candidates, catalyst_filings, verdicts,
                             confirm_overflow=confirm_overflow)


def _run_edgar(args) -> None:
    print("[1/3] Loading EDGAR ticker map + recent 8-K filings...")
    try:
        ticker_map = _load_ticker_map()
        filings = fetch_recent_8k_filings(lookback_minutes=args.lookback, ticker_map=ticker_map)
    except ScannerNetworkError as exc:
        _error_exit(args.json_out, f"EDGAR source unavailable: {exc}")

    best = {}
    for f in filings:
        if not f.ticker:
            continue
        if f.ticker not in best or f.filed_at > best[f.ticker].filed_at:
            best[f.ticker] = f
    rep_filings = list(best.values())
    tickers = list(best.keys())
    print(f"  {len(filings)} filings, {len(tickers)} unique tickers: {tickers or 'none'}")

    now_et = datetime.now(ET)
    try:
        candidates = scan_candidates(tickers, now_et=now_et) if tickers else []
    except ScannerNetworkError as exc:
        _error_exit(args.json_out, f"Schwab market data unavailable: {exc}")
    verdicts = _gate_candidates(candidates, rep_filings)
    _print_and_write(args, candidates, rep_filings, verdicts)


def _run_finnhub(args) -> None:
    # Live-mode gate: the Finnhub free tier is non-commercial -> refuse to run it in live mode unless a
    # commercial key is flagged. Enforced BEFORE any Finnhub call.
    if os.environ.get("SCANNER_MODE", "paper") == "live" and not config.FINNHUB_COMMERCIAL_USE:
        _error_exit(args.json_out, "Finnhub free tier is non-commercial; cannot use --source finnhub in live mode")
    from momentum_scanner.finnhub_news import FinnhubClient, aggregate_news
    now_et = datetime.now(ET)
    now_utc = datetime.now(timezone.utc)
    print("[1/3] Finnhub news-wire trigger (04:00-09:30 ET)...")
    try:
        client = FinnhubClient()
        trigger = client.fetch_trigger_news(now_et)
        agg_stats = {}
        news_by_ticker = aggregate_news(trigger, client, now_et, stats=agg_stats)
        tickers = list(news_by_ticker.keys())
        print(f"  Finnhub: {len(tickers)} confirmed ticker(s): {tickers or 'none'}")
        candidates = scan_candidates(tickers, now_et=now_et) if tickers else []
    except ScannerNetworkError as exc:
        _error_exit(args.json_out, f"Finnhub/Schwab unavailable: {exc}")
    verdicts = _gate_news(candidates, news_by_ticker, now_utc)
    # The watchlist 'catalyst' comes from the most-recent confirmed news item per ticker (NewsItem has
    # .ticker + .headline, which is all build_watchlist_dict reads).
    catalyst_filings = [max(news_by_ticker[c.ticker], key=lambda i: i.published)
                        for c in candidates if news_by_ticker.get(c.ticker)]
    _print_and_write(args, candidates, catalyst_filings, verdicts,
                     confirm_overflow=agg_stats.get("confirm_overflow", 0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Premarket momentum scanner")
    parser.add_argument("--lookback", type=int, default=60,
                        help="Minutes of EDGAR lookback window (default: 60)")
    parser.add_argument("--json-out", default=None,
                        help="If set, write watchlist.json to this path (atomic).")
    parser.add_argument("--source", choices=["edgar", "finnhub"], default="edgar",
                        help="Trigger source (default: edgar; finnhub once live-validated).")
    args = parser.parse_args()
    if args.source == "finnhub":
        _run_finnhub(args)
    else:
        _run_edgar(args)


if __name__ == "__main__":
    main()

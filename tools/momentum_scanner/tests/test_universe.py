import logging
from datetime import datetime, timezone, timedelta
from momentum_scanner import universe

NASDAQ = (
    "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\nAAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N\nMU|Micron Technology Inc. - Common Stock|Q|N|N|100|N|N\nZZZT|Nasdaq TEST|Q|Y|N|100|N|N\nQQQ|Invesco QQQ Trust - ETF|Q|N|N|100|Y|N\nABCDW|Acme Corp - Warrant|Q|N|N|100|N|N\nABCDU|Acme Corp - Unit|Q|N|N|100|N|N\nFile Creation Time: 06/29/2026 12:00|||||||\n"                         # footer -> drop
)
# otherlisted.txt has a DIFFERENT layout: ETF at col 4, Test Issue at col 6, symbol = ACT Symbol
OTHER = (
    "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\nIBM|International Business Machines - Common Stock|N|IBM|N|100|N|IBM\nSPY|SPDR S&P 500 ETF Trust|P|SPY|Y|100|N|SPY\nZTST|NYSE Test|N|ZTST|N|100|Y|ZTST\nFile Creation Time: 06/29/2026 12:00|||||||\n"
)

def test_parse_nasdaq_layout():
    syms = universe.parse_symbol_directory(NASDAQ)
    assert "AAPL" in syms and "MU" in syms
    assert "ZZZT" not in syms and "QQQ" not in syms
    assert "ABCDW" not in syms and "ABCDU" not in syms
    assert not any(s.startswith("File") for s in syms)

def test_parse_otherlisted_layout_uses_correct_columns():
    syms = universe.parse_symbol_directory(OTHER)
    assert "IBM" in syms
    assert "SPY" not in syms      # ETF col is 4 here, not 6 -- must not be misread as common stock
    assert "ZTST" not in syms

def test_parse_drops_plural_noncommon_names():
    txt = ("Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
           "WRNT|Foo Corp - Warrants|Q|N|N|100|N|N\n"
           "RGHT|Bar Corp - Rights|Q|N|N|100|N|N\n"
           "UNTS|Baz Corp - Units|Q|N|N|100|N|N\n"
           "REAL|Real Co - Common Stock|Q|N|N|100|N|N\n")
    assert universe.parse_symbol_directory(txt) == ["REAL"]

def test_parse_keeps_company_with_unit_or_preferred_in_company_name():
    """Company names containing 'Unit' or 'Preferred' before ' - Common Stock' must NOT be dropped."""
    txt = (
        "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
        "UCORP|Unit Corporation - Common Stock|Q|N|N|100|N|N\n"
        "PBNK|Preferred Bank - Common Stock|Q|N|N|100|N|N\n"
        "FADR|Foo Corp - American Depositary Shares|Q|N|N|100|N|N\n"
        "WRT|XYZ Corp - Warrant|Q|N|N|100|N|N\n"
        "UNTS|XYZ Corp - Units|Q|N|N|100|N|N\n"
        "RGHT|XYZ Corp - Rights|Q|N|N|100|N|N\n"
        "PRFD|XYZ Corp - Preferred Stock|Q|N|N|100|N|N\n"
    )
    syms = universe.parse_symbol_directory(txt)
    # Must be KEPT (descriptor is "Common Stock" or "American Depositary Shares")
    assert "UCORP" in syms, "Unit Corporation - Common Stock wrongly dropped"
    assert "PBNK" in syms, "Preferred Bank - Common Stock wrongly dropped"
    assert "FADR" in syms, "ADR wrongly dropped (ADRs are valid movers)"
    # Must be DROPPED (descriptor is the non-common type)
    assert "WRT" not in syms, "Warrant not dropped"
    assert "UNTS" not in syms, "Units not dropped"
    assert "RGHT" not in syms, "Rights not dropped"
    assert "PRFD" not in syms, "Preferred Stock not dropped"


def test_load_universe_fetch_failure_logs_warning_and_returns_stale(tmp_path, caplog):
    """A fetch failure falls back to stale cache AND logs a warning."""
    def fetch_ok(name):
        return NASDAQ if name == "nasdaqlisted.txt" else OTHER
    now = datetime(2026, 6, 29, tzinfo=timezone.utc)
    universe.load_universe(tmp_path, fetch_fn=fetch_ok, now_utc=now)

    def boom(name):
        raise RuntimeError("network down")
    later = now + timedelta(days=30)  # past refresh window -> triggers refetch
    with caplog.at_level(logging.WARNING):
        result = universe.load_universe(tmp_path, fetch_fn=boom, now_utc=later)
    assert any(r.levelno >= logging.WARNING for r in caplog.records), \
        "expected a WARNING log on stale-cache fallback"
    assert "AAPL" in result


def test_load_universe_caches_refresh_and_failclosed(tmp_path):
    calls = {"n": 0}
    def fetch_fn(name):
        calls["n"] += 1
        return NASDAQ if name == "nasdaqlisted.txt" else OTHER
    now = datetime(2026, 6, 29, tzinfo=timezone.utc)
    a = universe.load_universe(tmp_path, fetch_fn=fetch_fn, now_utc=now)
    assert "AAPL" in a and "IBM" in a
    n_after_build = calls["n"]
    # within window -> cache, no refetch
    universe.load_universe(tmp_path, fetch_fn=fetch_fn, now_utc=now + timedelta(days=1))
    assert calls["n"] == n_after_build
    # past window -> refetch
    universe.load_universe(tmp_path, fetch_fn=fetch_fn, now_utc=now + timedelta(days=8))
    assert calls["n"] > n_after_build
    # fetch failure later -> last cache, never empty
    def boom(name):
        raise RuntimeError("down")
    later = universe.load_universe(tmp_path, fetch_fn=boom, now_utc=now + timedelta(days=30))
    assert "AAPL" in later

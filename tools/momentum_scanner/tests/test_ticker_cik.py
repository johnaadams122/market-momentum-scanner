"""Centralized ticker normalization + strict ticker->CIK map.

The map is set-valued so the gate can REJECT a ticker with zero OR multiple CIKs (no auto-pass
around the authoritative EDGAR veto).
"""


def test_normalize_ticker():
    from momentum_scanner.edgar_rss import normalize_ticker
    assert normalize_ticker(" brk.b ") == "BRK-B"   # dot class-separator -> SEC hyphen
    assert normalize_ticker("aapl") == "AAPL"
    assert normalize_ticker(None) == ""


def test_ticker_to_ciks_builds_sets_and_flags_ambiguity(monkeypatch):
    from momentum_scanner import edgar_rss
    fake = {
        "0": {"cik_str": 123456, "ticker": "AAA"},
        "1": {"cik_str": 111, "ticker": "DUP"},
        "2": {"cik_str": 222, "ticker": "DUP"},     # same ticker, second CIK -> ambiguous
    }

    class R:
        def json(self):
            return fake
    monkeypatch.setattr(edgar_rss, "_get", lambda *a, **k: R())
    m = edgar_rss.ticker_to_ciks()
    # CIKs are normalized to 10-digit zero-padded strings.
    assert m["AAA"] == {str(123456).zfill(10)}
    assert m["DUP"] == {str(111).zfill(10), str(222).zfill(10)}   # 2 CIKs -> gate will REJECT no-data

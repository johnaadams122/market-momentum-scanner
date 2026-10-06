"""build_news_client selects the judge's news client behind a default-OFF flag.
'finnhub' (default) preserves current behavior; 'alpaca' = Alpaca primary + Finnhub fallback;
anything else fails safe back to Finnhub."""


def test_finnhub_is_the_default_and_explicit_choice():
    from momentum_scanner.scanner_movers_daemon import build_news_client
    from momentum_scanner.finnhub_news import FinnhubClient
    assert isinstance(build_news_client("finnhub"), FinnhubClient)


def test_alpaca_wraps_alpaca_primary_with_finnhub_fallback(monkeypatch):
    monkeypatch.setenv("SCANNER_MODE", "paper")      # hermetic: the live gate drops the fallback
    from momentum_scanner.scanner_movers_daemon import build_news_client
    from momentum_scanner.alpaca_news import AlpacaNewsClient, FallbackNewsClient
    from momentum_scanner.finnhub_news import FinnhubClient
    c = build_news_client("alpaca")
    assert isinstance(c, FallbackNewsClient)
    assert isinstance(c._primary, AlpacaNewsClient)      # Alpaca is primary
    assert isinstance(c._secondary, FinnhubClient)       # Finnhub is the degrade-not-blank fallback


def test_unknown_source_fails_safe_to_finnhub():
    from momentum_scanner.scanner_movers_daemon import build_news_client
    from momentum_scanner.finnhub_news import FinnhubClient
    assert isinstance(build_news_client("bogus"), FinnhubClient)


def test_none_source_fails_safe_to_finnhub():
    from momentum_scanner.scanner_movers_daemon import build_news_client
    from momentum_scanner.finnhub_news import FinnhubClient
    assert isinstance(build_news_client(None), FinnhubClient)


# ---- The Finnhub free tier is non-commercial. In live mode the Alpaca fallback must NOT
# silently use Finnhub (that path bypasses run_premarket_scan's live-gate) -- it must fail closed. ----

def test_alpaca_live_mode_disables_finnhub_fallback(monkeypatch):
    monkeypatch.setenv("SCANNER_MODE", "live")
    from momentum_scanner.scanner_movers_daemon import build_news_client
    from momentum_scanner.alpaca_news import AlpacaNewsClient
    c = build_news_client("alpaca")
    # Alpaca-only: an Alpaca outage now fails closed (ScannerNetworkError) rather than falling back
    # to the non-commercial Finnhub feed for live signals.
    assert isinstance(c, AlpacaNewsClient)


def test_alpaca_paper_mode_keeps_finnhub_fallback(monkeypatch):
    monkeypatch.setenv("SCANNER_MODE", "paper")
    from momentum_scanner.scanner_movers_daemon import build_news_client
    from momentum_scanner.alpaca_news import FallbackNewsClient
    assert isinstance(build_news_client("alpaca"), FallbackNewsClient)

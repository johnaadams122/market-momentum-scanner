"""tier2_enrich pre-judge EDGAR-signal hook (fail-isolated) + stats + daemon wiring.
The fake edgar mirrors EdgarContext: `.presence_and_dilution(ticker) -> (presence, form_dilution,
status)` (a 3-tuple with a status)."""
from datetime import datetime, timezone

from momentum_scanner import movers_scan, config

NOW = datetime(2026, 6, 29, 14, 0, tzinfo=timezone.utc)


class _Hit:
    def __init__(self, ticker):
        self.ticker = ticker; self.last = 5.0; self.prev_close = 4.0; self.gap_pct = 0.25
        self.total_volume = 500_000; self.avg_daily_volume = 100_000; self.rel_volume = 5.0
        self.day_high = 5.0; self.quote_time = None


class _Edgar:
    def __init__(self, result):
        self._r = result
        self.calls = []

    def presence_and_dilution(self, ticker):
        self.calls.append(ticker)
        return self._r


class _BoomEdgar:
    def presence_and_dilution(self, ticker):
        raise RuntimeError("edgar exploded")


def _enrich(hits, edgar, stats=None):
    return movers_scan.tier2_enrich(hits, float_cache={}, float_fetch_fn=lambda *a, **k: 5_000_000,
                                    now_utc=NOW, premarket=True, edgar=edgar, stats=stats)


def test_presence_no_dilution_sets_positive_signal():
    out = _enrich([_Hit("AAA")], _Edgar((True, False, "ok")))
    assert out[0].edgar_signal == config.EDGAR_PRESENCE_BONUS


def test_dilutive_form_nets_to_penalty():
    out = _enrich([_Hit("AAA")], _Edgar((True, True, "ok")))
    assert out[0].edgar_signal == config.EDGAR_PRESENCE_BONUS - config.EDGAR_FORM_DILUTION_PENALTY


def test_absent_filing_zero_signal():
    stats = {}
    out = _enrich([_Hit("AAA")], _Edgar((False, False, "ok")), stats=stats)
    assert out[0].edgar_signal == 0.0
    # absent -> no stat bump (not presence -> return 0.0, no stat)
    assert stats.get("edgar_presence_hit", 0) == 0


def test_edgar_none_disables_and_keeps_candidate():
    stats = {}
    out = _enrich([_Hit("AAA")], None, stats=stats)
    assert out[0].edgar_signal == 0.0
    assert stats["edgar_presence_disabled"] == 1


def test_edgar_disabled_by_config_even_when_object_present(monkeypatch):
    monkeypatch.setattr(config, "EDGAR_PREJUDGE_ENABLED", False)
    stats = {}
    out = _enrich([_Hit("AAA")], _Edgar((True, False, "ok")), stats=stats)
    assert out[0].edgar_signal == 0.0
    assert stats["edgar_presence_disabled"] == 1


def test_edgar_error_does_not_drop_candidate():
    stats = {}
    out = _enrich([_Hit("AAA")], _BoomEdgar(), stats=stats)
    assert len(out) == 1 and out[0].edgar_signal == 0.0
    assert stats["edgar_presence_error"] == 1


def test_edgar_status_error_zeroes_signal():
    stats = {}
    out = _enrich([_Hit("AAA")], _Edgar((False, False, "error")), stats=stats)
    assert out[0].edgar_signal == 0.0
    assert stats["edgar_presence_error"] == 1


def test_edgar_status_rate_limited_zeroes_signal():
    """A rate_limited status must be counted distinctly from a plain error."""
    stats = {}
    out = _enrich([_Hit("AAA")], _Edgar((False, False, "rate_limited")), stats=stats)
    assert out[0].edgar_signal == 0.0
    assert stats["edgar_presence_rate_limited"] == 1
    assert stats["edgar_presence_error"] == 0


def test_stats_keys_all_present_even_on_zero_counts():
    stats = {}
    _enrich([_Hit("AAA")], _Edgar((False, False, "ok")), stats=stats)
    for key in ("edgar_presence_hit", "edgar_presence_error", "edgar_presence_disabled",
               "edgar_presence_rate_limited"):
        assert key in stats


def test_discover_movers_forwards_edgar(monkeypatch):
    """discover_movers(..., edgar=..., stats=...) must forward edgar into tier2_enrich."""
    captured = {}
    real_tier2 = movers_scan.tier2_enrich

    def _spy(hits, **kw):
        captured.update(kw)
        return real_tier2(hits, **kw)

    monkeypatch.setattr(movers_scan, "tier1_filter", lambda quotes, now_et, **k: [_Hit("AAA")])
    monkeypatch.setattr(movers_scan, "batch_quotes", lambda *a, **k: {})
    monkeypatch.setattr(movers_scan, "tier2_enrich", _spy)
    edgar = _Edgar((True, False, "ok"))
    movers_scan.discover_movers(["AAA"], token_fn=lambda: "t", session=object(), float_cache={},
                               float_fetch_fn=lambda *a, **k: None, now_utc=NOW, now_et=NOW,
                               premarket=True, edgar=edgar, stats={})
    assert captured.get("edgar") is edgar

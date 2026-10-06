"""Regression + drift guard for the catalyst source-weight table.

THE FAILURE MODE: `catalyst_bonus` weights by `config.SOURCE_WEIGHTS.get(src, 0.0)`
(ranking.py). A provider missing from SOURCE_WEIGHTS makes `.get` return the 0.0 default,
so EVERY verdict from that provider earns a structural zero bonus no matter how real the
catalyst or how high the confidence.

It is a silent failure by construction -- a missing key is indistinguishable from a
deliberate 0.0 weight -- and it can be introduced by a correctness fix: before
`catalyst_evaluator` stamped the true `NewsItem.provider`, Alpaca items stamped the
literal "finnhub" and were scored at 0.7.

So this file pins two things:
  1. the regression itself (an alpaca-sourced real verdict earns a nonzero bonus), and
  2. the INVARIANT that would have caught it -- every provider literal a news client can
     stamp must have a positive weight in the table. Adding a third news client without a
     weight now fails here instead of silently zeroing its catalysts.
"""
from dataclasses import dataclass

import pytest

from momentum_scanner import alpaca_news, config, finnhub_news, ranking


@dataclass
class C:
    ticker: str = "AAA"
    gap_pct: float = 0.20
    rel_volume: float = 4.0
    float_shares: float | None = 5_000_000
    edgar_signal: float = 0.0
    score: float = 0.0
    catalyst_score: float = 0.0


@dataclass
class V:
    confidence: float | None = None
    content_source: str | None = None
    dilution_flag: str | None = None
    label: str = "real"


def test_alpaca_sourced_real_verdict_earns_a_nonzero_bonus():
    """The regression: without an `alpaca` weight this returns exactly 0.0."""
    bonus = ranking.catalyst_bonus(C(), V(confidence=0.9, content_source="alpaca"))
    assert bonus > 0.0
    assert bonus == round(config.W_CATALYST * 0.9 * config.SOURCE_WEIGHTS["alpaca"], 6)


def test_alpaca_is_weighted_like_finnhub():
    """Both are third-party news wires, not primary filings. 0.7 also RESTORES the exact
    pre-regression behaviour: alpaca items used to stamp the literal 'finnhub' and score at
    0.7, so this fix changes no historical intent -- it re-establishes it."""
    assert config.SOURCE_WEIGHTS["alpaca"] == config.SOURCE_WEIGHTS["finnhub"] == 0.7


@pytest.mark.parametrize("provider", [finnhub_news.PROVIDER, alpaca_news.PROVIDER])
def test_every_news_client_provider_has_a_positive_source_weight(provider):
    """The drift guard. A news client whose provider literal is missing from the table
    scores every one of its catalysts at zero, silently."""
    assert provider in config.SOURCE_WEIGHTS, (
        f"news client stamps provider={provider!r} but SOURCE_WEIGHTS has no entry, so "
        f"ranking.catalyst_bonus will silently score all of its verdicts 0.0"
    )
    assert config.SOURCE_WEIGHTS[provider] > 0.0


def test_unknown_source_still_scores_zero():
    """Unchanged, and deliberate: an unrecognised source earns no boost. The guard above is
    what keeps a REAL provider from ever landing in this branch."""
    assert ranking.catalyst_bonus(C(), V(confidence=0.9, content_source="not-a-provider")) == 0.0


def test_none_source_still_scores_zero():
    """'none' is the evaluator's explicit no-content stamp (watchlist_io.stamp_verdict) and its 0.0
    is designed behaviour, not a missing key."""
    assert config.SOURCE_WEIGHTS["none"] == 0.0
    assert ranking.catalyst_bonus(C(), V(confidence=0.9, content_source="none")) == 0.0

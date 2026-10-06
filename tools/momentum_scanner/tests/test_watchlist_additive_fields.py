"""Test watchlist v2 additive fields: catalyst_score + catalyst_content_source."""
from momentum_scanner import watchlist_io
from momentum_scanner.catalyst_evaluator import CatalystVerdict


def test_stamp_writes_content_source_and_keeps_provenance():
    """catalyst_content_source is set from verdict.content_source; catalyst_source (provenance) unchanged."""
    d = {}
    v = CatalystVerdict("PROCEED", "real", "ok", 0.9, "llm", content_source="finnhub")
    watchlist_io.stamp_verdict(d, v)
    assert d["catalyst_content_source"] == "finnhub"
    assert d["catalyst_source"] == "llm"          # provenance field unchanged (not overloaded)


def test_stamp_defaults_content_source_none_when_absent():
    """When verdict is None, catalyst_content_source defaults to 'none'."""
    d = {}
    watchlist_io.stamp_verdict(d, None)
    assert d["catalyst_content_source"] == "none"


def test_envelope_still_schema_version_2():
    """The additive fields do not change the envelope schema version."""
    env = watchlist_io.v2_envelope([{"ticker": "AAA"}])
    assert env["schema_version"] == 2             # additive fields keep schema v2

from momentum_scanner import config


def test_additive_score_weights_present():
    assert config.W_EDGAR == 0.3
    assert config.W_CATALYST == 0.2
    # Deliberate-change gate: this pins the table exactly, so adding/removing a source must be a
    # conscious edit here. A missing provider entry silently zeroes that provider's catalyst
    # bonus -- see tests/test_catalyst_source_weight_coverage.py.
    assert config.SOURCE_WEIGHTS == {
        "edgar": 1.0, "finnhub": 0.7, "alpaca": 0.7, "web": 0.5, "none": 0.0,
    }


def test_unknown_dilution_factor_is_a_penalty():
    # Must be < 1.0 (a real demotion) -- absent today, dilution_factor() would default to 1.0 (no penalty).
    assert config.DILUTION_FACTORS["unknown"] == 0.85


def test_edgar_prejudge_knobs_present():
    assert config.EDGAR_PREJUDGE_ENABLED is True
    assert config.EDGAR_PREJUDGE_MAX_PER_CYCLE == 40
    assert config.EDGAR_PRESENCE_BONUS == 0.3
    assert config.EDGAR_FORM_DILUTION_PENALTY == 0.3

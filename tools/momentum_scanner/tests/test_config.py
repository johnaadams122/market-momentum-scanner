from momentum_scanner import config


def test_item_sets_are_correct_and_disjoint():
    assert config.DILUTION_8K_ITEMS == {"1.03", "3.01", "3.02"}
    assert config.AMBIGUOUS_8K_ITEMS == {"1.01", "2.02", "8.01"}
    assert config.NO_DATA_8K_ITEMS == {"5.02", "9.01"}
    assert config.BODY_SCAN_8K_ITEMS == {"1.01", "7.01", "8.01"}
    assert config.DILUTION_8K_ITEMS.isdisjoint(config.AMBIGUOUS_8K_ITEMS)


def test_6k_is_body_scan_only_not_blanket_veto():
    assert "6-K" not in config.DILUTION_FORM_TYPES
    assert config.BODY_SCAN_FORM_TYPES == {"6-K"}
    # spot-check the expanded offering taxonomy is present
    for f in ("S-3", "424B5", "424B2", "POS AM", "EFFECT"):
        assert f in config.DILUTION_FORM_TYPES


def test_dilution_regex_matches_offering_language():
    r = config.DILUTION_REGEX
    for s in ["Securities Purchase Agreement", "at-the-market offering", "PIPE financing",
              "registered direct offering", "convertible note", "going concern",
              "1-for-10 reverse stock split", "shelf takedown"]:
        assert r.search(s), s


def test_dilution_regex_ignores_clean_catalyst():
    assert not config.DILUTION_REGEX.search("FDA approves Acme drug for broad use")


def test_thresholds_and_windows():
    assert config.CONF_MIN == 0.70
    assert config.WATCHLIST_TTL_MINUTES == 90
    assert config.DILUTION_LOOKBACK_CALENDAR_DAYS == 14
    assert config.REVERSE_SPLIT_LOOKBACK_DAYS == 365


def test_ineligible_suffixes_present():
    assert {"W", "U", "R"} <= config.INELIGIBLE_TICKER_SUFFIXES

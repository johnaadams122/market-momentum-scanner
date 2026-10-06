import importlib
from momentum_scanner import config

def test_scanner_constants_present_and_typed():
    assert config.PRICE_MIN == 1.0 and config.PRICE_MAX == 20.0
    assert config.GAP_MIN_PCT == 0.05
    assert config.RELVOL_MIN_PREMARKET == 5.0 and config.RELVOL_MIN_INTRADAY == 2.0
    assert config.VOLUME_FLOOR_PREMARKET == 100_000 and config.VOLUME_FLOOR_INTRADAY == 200_000
    assert config.FLOAT_MAX == 10_000_000 and config.FLOAT_SOFT_MAX == 20_000_000
    assert config.SCHWAB_QUOTE_CHUNK == 500
    assert config.FRACTION_FLOOR == 0.02
    assert config.OPENING_GUARD_MIN == 2
    # NEW_HOD_TOLERANCE_PCT is a tuned knob. This literal is the drift guard, so it is
    # updated deliberately with the knob -- the behavioral boundary tests derive from config.
    assert config.NEW_HOD_TOLERANCE_PCT == 0.02
    assert config.UNIVERSE_REFRESH_DAYS == 7
    assert config.UNIVERSE_SOURCES == ["nasdaqlisted.txt", "otherlisted.txt"]
    assert config.PREMARKET_SESSION_START_ET == "04:00"
    assert config.RTH_START_ET == "09:30" and config.RTH_END_ET == "16:00"
    assert config.QUOTE_MAX_AGE_SEC == 30
    # Constants the movers daemon, judge and consumers depend on -- defined here so config is self-contained
    assert config.WATCHLIST_MAX_AGE_SEC == 120
    assert config.VERDICT_TTL_MIN_PREMARKET == 30 and config.VERDICT_TTL_MIN_INTRADAY == 15
    assert config.JUDGE_CONCURRENCY == 1
    assert config.MAX_QUEUE_AGE_SEC == 120
    assert config.UNKNOWN_FLOAT_CAP == 5
    assert config.REMOTE_JUDGE_BREAKER_K == 3 and config.REMOTE_JUDGE_COOLDOWN_SEC == 120
    assert config.LOCK_STALE_SEC == 60
    assert config.WATCHED_SYMBOL_CAP == 15
    assert config.DISCOVERY_INTERVAL_SEC == 10 and config.EXECUTOR_INTERVAL_SEC == 30
    assert config.SHORTLIST_PRICEHISTORY_REFRESH_SEC == 90

def test_adapters_is_a_package():
    assert importlib.import_module("momentum_scanner.adapters") is not None


def test_float_decay_constants():
    assert config.FLOAT_SOFT_MAX == 20_000_000
    assert config.FLOAT_DECAY_REF == 20_000_000
    assert config.FLOAT_DECAY_K == 1.5
    assert config.FLOAT_HARD_MAX == 36_000_000


def test_dilution_factors_map():
    assert config.DILUTION_FACTORS["reverse_split_history"] == 1.0
    assert config.DILUTION_FACTORS["shelf_risk"] == 0.7
    assert config.DILUTION_FACTORS["active_offering"] == 0.3
    assert config.DILUTION_FACTORS[None] == 1.0


def test_form_type_split_union_matches_legacy():
    # Guards membership drift on the still-live filing path.
    assert config.ACTIVE_OFFERING_FORM_TYPES | config.SHELF_FORM_TYPES == config.DILUTION_FORM_TYPES
    assert "424B6" not in config.ACTIVE_OFFERING_FORM_TYPES


def test_foreign_recall_window():
    assert config.FOREIGN_6K_RECALL_LOOKBACK_DAYS == 3

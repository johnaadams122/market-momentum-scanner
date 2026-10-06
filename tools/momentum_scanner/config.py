"""Configuration constants for the momentum scanner.

Covers the catalyst hard gate, news sources, the movers scan, ranking levers and the judge
backends. The deterministic dilution veto is the primary guard; the LLM judge is secondary
and can never promote a vetoed candidate.
"""
import math
import os
import re
from collections import namedtuple

# --- judge / freshness thresholds ---
CONF_MIN = 0.70               # judge confidence floor; below -> REJECT no-data
WATCHLIST_TTL_MINUTES = 90    # watchlist.json expires_at = scanned_at + this

# --- two distinct lookback windows (must NOT be conflated) ---
DILUTION_LOOKBACK_CALENDAR_DAYS = 14   # ~10 trading days; active-offering window
REVERSE_SPLIT_LOOKBACK_DAYS = 365      # 12 months; reverse-split/charter window

# --- 8-K item taxonomy ---
DILUTION_8K_ITEMS = {"1.03", "3.01", "3.02"}      # hard reject (pump)
NO_DATA_8K_ITEMS = {"5.02", "9.01"}               # no catalyst when standalone
AMBIGUOUS_8K_ITEMS = {"1.01", "2.02", "8.01"}     # body + judge
BODY_SCAN_8K_ITEMS = {"1.01", "7.01", "8.01"}     # 8-K items whose body is regex-scanned

# --- form-type taxonomy ---
# Blanket-veto within DILUTION_LOOKBACK_CALENDAR_DAYS (offering/prospectus/effectiveness).
DILUTION_FORM_TYPES = {
    "S-1", "S-3", "F-1", "F-3", "424A", "424B1", "424B2", "424B3",
    "424B4", "424B5", "424B7", "POS AM", "EFFECT",
}
# 6-K (foreign issuer) is vetoed ONLY if its body matches DILUTION_REGEX (within 14d).
BODY_SCAN_FORM_TYPES = {"6-K"}
# Reverse-split proxy forms: vetoed only if the reverse-split regex matches the body
# within REVERSE_SPLIT_LOOKBACK_DAYS (plus 8-K Item 5.03, body-confirmed).
REVERSE_SPLIT_FORM_TYPES = {"DEF 14A", "DEFA14A", "PRE 14A", "DEF 14C", "PRE 14C"}

DILUTION_REGEX = re.compile(
    r"securities purchase agreement|registered direct|at[- ]the[- ]market|\"?atm\"? offering|"
    r"shelf takedown|convertible (note|debenture)|\bpipe\b|warrant (issuance|exercise)|"
    r"equity line|going concern|dilut(e|ion)|offering of|reverse[- ]?(stock )?split",
    re.IGNORECASE,
)
# Reverse-split-specific (used for the 365-day window so stale general offering language
# in a DEF 14A / Item 5.03 cannot veto outside the 14-day offering window).
REVERSE_SPLIT_REGEX = re.compile(r"reverse[- ]?(stock )?split", re.IGNORECASE)

# --- instrument eligibility (heuristic suffix check; not a full security-type lookup) ---
INELIGIBLE_TICKER_SUFFIXES = {"W", "WS", "U", "R", "RT"}

# --- judge orientation hints (the judge decides; these orient the prompt) ---
GREENLIST_HINTS = [
    "FDA approval / met primary endpoint",
    "earnings beat with raised/strong guidance",
    "material contract or award with a named counterparty and dollar value",
    "binding partnership with economic terms",
]

# --- local Gemma judge backend ---
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma4:12b-it-qat"
# Explicit context ceiling. Like the qwen3 shadow judge below, this judge call sends an
# explicit num_ctx + top-level truncate:false so an oversized prompt fails loud (HTTP 400)
# instead of being silently truncated or silently riding the server's VRAM-based default.
# 4096 matches Ollama's default context on smaller GPUs, and this judge's prompt runs
# ~2500 tokens, comfortably under it -- a fail-loud guarantee, not a behavior change.
SCANNER_OLLAMA_NUM_CTX = int(os.environ.get("SCANNER_OLLAMA_NUM_CTX", "4096"))

# --- Schwab market-data ---
PREMARKET_WINDOW_START_ET = "04:00"   # ext-hours session start for the relvol window
RELVOL_BASELINE_SESSIONS = 10         # prior completed sessions for the median premarket-volume baseline
SOURCE_MAX_AGE_MINUTES = 15           # premarket price-candle staleness gate (fail-closed beyond this)

# --- network timeouts (fail fast into status:error, never hang the scheduled run) ---
SCANNER_HTTP_TIMEOUT = 15             # seconds, for Finnhub + scanner-side Schwab market-data calls

# --- Alpaca news ---
# Benzinga-sourced full-content news; replaces the Finnhub free feed that is empty for thin low-floats.
ALPACA_NEWS_BASE_URL = "https://data.alpaca.markets/v1beta1/news"
ALPACA_NEWS_LIMIT = 50                # Alpaca page-size cap (1-50)
ALPACA_NEWS_MAX_PAGES = 5             # per ticker/day pagination safety cap
# Default OFF: "finnhub" preserves current behavior; SCANNER_NEWS_SOURCE=alpaca activates the Alpaca
# primary + Finnhub fallback (one-line rollback = unset the env / set finnhub).
SCANNER_NEWS_SOURCE = os.environ.get("SCANNER_NEWS_SOURCE", "finnhub").strip().lower()
# ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY read from env at call time; sent as APCA-* headers; never logged.

# --- prior-overnight catalyst window ---
# Default OFF preserves the same-day [04:00 ET, now] window. When True, the catalyst window extends back
# to the prior session close (prior weekday 16:00 ET) so premarket gaps on prior-evening/overnight news
# are caught (catalysts are often released the day before the gap). NOTE: skips
# weekends, not holidays (a holiday just yields a harmless empty extra day; the window/fetch over-covers).
CATALYST_PRIOR_SESSION_ENABLED = os.environ.get("CATALYST_PRIOR_SESSION", "0").strip().lower() in {"1", "true", "yes", "on"}

# --- selectivity: market-roundup filter ---
# "N Stocks Moving In X Session", "top gainers/losers", etc. are NOT single-name catalysts; they
# dominate the Benzinga "content" items and would drive false PROCEEDs. Drop them so a roundup-only
# news set falls through to the no-confirmed-news floor. Tight patterns (require the roundup context word)
# so real single-name PRs (for example a company authorizing a share buyback) are never dropped.
ROUNDUP_FILTER_ENABLED = True
ROUNDUP_HEADLINE_REGEX = re.compile(
    r"\bstocks?\s+moving\b"
    r"|\d+\s+[\w'\-\s]{0,30}?\bstocks\s+(with|to\s+watch)\b"
    r"|\btop\s+(gainers?|losers?)\b"
    r"|\bgap\s+up\s+and\s+gap\s+down\b"
    r"|\b(pre-?market|after-?market|after-?hours|intraday)\s+movers?\b"
    r"|\bbiggest\s+(gainers?|losers?|movers?)\b",
    re.IGNORECASE,
)

# --- Finnhub news-wire trigger ---
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
FINNHUB_WINDOW_START_ET = "04:00"     # one ET-anchored catalyst window [04:00, min(now, 09:30)]
FINNHUB_WINDOW_END_ET = "09:30"
FINNHUB_MAX_CONFIRM_CALLS = 40        # per-run /company-news fan-out cap (log drops, never silent)
FINNHUB_COMMERCIAL_USE = False        # free tier is for non-commercial use; see SCANNER_MODE handling
# FINNHUB_API_KEY read from env at call time; sent as X-Finnhub-Token header; never logged.

# News-PR dilution language (filing-body DILUTION_REGEX misses PR phrasing) -- run over headline+summary.
NEWS_DILUTION_REGEX = re.compile(
    r"announces pricing|pricing of (a |an |its )?(public |registered )?offering|gross proceeds|"
    r"private placement|registered direct|at[- ]the[- ]market|\bATM\b|priced (an|its) offering|"
    r"securities purchase agreement|convertible (note|debenture)|\bPIPE\b|equity line|"
    r"reverse[- ]?(stock )?split|going concern|dilut(e|ion)|shelf (registration|takedown)|warrant",
    re.IGNORECASE,
)

# --- movers scanner ---
PRICE_MIN, PRICE_MAX = 1.0, 20.0
GAP_MIN_PCT = 0.05
RELVOL_MIN_PREMARKET = 5.0
RELVOL_MIN_INTRADAY = 2.0
VOLUME_FLOOR_PREMARKET = 100_000
VOLUME_FLOOR_INTRADAY = 200_000
FLOAT_MAX = 10_000_000
FLOAT_SOFT_MAX = 20_000_000
SCHWAB_QUOTE_CHUNK = 500
FRACTION_FLOOR = 0.02
OPENING_GUARD_MIN = 2
NEW_HOD_TOLERANCE_PCT = 0.02          # decimal fraction (2.0%); a 0.1% tolerance starved the candidate funnel
UNIVERSE_REFRESH_DAYS = 7
UNIVERSE_SOURCES = ["nasdaqlisted.txt", "otherlisted.txt"]
NASDAQTRADER_SYMDIR_URL = "https://www.nasdaqtrader.com/dynamic/SymDir"   # universe source
PREMARKET_SESSION_START_ET = "04:00"
RTH_START_ET = "09:30"
RTH_END_ET = "16:00"
QUOTE_MAX_AGE_SEC = 30
SCHWAB_QUOTES_URL = "https://api.schwabapi.com/marketdata/v1/quotes"
DISCOVERY_INTERVAL_SEC = 10
EXECUTOR_INTERVAL_SEC = 30             # polling interval a downstream watchlist consumer is expected to use
WATCHED_SYMBOL_CAP = 15
SHORTLIST_PRICEHISTORY_REFRESH_SEC = 90
WATCHLIST_MAX_AGE_SEC = 120            # maximum watchlist age a consumer should accept
VERDICT_TTL_MIN_PREMARKET = 30         # verdict cache TTL
VERDICT_TTL_MIN_INTRADAY = 15
JUDGE_CONCURRENCY = 1                  # gemma serial; set 4 for haiku
MAX_QUEUE_AGE_SEC = 120
UNKNOWN_FLOAT_CAP = 5                  # unknown-float candidates capped/ranked low
REMOTE_JUDGE_BREAKER_K = 3            # remote-judge circuit breaker
REMOTE_JUDGE_COOLDOWN_SEC = 120
LOCK_STALE_SEC = 60                   # daemon single-instance stale reclaim
SCANNER_JUDGE_BACKEND = "gemma"      # gemma | haiku | sonnet (env override)
HAIKU_MODEL = "claude-haiku-4-5"
SONNET_MODEL = "claude-sonnet-4-6"

# --- Haiku dual-log shadow judge (A/B comparison) ---
# ORTHOGONAL to SCANNER_JUDGE_BACKEND above: the backend still resolves to gemma and
# gemma still DECIDES. This flag only adds a second, observed-only haiku call whose
# verdict is logged beside gemma's and never consulted. Default OFF so the CODE path
# stays the proven gemma-only one; rollback is unsetting the env var, not
# a redeploy. Requires ANTHROPIC_API_KEY in the launcher env (never in the repo);
# with the flag on and no key, resolve_news_judge degrades to gemma-only and says so.
SCANNER_JUDGE_SHADOW = os.environ.get("SCANNER_JUDGE_SHADOW", "0").strip().lower() in {"1", "true", "yes", "on"}
# Absolute path preferred (the daemons' cwd is the repo root, not the state dir).
SHADOW_JUDGE_LOG_PATH = os.environ.get("SCANNER_SHADOW_LOG_PATH", "").strip() or "judge_shadow.jsonl"
SHADOW_JUDGE_TIMEOUT = 20            # seconds; shadow sits inline in a serial batch
SHADOW_JUDGE_MAX_TOKENS = 512        # the judge returns one small JSON object
# Headroom only: billing is on tokens ACTUALLY generated, not the cap, so headroom is
# free. NOTE FOR ANYONE DEBUGGING SHORT RATIONALES: this token cap does not bound the
# rationale length -- the binding limit is the CHARACTER slice in
# catalyst_evaluator._parse_judge_json (200 by default). Changing this constant
# alone fixes nothing; the cap below is the one that binds.
SHADOW_JUDGE_RATIONALE_MAX = 600     # shadow-only; the deciding path stays at 200
SHADOW_JUDGE_QUEUE_CAP = 20
SHADOW_JUDGE_DAILY_CALL_CAP = 250
SHADOW_JUDGE_SHUTDOWN_SEC = 2.0
SHADOW_JUDGE_LOG_EVERY_DROPS = 25

# --- Qwen3:8b dual-log shadow judge (second, independent lane) ---
# Same observed-only invariant as the Haiku shadow above (gemma decides; this only logs a
# second opinion beside it), but this lane is fundamentally different in ONE way: Haiku is a
# cloud API call with zero GPU footprint, while qwen3 hits the SAME local Ollama server the
# gemma judge depends on. On a GPU too small to keep gemma and a qwen model resident together,
# a qwen3 shadow call risks EVICTING gemma, forcing the NEXT deciding judge call to pay a full
# model reload instead of hitting an already-resident model. This cannot delay or corrupt a
# verdict (the invariant in shadow_judge.py still holds byte-for-byte), but it CAN slow the
# deciding path's next call, which the plain nonblocking-queue invariant does not cover.
# Mitigation: a daily cap an order of magnitude below Haiku's 250,
# to bound how often that eviction can happen while data still accumulates. Default OFF;
# rollback is unsetting the env var, same as the Haiku lane.
SCANNER_JUDGE_SHADOW_QWEN3 = os.environ.get("SCANNER_JUDGE_SHADOW_QWEN3", "0").strip().lower() in {"1", "true", "yes", "on"}
SHADOW_JUDGE_LOG_PATH_QWEN3 = os.environ.get("SCANNER_SHADOW_LOG_PATH_QWEN3", "").strip() or "judge_shadow_qwen3.jsonl"
QWEN3_MODEL = "qwen3:8b"
SHADOW_JUDGE_QWEN3_TIMEOUT = 60      # generous vs. Haiku's 20s: a shadow call may pay a model-load, not just inference
SHADOW_JUDGE_QWEN3_DAILY_CALL_CAP = 25   # heavy rate-limit vs Haiku's 250 -- GPU-eviction mitigation, not a data-quality choice
FLOAT_REFRESH_DAYS_SHORTLIST = 1   # active Tier-2 shortlist float refresh cadence
MAX_JUDGE_PER_CYCLE = 20   # judge-batch per-cycle cap; lower-priority remainder left unjudged (logged)
DAEMON_HEARTBEAT_FILE = "discovery_heartbeat.json"   # daemon liveness/freshness heartbeat
FUNDAMENTALS_CACHE_FILE = "fundamentals.json"   # persisted float cache
DAEMON_IDLE_INTERVAL_SEC = 60        # closed-mode loop sleep (still heartbeats each loop)
DAEMON_WORKER_JOIN_SEC = 5           # bounded join on the judge worker at teardown
# Opt-in to run discovery ungated when no rate limiter is available (default OFF: an absent
# limiter fails CLOSED). Parsed defensively so an unrecognized value degrades safe (not an import crash):
DAEMON_ALLOW_UNGATED = os.environ.get("DAEMON_ALLOW_UNGATED", "0").strip().lower() in {"1", "true", "yes", "on"}

# --- low-float gap ranking levers ---
# Float soft-ceiling 20M + quadratic decay; hard-drop ~36M (curve zero-crossing 36.33M).
FLOAT_DECAY_REF = 20_000_000          # normalizer width (keeps the k=1.5 curve exactly)
FLOAT_DECAY_K = 1.5                    # decay steepness (prior, tunable)
FLOAT_HARD_MAX = 36_000_000           # tier2 drop threshold (prior)

# Dilution ranking factors (priors, tunable). None/unknown -> 1.0 (no demotion).
DILUTION_FACTORS = {
    "reverse_split_history": 1.0,     # strongest low-float setup -- explosiveness already priced into gap*relvol
    "shelf_risk": 0.7,                # trade smaller
    "active_offering": 0.3,           # the pop is the fade -- heavy demotion
    "unknown": 0.85,                  # CIK unresolvable -> can't run authoritative dilution -> small penalty
    None: 1.0,
}
# Split of DILUTION_FORM_TYPES into pricing-now vs. shelf-capacity (UNION must equal the legacy set).
ACTIVE_OFFERING_FORM_TYPES = {"424A", "424B1", "424B2", "424B3", "424B4", "424B5", "424B7"}
SHELF_FORM_TYPES = {"S-1", "S-3", "F-1", "F-3", "POS AM", "EFFECT"}

# Foreign-issuer 6-K catalyst recall lookback (calendar days; filingDate is date-granular).
FOREIGN_6K_RECALL_LOOKBACK_DAYS = 3

# --- catalyst additive score ---
# Additive catalyst terms. Priors in gap*relvol units, tunable from observed outcomes.
# The effective score is UNBOUNDED above via gap*relvol (relvol's denominator is floored,
# its numerator is not capped, and gap has no upper filter), so it is not confined to a
# small fixed band; see the optional score cap below.
W_EDGAR = 0.3                 # pre-judge net EDGAR signal weight (presence minus form-dilution)
W_CATALYST = 0.2             # post-judge confidence-weighted catalyst bonus weight
# Weight per verdict content_source. EVERY provider literal a news client can stamp must appear
# here: ranking.catalyst_bonus uses SOURCE_WEIGHTS.get(src, 0.0), so a missing key silently scores
# that provider's catalysts 0.0 and is indistinguishable from a deliberate zero. The coverage guard
# in tests/test_catalyst_source_weight_coverage.py fails if a client's provider has no entry.
# alpaca == finnhub (0.7): both are third-party news wires rather than primary filings, and 0.7 is
# also what Alpaca items scored before the evaluator began stamping the true provider.
SOURCE_WEIGHTS = {"edgar": 1.0, "finnhub": 0.7, "alpaca": 0.7, "web": 0.5, "none": 0.0}

# Pre-judge EDGAR check (folded into BASE score so it reorders the judge queue).
EDGAR_PREJUDGE_ENABLED = True         # False -> fallback: edgar_signal stays 0 (no queue reorder)
EDGAR_PREJUDGE_MAX_PER_CYCLE = 40     # per-cycle EDGAR lookup budget (SEC fair-access); excess -> edgar_signal 0
EDGAR_PRESENCE_BONUS = 0.3            # +bonus when a fresh filing exists
EDGAR_FORM_DILUTION_PENALTY = 0.3    # -penalty when that filing is an offering/shelf form (net can be <= 0)

# --- Score cap -- config-gated, DEFAULT OFF ---
# The stamped candidate `score` is clamped to min(E, cap) at the rank_candidates stamp site when
# the cap is ON; every ordering key stays RAW. The minimum allowed cap keeps the cap above the
# typical high-score band so the cap cannot flatten it; the maximum stops a fat-fingered huge
# value running an effectively-disabled cap under an ENABLED banner.
SCORE_CAP_DEFAULT = 8.0
SCORE_CAP_MIN_ALLOWED = 8.0
SCORE_CAP_MAX_ALLOWED = 1500.0
_SCORE_CAP_ON_TOKENS = {"1", "true", "yes", "on"}
_SCORE_CAP_OFF_TOKENS = {"0", "false", "no", "off", ""}

# Tri-state resolution result: state is "off" | "on" | "invalid". `value` is the active cap when
# ON, else None. `reason` names the raw rejected token when INVALID, else None.
ScoreCapResolution = namedtuple("ScoreCapResolution", ["state", "value", "reason"])


class ScoreCapConfigError(ValueError):
    """An INVALID score-cap config reached a code path that must never run under one.

    The daemon entrypoint enforces INVALID as a startup ABORT (banner, no scan loop, no
    publish); rank_candidates / the movers writer raise this so a library caller that skipped
    the entrypoint fails LOUDLY instead of running silently uncapped or silently capped."""


def resolve_score_cap(env=None):
    """Resolve SCANNER_SCORE_CAP / SCANNER_SCORE_CAP_VALUE into a ScoreCapResolution.

    Fail-closed contract: tokens are matched after
    strip().lower(); an unset flag is OFF; any OTHER present token is INVALID -- never
    silently OFF. The value must parse as a finite float inside
    [SCORE_CAP_MIN_ALLOWED, SCORE_CAP_MAX_ALLOWED]; anything else present is INVALID --
    never a silent default, never a clamp-into-range. A present-but-garbage value is
    INVALID even when the flag resolves OFF: a broken launcher env is a config error,
    not something to run past. Parsing never raises (import-safe); enforcement of
    INVALID is the caller's job (the daemon aborts; ranking/writer raise)."""
    env = os.environ if env is None else env
    raw_flag = env.get("SCANNER_SCORE_CAP")
    raw_value = env.get("SCANNER_SCORE_CAP_VALUE")

    if raw_flag is None:
        flag_on = False
    else:
        tok = raw_flag.strip().lower()
        if tok in _SCORE_CAP_ON_TOKENS:
            flag_on = True
        elif tok in _SCORE_CAP_OFF_TOKENS:
            flag_on = False
        else:
            return ScoreCapResolution(
                "invalid", None,
                "SCANNER_SCORE_CAP=%r is not a recognized on/off token" % raw_flag.strip())

    if raw_value is None:
        value = SCORE_CAP_DEFAULT
    else:
        try:
            value = float(raw_value.strip())
        except (TypeError, ValueError):
            return ScoreCapResolution(
                "invalid", None,
                "SCANNER_SCORE_CAP_VALUE=%r does not parse as a float" % raw_value.strip())
        if not math.isfinite(value) or not (SCORE_CAP_MIN_ALLOWED <= value <= SCORE_CAP_MAX_ALLOWED):
            return ScoreCapResolution(
                "invalid", None,
                "SCANNER_SCORE_CAP_VALUE=%r outside the allowed range [%s, %s]"
                % (raw_value.strip(), SCORE_CAP_MIN_ALLOWED, SCORE_CAP_MAX_ALLOWED))

    if flag_on:
        return ScoreCapResolution("on", value, None)
    return ScoreCapResolution("off", None, None)

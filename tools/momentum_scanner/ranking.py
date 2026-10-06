"""Candidate ranking for the movers watchlist. `score` (base) is the numeric pre-judge
priority (gap * relvol + W_EDGAR * net EDGAR signal) that drives the judge queue.
`catalyst_bonus` is the post-judge, label-aware, confidence- and source-weighted additive
term (real label only -- pump/no-data/avoid never earn a boost). `rank_candidates` writes
the EFFECTIVE priority ((base * float_factor + catalyst_bonus) * dilution_factor) into
c.score -- the published wire field consumed by downstream tools -- and stamps the isolated
bonus onto c.catalyst_score for tuning. Unknown-float candidates are kept, ranked below
every known-float name, and capped (config.UNKNOWN_FLOAT_CAP).

Score cap (config-gated, DEFAULT OFF): when
config.resolve_score_cap() is ON and the raw effective score E is finite, the STAMPED wire
value is min(E, cap) plus provenance stamps (score_uncapped=E, score_cap_value=cap). Every
ORDERING key stays RAW E -- producer ordering is identical ON vs OFF, no manufactured ties.
A non-finite E skips the clamp (min(NaN, cap) would return NaN) and stamps no new keys; the
watchlist publish then fails atomically downstream (allow_nan=False) exactly as it does OFF."""
import math

from momentum_scanner import config


def score(c):
    relvol_mult = c.rel_volume if c.rel_volume is not None else 1.0
    edgar_signal = getattr(c, "edgar_signal", 0.0) or 0.0
    return round(c.gap_pct * relvol_mult + config.W_EDGAR * edgar_signal, 6)


def float_factor(f):
    if f is None or f <= config.FLOAT_SOFT_MAX:
        return 1.0
    x = (f - config.FLOAT_SOFT_MAX) / config.FLOAT_DECAY_REF
    return max(0.0, 1.0 - config.FLOAT_DECAY_K * x * x)


def dilution_factor(flag):
    return config.DILUTION_FACTORS.get(flag, 1.0)


def _dilution_flag(c, verdicts):
    v = (verdicts or {}).get(c.ticker)
    return getattr(v, "dilution_flag", None) if v is not None else None


_LABEL_BONUS_MULT = {"real": 1.0, "pump": 0.0, "no-data": 0.0, "avoid": 0.0}


def catalyst_bonus(c, verdict):
    """Post-judge additive catalyst term: W_CATALYST * confidence * source_weight * label_mult.
    No verdict / no confidence / no source / non-real label -> 0.0 (no boost)."""
    conf = getattr(verdict, "confidence", None) if verdict is not None else None
    src = getattr(verdict, "content_source", None) if verdict is not None else None
    label = getattr(verdict, "label", None) if verdict is not None else None
    if conf is None or src is None:
        return 0.0
    mult = _LABEL_BONUS_MULT.get(label, 0.0)
    return round(config.W_CATALYST * conf * config.SOURCE_WEIGHTS.get(src, 0.0) * mult, 6)


def effective_score(c, verdicts=None):
    v = (verdicts or {}).get(c.ticker)
    base = score(c)
    bonus = catalyst_bonus(c, v)
    dil = dilution_factor(_dilution_flag(c, verdicts))
    # dilution wraps the ENTIRE post-judge priority (base*float + bonus); float attenuates
    # only the move+edgar base, never the news-judge quality.
    return round((base * float_factor(c.float_shares) + bonus) * dil, 6)


def _confidence(c, verdicts):
    v = (verdicts or {}).get(c.ticker)
    conf = getattr(v, "confidence", None) if v is not None else None
    return conf if conf is not None else -1.0


def rank_candidates(candidates, verdicts=None):
    cap_cfg = config.resolve_score_cap()
    if cap_cfg.state == "invalid":
        # The daemon entrypoint aborts startup on INVALID before ranking can run; a library
        # caller that skipped that gate fails loudly here rather than running silently
        # uncapped-under-an-enabled-banner or silently capped (fail-closed).
        raise config.ScoreCapConfigError(cap_cfg.reason)
    cap = cap_cfg.value if cap_cfg.state == "on" else None
    raw = {}                                             # id(c) -> raw effective E (ordering key)
    for c in candidates:
        v = (verdicts or {}).get(c.ticker)
        c.catalyst_score = catalyst_bonus(c, v)          # isolated bonus, for stamping + tuning
        e = effective_score(c, verdicts)                 # raw E -- formula unchanged, OFF and ON
        raw[id(c)] = e
        if cap is not None and math.isfinite(e):
            c.score = min(e, cap)                        # stamped wire value, clamped
            c.score_uncapped = e
            c.score_cap_value = cap
        else:
            c.score = e                                  # OFF (or non-finite E): stamped == raw
            # Clear any stale ON-era provenance so re-ranking the SAME objects with the cap
            # OFF (or after E went non-finite) publishes pure OFF rows -- None matches the
            # MoverCandidate field defaults and the writer's None-key dropping removes them.
            c.score_uncapped = None
            c.score_cap_value = None
    key = lambda c: (-raw[id(c)], -_confidence(c, verdicts))   # ordering ALWAYS on raw E
    known = sorted((c for c in candidates if c.float_shares is not None), key=key)
    unknown = sorted((c for c in candidates if c.float_shares is None), key=key)
    return known + unknown[:config.UNKNOWN_FLOAT_CAP]

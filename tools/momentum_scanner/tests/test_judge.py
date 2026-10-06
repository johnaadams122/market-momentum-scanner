# tests/test_judge.py
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from dataclasses import dataclass
from momentum_scanner import judge
from momentum_scanner.catalyst_evaluator import CatalystVerdict, default_news_judge
from momentum_scanner.errors import ScannerNetworkError, ScannerError
from momentum_scanner.finnhub_news import NewsItem

ET = ZoneInfo("America/New_York")
PM = datetime(2026, 6, 29, 8, 30, tzinfo=ET)             # premarket
INTRA = datetime(2026, 6, 29, 13, 0, tzinfo=ET)         # intraday


@dataclass
class _Cand:
    ticker: str = "ABCD"


def _item(ts):
    return NewsItem(ticker="ABCD", headline="ABCD wins FDA approval", summary="primary endpoint met",
                    published=ts, url="http://x/1", source="wire", id=1)


class _Client:
    def __init__(self, items=None, boom=False):
        self._items, self._boom, self.calls = items or [], boom, []
    def fetch_company_news(self, ticker, frm, to):
        self.calls.append((ticker, frm, to))
        if self._boom:
            raise ScannerNetworkError("finnhub down")
        return self._items


def test_resolve_backend_gemma_and_remote_deferred():
    assert judge.resolve_news_judge("gemma") is default_news_judge
    assert judge.resolve_news_judge() is default_news_judge
    assert judge.resolve_news_judge("haiku") is default_news_judge      # deferred -> gemma seam
    assert judge.resolve_news_judge("sonnet") is default_news_judge


def test_intraday_post_open_news_is_judged_not_dropped():
    # An 11:00 ET item at 13:00 ET must reach evaluate (premarket et_window would drop it).
    ts = datetime(2026, 6, 29, 15, 0, tzinfo=timezone.utc)              # 11:00 ET
    client = _Client(items=[_item(ts)])
    seen = {}
    def fake_eval(c, items, **k):
        seen["items"] = items
        return CatalystVerdict("PROCEED", "real", "FDA", 0.9, "llm")
    verdict, max_ts = judge.judge_candidate(_Cand(), news_client=client, now_et=INTRA, premarket=False,
                                            evaluate=fake_eval)
    assert len(seen["items"]) == 1                                      # NOT dropped by a 09:30 cap
    assert verdict.decision == "PROCEED" and verdict.judged_at is not None
    assert verdict.headline == "ABCD wins FDA approval" and max_ts == ts


def test_premarket_window_excludes_after_0930():
    # premarket window is [04:00, min(now,09:30)]; a 10:00 ET item must be EXCLUDED in premarket mode.
    after = datetime(2026, 6, 29, 14, 0, tzinfo=timezone.utc)           # 10:00 ET
    client = _Client(items=[_item(after)])
    seen = {}
    def fake_eval(c, items, **k):
        seen["items"] = items
        return CatalystVerdict("REJECT", "no-data", "no confirmed news", None, "rule")
    judge.judge_candidate(_Cand(), news_client=client, now_et=PM, premarket=True, evaluate=fake_eval)
    assert seen["items"] == []                                          # out-of-premarket-window dropped


def test_no_news_is_real_reject_stamped():
    client = _Client(items=[])
    reject = CatalystVerdict("REJECT", "no-data", "no confirmed news", None, "rule")
    verdict, max_ts = judge.judge_candidate(_Cand(), news_client=client, now_et=INTRA, premarket=False,
                                            evaluate=lambda c, items, **k: reject)
    assert verdict.decision == "REJECT" and verdict.judged_at is not None
    assert verdict.headline is None and max_ts is None


def test_finnhub_systemic_failure_unjudged():
    client = _Client(boom=True)
    verdict, max_ts = judge.judge_candidate(_Cand(), news_client=client, now_et=INTRA, premarket=False,
                                            evaluate=lambda c, items, **k: None)
    assert verdict is None and max_ts is None


def test_edgar_systemic_failure_unjudged():
    # evaluate (raise_systemic=True) re-raises a ScannerError -> judge leaves it UNJUDGED, not a cached REJECT.
    ts = datetime(2026, 6, 29, 15, 0, tzinfo=timezone.utc)
    client = _Client(items=[_item(ts)])
    def boom_eval(c, items, **k):
        raise ScannerError("edgar veto failed")
    verdict, max_ts = judge.judge_candidate(_Cand(), news_client=client, now_et=INTRA, premarket=False,
                                            evaluate=boom_eval)
    assert verdict is None and max_ts is None


def test_evaluate_called_with_raise_systemic_true():
    client = _Client(items=[])
    seen = {}
    def fake_eval(c, items, **k):
        seen["raise_systemic"] = k.get("raise_systemic")
        return CatalystVerdict("REJECT", "no-data", "x", None, "rule")
    judge.judge_candidate(_Cand(), news_client=client, now_et=INTRA, premarket=False, evaluate=fake_eval)
    assert seen["raise_systemic"] is True


def test_reject_with_news_still_stamps_headline():
    ts = datetime(2026, 6, 29, 15, 0, tzinfo=timezone.utc)
    client = _Client(items=[_item(ts)])
    reject = CatalystVerdict("REJECT", "pump", "dilution language in news", None, "rule")
    verdict, max_ts = judge.judge_candidate(_Cand(), news_client=client, now_et=INTRA, premarket=False,
                                            evaluate=lambda c, items, **k: reject)
    assert verdict.decision == "REJECT" and verdict.headline == "ABCD wins FDA approval"
    assert verdict.judged_at is not None and max_ts == ts


# ---- prior-overnight catalyst window (flag default OFF preserves same-day behavior) ----
# Verified weekdays: 2026-07-06 = Monday, 2026-07-07 = Tuesday, 2026-07-03 = Friday.
TUE_PM = datetime(2026, 7, 7, 8, 30, tzinfo=ET)                    # Tuesday premarket 08:30 ET
MON_EVE = datetime(2026, 7, 6, 22, 0, tzinfo=timezone.utc)        # Mon 2026-07-06 18:00 ET (prior evening)


def _seen_eval(seen):
    def _e(c, items, **k):
        seen["items"] = items
        seen["kwargs"] = k
        return CatalystVerdict("REJECT", "no-data", "x", None, "rule")
    return _e


def test_prior_session_off_by_default_fetches_same_day_only(monkeypatch):
    monkeypatch.setattr(judge.config, "CATALYST_PRIOR_SESSION_ENABLED", False)
    client = _Client(items=[_item(MON_EVE)])
    seen = {}
    judge.judge_candidate(_Cand(), news_client=client, now_et=TUE_PM, premarket=True, evaluate=_seen_eval(seen))
    assert seen["items"] == []                                     # prior-evening excluded (same-day window)
    assert client.calls[0][1] == client.calls[0][2] == "2026-07-07"   # from==to==Tue


def test_prior_session_on_includes_prior_evening_and_widens_fetch(monkeypatch):
    monkeypatch.setattr(judge.config, "CATALYST_PRIOR_SESSION_ENABLED", True)
    client = _Client(items=[_item(MON_EVE)])
    seen = {}
    judge.judge_candidate(_Cand(), news_client=client, now_et=TUE_PM, premarket=True, evaluate=_seen_eval(seen))
    assert len(seen["items"]) == 1                                 # prior-evening catalyst now in-window
    assert client.calls[0][1] == "2026-07-06"                      # from == Mon (prior session close)
    assert client.calls[0][2] == "2026-07-07"                      # to == Tue (today)


def test_prior_session_on_skips_weekend_to_friday(monkeypatch):
    monkeypatch.setattr(judge.config, "CATALYST_PRIOR_SESSION_ENABLED", True)
    mon = datetime(2026, 7, 6, 8, 30, tzinfo=ET)                   # Monday premarket
    fri_eve = datetime(2026, 7, 3, 21, 0, tzinfo=timezone.utc)     # Fri 2026-07-03 17:00 ET
    client = _Client(items=[_item(fri_eve)])
    seen = {}
    judge.judge_candidate(_Cand(), news_client=client, now_et=mon, premarket=True, evaluate=_seen_eval(seen))
    assert len(seen["items"]) == 1                                 # Friday-evening catalyst included
    assert client.calls[0][1] == "2026-07-03"                      # from == Friday (weekend skipped)


# ---- Systemic-failure visibility ----------------------------------------------
# Without a counter, a total news-feed outage is INVISIBLE: every ticker raises ScannerError, a
# DEBUG-only log is hidden (daemons run at INFO) and judge returns None, so the day looks like an
# honest 0-PROCEED session instead of a dead feed. These pin the counter + above-DEBUG log that tell those two apart.

def _boom(**kw):
    """news_client that always raises a systemic error."""
    return _Client(boom=True)


def _judge_boom():
    return judge.judge_candidate(_Cand(), news_client=_boom(), now_et=INTRA, premarket=False,
                                 evaluate=lambda c, items, **k: None)


def test_systemic_failure_counter_starts_at_zero_after_reset():
    judge.reset_systemic_failures()
    assert judge.systemic_failures() == 0


def test_systemic_failure_increments_counter_and_logs_above_debug(caplog):
    judge.reset_systemic_failures()
    with caplog.at_level(logging.INFO, logger="momentum_scanner.judge"):
        verdict, max_ts = _judge_boom()
    assert verdict is None and max_ts is None                      # unchanged fail-closed contract
    assert judge.systemic_failures() == 1
    above_debug = [r for r in caplog.records if r.levelno > logging.DEBUG]
    assert len(above_debug) == 1
    assert above_debug[0].levelno >= logging.WARNING
    assert "ABCD" in above_debug[0].getMessage()


def test_systemic_failures_accumulate_across_calls():
    judge.reset_systemic_failures()
    for _ in range(3):
        _judge_boom()
    assert judge.systemic_failures() == 3


def test_repeated_systemic_failures_are_throttled_then_resummarised(caplog, monkeypatch):
    # An outage must not emit one WARNING per ticker per cycle; it warns on the FIRST of a burst and
    # then every _SYSTEMIC_LOG_EVERY, carrying the cumulative count so the volume is bounded.
    monkeypatch.setattr(judge, "_SYSTEMIC_LOG_EVERY", 3)
    judge.reset_systemic_failures()
    with caplog.at_level(logging.INFO, logger="momentum_scanner.judge"):
        for _ in range(7):
            _judge_boom()
    above_debug = [r for r in caplog.records if r.levelno > logging.DEBUG]
    assert len(above_debug) == 3                                   # bursts 1, 3, 6 -- the 7th is muted
    assert judge.systemic_failures() == 7                          # every failure still counted
    assert "systemic_failures=6" in above_debug[-1].getMessage()   # cumulative count is reported


def test_successful_judge_does_not_increment_and_rearms_the_burst_warning(caplog, monkeypatch):
    # A recovery must re-arm the alarm: the NEXT outage warns immediately instead of staying silent
    # until the throttle interval comes around again.
    monkeypatch.setattr(judge, "_SYSTEMIC_LOG_EVERY", 100)
    judge.reset_systemic_failures()
    _judge_boom()                                                  # burst 1 -> warns
    ok = CatalystVerdict("REJECT", "no-data", "no confirmed news", None, "rule")
    judge.judge_candidate(_Cand(), news_client=_Client(items=[]), now_et=INTRA, premarket=False,
                          evaluate=lambda c, items, **k: ok)
    assert judge.systemic_failures() == 1                          # a success never increments
    caplog.clear()                                                 # caplog accumulates for the whole test
    with caplog.at_level(logging.INFO, logger="momentum_scanner.judge"):
        _judge_boom()                                              # NEW burst -> must warn again
    above_debug = [r for r in caplog.records if r.levelno > logging.DEBUG]
    assert len(above_debug) == 1
    assert judge.systemic_failures() == 2


# ---- shadow ticker stamping: EITHER shadow lane needs the ticker on its comparison rows ----

import pytest  # noqa: E402
from momentum_scanner import shadow_judge  # noqa: E402


@pytest.mark.parametrize("haiku,qwen3,expected", [
    (True, False, "ABCD"), (False, True, "ABCD"), (True, True, "ABCD"), (False, False, None),
])
def test_ticker_is_stamped_when_either_shadow_lane_is_enabled(monkeypatch, haiku, qwen3, expected):
    monkeypatch.setattr(judge.config, "SCANNER_JUDGE_SHADOW", haiku)
    monkeypatch.setattr(judge.config, "SCANNER_JUDGE_SHADOW_QWEN3", qwen3)
    shadow_judge.set_current_ticker(None)
    try:
        judge.judge_candidate(_Cand(), news_client=_Client(), now_et=INTRA, premarket=False,
                              news_judge=default_news_judge,
                              evaluate=lambda c, items, **k: CatalystVerdict("REJECT", "no-data", "x", None, "rule"))
        assert shadow_judge._current_ticker == expected
    finally:
        shadow_judge.set_current_ticker(None)

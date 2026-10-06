import pytest
from datetime import datetime, timezone
from momentum_scanner.adapters import schwab_quotes as sq
from momentum_scanner.errors import ScannerNetworkError

class FakeResp:
    def __init__(self, status, payload):
        self.status_code = status
        self._p = payload
    def json(self):
        return self._p

def _row(last, close, vol, high, avg, ext=None, qt=1_900_000_000_000):
    r = {"quote": {"lastPrice": last, "closePrice": close, "totalVolume": vol,
                   "highPrice": high, "quoteTime": qt},
         "fundamental": {"avg10DaysVolume": avg}}
    if ext is not None:
        r["extended"] = ext
    return r

def _session(payload):
    class S:
        def get(self, url, headers=None, params=None, timeout=None):
            syms = params["symbols"].split(",")
            if isinstance(payload, dict):
                return FakeResp(200, {s: payload.get(s, {}) for s in syms})
            return payload(syms)
    return S()

NOW = datetime(2026, 6, 29, 14, 0, tzinfo=timezone.utc)

def test_regular_fields():
    out = sq.batch_quotes(["AAPL"], token_fn=lambda: "t",
                          session=_session({"AAPL": _row(190.0, 188.0, 1_000_000, 191.0, 50_000_000)}),
                          now_utc=NOW, premarket=False)
    q = out["AAPL"]
    assert (q.last, q.prev_close, q.total_volume, q.day_high, q.avg_daily_volume) == \
           (190.0, 188.0, 1_000_000, 191.0, 50_000_000.0)
    assert q.source == "regular" and q.quote_time is not None

def test_premarket_uses_extended_block():
    row = _row(None, 1.8, None, 2.1, 400_000, ext={"lastPrice": 2.0, "totalVolume": 80_000})
    out = sq.batch_quotes(["X"], token_fn=lambda: "t", session=_session({"X": row}), now_utc=NOW, premarket=True)
    q = out["X"]
    assert q.last == 2.0 and q.total_volume == 80_000 and q.source == "extended"

def test_premarket_quote_time_from_extended_block():
    # extended block carries its own timestamp; quote_time must reflect IT, not the regular quoteTime
    ext = {"lastPrice": 2.0, "totalVolume": 80_000, "quoteTime": 1_900_000_500_000}
    row = _row(None, 1.8, None, 2.1, 400_000, ext=ext, qt=1_900_000_000_000)
    out = sq.batch_quotes(["X"], token_fn=lambda: "t", session=_session({"X": row}), now_utc=NOW, premarket=True)
    assert out["X"].quote_time == datetime.fromtimestamp(1_900_000_500_000 / 1000.0, tz=timezone.utc)

def test_premarket_absent_extended_skips_symbol():
    row = _row(190.0, 188.0, 1_000_000, 191.0, 50_000_000)   # no 'extended' block
    out = sq.batch_quotes(["X"], token_fn=lambda: "t", session=_session({"X": row}), now_utc=NOW, premarket=True)
    q = out["X"]
    assert q.last is None and q.total_volume is None   # skipped, NOT regular fallback

def test_chunking_multiple_calls():
    seen = []
    def payload(syms):
        seen.append(tuple(syms))
        return FakeResp(200, {s: _row(2.0, 1.8, 300_000, 2.1, 400_000) for s in syms})
    big = [f"S{i}" for i in range(sq.config.SCHWAB_QUOTE_CHUNK + 5)]
    out = sq.batch_quotes(big, token_fn=lambda: "t", session=_session(payload), now_utc=NOW, premarket=False)
    assert len(seen) == 2 and len(out) == len(big)

def test_http_error_raises_after_one_retry():
    calls = {"n": 0}
    def payload(syms):
        calls["n"] += 1
        return FakeResp(429, {})
    with pytest.raises(ScannerNetworkError):
        sq.batch_quotes(["AAPL"], token_fn=lambda: "t", session=_session(payload), now_utc=NOW, premarket=False)
    assert calls["n"] == 2

def test_token_failure_raises_scannernetworkerror():
    def boom():
        raise RuntimeError("schwab auth expired")
    with pytest.raises(ScannerNetworkError):
        sq.batch_quotes(["AAPL"], token_fn=boom, session=_session({}), now_utc=NOW, premarket=False)

def test_garbled_or_nonfinite_fields_yield_none():
    bad = {"quote": {"lastPrice": "oops", "totalVolume": float("inf"), "closePrice": float("nan"),
                     "highPrice": None, "quoteTime": None}, "fundamental": {"avg10DaysVolume": "x"}}
    out = sq.batch_quotes(["BAD"], token_fn=lambda: "t", session=_session({"BAD": bad}), now_utc=NOW, premarket=False)
    q = out["BAD"]
    assert q.last is None and q.total_volume is None and q.avg_daily_volume is None
    assert q.prev_close is None, "float('nan') field must yield None, not NaN"


def test_connection_error_raises_scannernetworkerror_after_two_attempts():
    """A ConnectionError from session.get must surface as ScannerNetworkError after 2 retries."""
    calls = {"n": 0}
    class ErrorSession:
        def get(self, url, headers=None, params=None, timeout=None):
            calls["n"] += 1
            raise ConnectionError("connection refused")
    with pytest.raises(ScannerNetworkError):
        sq.batch_quotes(["AAPL"], token_fn=lambda: "t", session=ErrorSession(),
                        now_utc=NOW, premarket=False)
    assert calls["n"] == 2, f"expected 2 retry attempts, got {calls['n']}"

# tests/test_catalyst_raise_systemic.py
import pytest
from dataclasses import dataclass
from datetime import datetime, timezone
from momentum_scanner.catalyst_evaluator import evaluate_news
from momentum_scanner.errors import ScannerError

NOW = datetime(2026, 6, 29, 13, 0, tzinfo=timezone.utc)


@dataclass
class _Cand:
    ticker: str = "ABCD"; float_shares: float = 4_000_000; rel_volume: float = 5.0


@dataclass
class _News:
    ticker: str = "ABCD"; headline: str = "h"; summary: str = "s"; published: datetime = NOW


def _boom_map():
    raise ScannerError("ticker map down")


def test_default_returns_reject_on_edgar_failure():
    v = evaluate_news(_Cand(), [_News()], ticker_to_ciks=_boom_map, now=NOW)
    assert v.decision == "REJECT" and v.label == "no-data"


def test_raise_systemic_reraises_on_edgar_failure():
    with pytest.raises(ScannerError):
        evaluate_news(_Cand(), [_News()], ticker_to_ciks=_boom_map, now=NOW, raise_systemic=True)

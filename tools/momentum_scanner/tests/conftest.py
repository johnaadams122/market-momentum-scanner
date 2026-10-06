import sys
from pathlib import Path

# tests/ -> momentum_scanner/ -> tools/ : add tools/ so `momentum_scanner` resolves as a package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


import pytest


@pytest.fixture(autouse=True)
def synthetic_sec_contact(monkeypatch):
    """Offline synthetic contact, never an operator identity."""
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "market-momentum-synthetic-contact")

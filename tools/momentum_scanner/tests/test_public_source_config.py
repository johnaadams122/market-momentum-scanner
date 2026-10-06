"""Public configuration boundaries; all provider and import boundaries are synthetic."""
import sys
from types import SimpleNamespace
import pytest
from momentum_scanner import calibrate_min_score_live as cal
from momentum_scanner import edgar_rss as edgar
from momentum_scanner import schwab_data as sd
from momentum_scanner import rate_gate as rg
from momentum_scanner import daemon_clock as dc

@pytest.mark.parametrize('raw', [None, '', '   '])
def test_missing_or_blank_sec_contact_precedes_provider_call(monkeypatch, raw):
    if raw is None:
        monkeypatch.delenv('SEC_EDGAR_USER_AGENT', raising=False)
    else:
        monkeypatch.setenv('SEC_EDGAR_USER_AGENT', raw)
    calls = []
    with pytest.raises(edgar.ScannerNetworkError, match='explicitly configured'):
        edgar._get('https://synthetic.invalid', fetcher=lambda *a, **k: calls.append(True))
    assert calls == []

def test_explicit_sec_contact_is_forwarded(monkeypatch):
    monkeypatch.setenv('SEC_EDGAR_USER_AGENT', 'synthetic-fixture-contact')
    assert edgar._headers() == {'User-Agent': 'synthetic-fixture-contact'}

@pytest.mark.parametrize('raw', [None, '', '   '])
def test_missing_cost_adapter_precedes_journal_read(monkeypatch, raw):
    monkeypatch.setattr(cal, '_COST_MODEL', None)
    if raw is None:
        monkeypatch.delenv('MARKET_MOMENTUM_COST_MODEL', raising=False)
    else:
        monkeypatch.setenv('MARKET_MOMENTUM_COST_MODEL', raw)
    calls = []
    monkeypatch.setattr(cal, 'read_journal', lambda *a: calls.append(True))
    with pytest.raises(RuntimeError, match='explicitly configured'):
        cal.main(['--journal', 'synthetic.jsonl'])
    assert calls == []

def test_missing_journal_fails_before_cost_adapter_or_journal_read(monkeypatch):
    monkeypatch.setattr(cal, 'DEFAULT_JOURNAL', None)
    calls = []
    monkeypatch.setattr(cal, 'cost_model', lambda: calls.append('adapter'))
    monkeypatch.setattr(cal, 'read_journal', lambda *a: calls.append('journal'))
    with pytest.raises(SystemExit) as failure:
        cal.main([])
    assert failure.value.code == 2
    assert calls == []

@pytest.mark.parametrize('model', [SimpleNamespace(), SimpleNamespace(scanner_spread_cost=7, trade_cost=lambda row: 1)])
def test_invalid_adapter_interface_precedes_journal_read(monkeypatch, model):
    monkeypatch.setattr(cal, '_COST_MODEL', None)
    monkeypatch.setenv('MARKET_MOMENTUM_COST_MODEL', 'synthetic_cost_fixture')
    monkeypatch.setitem(sys.modules, 'synthetic_cost_fixture', model)
    calls = []
    monkeypatch.setattr(cal, 'read_journal', lambda *a: calls.append(True))
    with pytest.raises(RuntimeError, match='both cost callbacks'):
        cal.main(['--journal', 'synthetic.jsonl'])
    assert calls == []

def test_explicit_adapter_loads_and_memoizes_both_callbacks(monkeypatch):
    monkeypatch.setattr(cal, '_COST_MODEL', None)
    monkeypatch.setenv('MARKET_MOMENTUM_COST_MODEL', 'synthetic_cost_fixture')
    model = SimpleNamespace(scanner_spread_cost=lambda row: 3, trade_cost=lambda row: 1)
    monkeypatch.setitem(sys.modules, 'synthetic_cost_fixture', model)
    assert cal.cost_model() is model
    assert cal.cost_model() is model
    assert cal.default_cost({}) == 3.0

def test_gross_basis_needs_no_external_cost_adapter(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(cal, 'cost_model', lambda: calls.append('adapter'))
    monkeypatch.setattr(cal, 'calibrate', lambda *a, **k: {'synthetic': True})
    assert cal.main(['--journal', 'synthetic.jsonl', '--basis', 'gross_pnl', '--json']) == 0
    assert calls == []
    assert 'synthetic' in capsys.readouterr().out

@pytest.mark.parametrize('raw', [None, '', '   '])
def test_token_provider_missing_fails_before_import_or_http(monkeypatch, raw):
    if raw is None:
        monkeypatch.delenv('MARKET_MOMENTUM_TOKEN_PROVIDER', raising=False)
    else:
        monkeypatch.setenv('MARKET_MOMENTUM_TOKEN_PROVIDER', raw)
    calls = []
    monkeypatch.setattr(sd.importlib, 'import_module', lambda *a: calls.append('import'))
    with pytest.raises(sd.ScannerNetworkError, match='explicitly configured'):
        sd._default_token_fn()
    assert calls == []

@pytest.mark.parametrize('model', [SimpleNamespace(), SimpleNamespace(get_valid_token=3)])
def test_token_provider_interface_is_required(monkeypatch, model):
    monkeypatch.setenv('MARKET_MOMENTUM_TOKEN_PROVIDER', 'synthetic_token_fixture')
    monkeypatch.setitem(sys.modules, 'synthetic_token_fixture', model)
    with pytest.raises(sd.ScannerNetworkError, match='get_valid_token'):
        sd._default_token_fn()

def test_token_provider_callback_is_forwarded(monkeypatch):
    monkeypatch.setenv('MARKET_MOMENTUM_TOKEN_PROVIDER', 'synthetic_token_fixture')
    monkeypatch.setitem(sys.modules, 'synthetic_token_fixture', SimpleNamespace(get_valid_token=lambda: 'synthetic-fixture-value'))
    assert sd._default_token_fn() == 'synthetic-fixture-value'

@pytest.mark.parametrize('raw', [None, '', '   '])
def test_missing_rate_limiter_retains_fail_closed_default(monkeypatch, raw):
    if raw is None:
        monkeypatch.delenv('MARKET_MOMENTUM_RATE_LIMITER', raising=False)
    else:
        monkeypatch.setenv('MARKET_MOMENTUM_RATE_LIMITER', raw)
    calls = []
    monkeypatch.setattr(rg.importlib, 'import_module', lambda *a: calls.append('import'))
    assert rg.load_rate_limiter('synthetic.json') is None
    assert rg.acquire_discovery_budget(None, 1) is False
    assert calls == []

@pytest.mark.parametrize('model', [SimpleNamespace(), SimpleNamespace(SchwabRateLimiter=lambda state: SimpleNamespace())])
def test_invalid_rate_limiter_interface_remains_closed(monkeypatch, model):
    monkeypatch.setenv('MARKET_MOMENTUM_RATE_LIMITER', 'synthetic_rate_fixture')
    monkeypatch.setitem(sys.modules, 'synthetic_rate_fixture', model)
    assert rg.load_rate_limiter('synthetic.json') is None
    assert rg.acquire_discovery_budget(None, 1) is False

def test_explicit_rate_limiter_routes_existing_budget_call(monkeypatch):
    monkeypatch.setenv('MARKET_MOMENTUM_RATE_LIMITER', 'synthetic_rate_fixture')
    calls = []
    def factory(state):
        calls.append(('state', state))
        return SimpleNamespace(acquire=lambda **kw: calls.append(('budget', kw)) or True)
    monkeypatch.setitem(sys.modules, 'synthetic_rate_fixture', SimpleNamespace(SchwabRateLimiter=factory))
    limiter = rg.load_rate_limiter('synthetic.json')
    assert rg.acquire_discovery_budget(limiter, 2, now=7) is True
    assert calls == [('state', 'synthetic.json'), ('budget', {'n': 2, 'tier': 'discovery', 'now': 7})]

@pytest.mark.parametrize('raw', [None, '', '   '])
def test_missing_calendar_treats_every_day_as_closed(monkeypatch, raw):
    from datetime import date
    if raw is None:
        monkeypatch.delenv('MARKET_MOMENTUM_CALENDAR', raising=False)
    else:
        monkeypatch.setenv('MARKET_MOMENTUM_CALENDAR', raw)
    monkeypatch.setattr(dc, '_HOLIDAY_IMPORT_FAILED', False)
    calls = []
    monkeypatch.setattr(dc.importlib, 'import_module', lambda *a: calls.append('import'))
    assert dc._default_is_holiday(date(2026, 10, 1)) is True
    assert dc._HOLIDAY_IMPORT_FAILED is True
    assert calls == []

@pytest.mark.parametrize('model', [SimpleNamespace(), SimpleNamespace(is_market_holiday=3)])
def test_invalid_calendar_interface_remains_closed(monkeypatch, model):
    from datetime import date
    monkeypatch.setenv('MARKET_MOMENTUM_CALENDAR', 'synthetic_calendar_fixture')
    monkeypatch.setitem(sys.modules, 'synthetic_calendar_fixture', model)
    assert dc._default_is_holiday(date(2026, 10, 1)) is True

@pytest.mark.parametrize('is_holiday', [False, True])
def test_explicit_calendar_recovers_and_routes_callback(monkeypatch, is_holiday):
    from datetime import date
    monkeypatch.setenv('MARKET_MOMENTUM_CALENDAR', 'synthetic_calendar_fixture')
    monkeypatch.setattr(dc, '_HOLIDAY_IMPORT_FAILED', True)
    seen = []
    monkeypatch.setitem(sys.modules, 'synthetic_calendar_fixture', SimpleNamespace(is_market_holiday=lambda d: seen.append(d) or is_holiday))
    day = date(2026, 10, 1)
    assert dc._default_is_holiday(day) is is_holiday
    assert dc._HOLIDAY_IMPORT_FAILED is False
    assert seen == [day]

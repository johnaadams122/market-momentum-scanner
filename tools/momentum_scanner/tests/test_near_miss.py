from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from momentum_scanner import movers_scan
from momentum_scanner.adapters.schwab_quotes import QuoteSnapshot
ET = ZoneInfo("America/New_York"); NOW = datetime(2026,6,29,13,0,tzinfo=ET)
def _q(t,last,prev,vol,high,avg): return QuoteSnapshot(t,last,prev,vol,high,avg,NOW.astimezone(timezone.utc),"regular")

def test_gap_near_miss_emitted_without_changing_hits():
    sink=[]
    quotes={"X": _q("X",4.16,4.0,1_500_000,4.2,500_000)}   # gap 0.04 in [0.03,0.05); vol+relvol pass
    hits=movers_scan.tier1_filter(quotes, NOW, premarket=False, near_miss=sink.append)
    assert hits==[]                                        # decision unchanged (gap<0.05)
    assert any(r["reason"]=="gap" and r["ticker"]=="X" for r in sink)

def test_no_near_miss_for_full_pass():
    sink=[]
    quotes={"A": _q("A",5.0,4.0,1_500_000,5.1,500_000)}    # a real hit
    hits=movers_scan.tier1_filter(quotes, NOW, premarket=False, near_miss=sink.append)
    assert len(hits)==1 and sink==[]

def test_float_near_miss_emitted():
    sink=[]
    # Tier1Hit fields: ticker,last,prev_close,gap_pct,total_volume,day_high,avg_daily_volume,rel_volume,quote_time,source
    hit=movers_scan.Tier1Hit("Y",5.0,4.0,0.25,1_500_000,5.0,500_000,5.5,NOW.astimezone(timezone.utc),"regular")
    movers_scan.tier2_enrich([hit], float_cache={}, float_fetch_fn=lambda t: 50_000_000,
                             now_utc=NOW.astimezone(timezone.utc), premarket=False, near_miss=sink.append)
    assert any(r["reason"]=="float" and r["ticker"]=="Y" and r["float_shares"]==50_000_000 for r in sink)

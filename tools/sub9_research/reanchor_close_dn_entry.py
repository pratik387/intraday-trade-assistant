import json, asyncio, sys
import pandas as pd, numpy as np
sys.path.insert(0,'.')
from broker.upstox.upstox_data_client import UpstoxDataClient

P=pd.DataFrame(json.load(open('state/decay_tripwire_close_dn_overnight_long.json'))['trades'])
P['ts']=pd.to_datetime(P['ts_iso'],format='ISO8601',errors='coerce')
R=P[P.exit_reason=='reconstructed_paper'].copy()
R['exit_date']=R.ts.dt.normalize()
R['notional']=R.entry_price*R.qty
R['fee_pct']=100*R.fees_inr/R.notional
R['old_net_pct']=100*R.net_pnl_inr/R.notional
syms=sorted(R.symbol.unique())
lo=(R.exit_date.min()-pd.Timedelta(days=6)).strftime('%Y-%m-%d')
hi=R.exit_date.max().strftime('%Y-%m-%d')
print('re-anchoring %d recon trades | %d symbols | %s..%s'%(len(R),len(syms),lo,hi), flush=True)
sdk=UpstoxDataClient()
m=asyncio.run(sdk.async_fetch_historical_5m_batch(syms, lo, hi, concurrency=8, rps=8.0))
print('fetched %d/%d symbols'%(sum(1 for v in m.values() if v is not None and len(v)), len(syms)), flush=True)

rows=[]
for r in R.itertuples():
    df=m.get(r.symbol)
    if df is None or not len(df): continue
    idx=pd.to_datetime(df.index)
    days=sorted(set(idx.normalize()))
    prior=[d for d in days if d < r.exit_date]
    if not prior: continue
    entry_day=prior[-1]
    day=df[idx.normalize()==entry_day]
    b=day[pd.to_datetime(day.index).strftime('%H:%M')=='15:25']
    if b.empty: continue
    open_1525=float(b['open'].iloc[0]); close_1530=float(b['close'].iloc[0])
    rows.append(dict(symbol=r.symbol, entry_day=entry_day.date(),
                     paper_entry=r.entry_price, open_1525=open_1525, close_1530=close_1530,
                     exit_price=r.exit_price, fee_pct=r.fee_pct, old_net_pct=r.old_net_pct))
D=pd.DataFrame(rows)
print('matched %d of %d trades to a 15:25 bar'%(len(D),len(R)))
D['new_net_pct']=100*(D.exit_price/D.open_1525-1)-D.fee_pct
D['recon_check_pct']=100*(D.exit_price/D.close_1530-1)-D.fee_pct
D['intrabar_bp']=1e4*(D.close_1530/D.open_1525-1)
def pf(x):
    w=x[x>0].sum(); l=-x[x<0].sum(); return w/l if l else float('inf')
print()
print('sanity: recon anchor reproduced? mean |diff| vs ledger = %.4f pp'%(D.recon_check_pct-D.old_net_pct).abs().mean())
print()
print('%-46s %5s %9s %8s %7s'%('entry anchor','n','mean%','PF','win%'))
print('%-46s %5d %+8.3f%% %8.3f %6.1f%%'%('15:30 close  (paper, UNREACHABLE)',len(D),D.recon_check_pct.mean(),pf(D.recon_check_pct),100*(D.recon_check_pct>0).mean()))
print('%-46s %5d %+8.3f%% %8.3f %6.1f%%'%('15:25 open   (ACHIEVABLE, ~live)',len(D),D.new_net_pct.mean(),pf(D.new_net_pct),100*(D.new_net_pct>0).mean()))
print()
print('cost of the unreachable anchor: %+.3f%%/trade'%(D.new_net_pct.mean()-D.recon_check_pct.mean()))
print('final-5min move (15:25 open -> 15:30 close): median %+.1f bp  mean %+.1f bp'%(D.intrabar_bp.median(),D.intrabar_bp.mean()))
print('  fell into the close: %.0f%% of fires'%(100*(D.intrabar_bp<0).mean()))

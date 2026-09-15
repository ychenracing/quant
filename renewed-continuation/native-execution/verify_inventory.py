from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import json, math, sys
from pathlib import Path
import pandas as pd

@dataclass
class Gate:
    sellable: dict[str,int]
    deferred: dict[str,tuple[int,str]]

    @classmethod
    def open(cls, beginning: dict[str,int], deferred=None):
        return cls(dict(beginning), dict(deferred or {}))

    def request(self, symbol: str, requested: int, held: int, reason: str):
        requested=max(0,min(int(requested),int(held)))
        available=max(0,int(self.sellable.get(symbol,0)))
        filled=min(requested,available)
        self.sellable[symbol]=available-filled
        blocked=requested-filled
        if blocked:
            prev=self.deferred.get(symbol)
            if prev is None or blocked>prev[0]:
                self.deferred[symbol]=(blocked,reason)
        return filled,blocked

def audit_t1(path: Path):
    df=pd.read_csv(path)
    df['day']=df.date.astype(str).str[:10]
    holdings=defaultdict(int); violations=[]
    for day,g in df.groupby('day',sort=False):
        start=dict(holdings); sold=defaultdict(int)
        for r in g.itertuples(index=False):
            q=int(r.quantity)
            if r.side=='BUY':
                holdings[r.symbol]+=q
            else:
                avail=max(0,start.get(r.symbol,0)-sold[r.symbol])
                if q>avail:
                    violations.append({'day':day,'symbol':r.symbol,'quantity':q,'available':avail,
                                       'excess':q-avail,'reason':r.reason,'price':float(r.price)})
                sold[r.symbol]+=q
                holdings[r.symbol]-=q
                if holdings[r.symbol]==0: holdings.pop(r.symbol,None)
    return df,violations,dict(holdings)

def replay_cash(trades: pd.DataFrame, settings: dict, initial=2_000_000.0):
    cash=initial; hold=defaultdict(int)
    for r in trades.itertuples(index=False):
        q=int(r.quantity); v=q*float(r.price)
        if r.side=='BUY':
            cash-=v*(1+settings['fee']); hold[r.symbol]+=q
        else:
            cash+=v*(1-settings['fee']-settings['stamp_tax']); hold[r.symbol]-=q
            if hold[r.symbol]==0: hold.pop(r.symbol,None)
    return cash,dict(hold)

def semantic_cases():
    # Old inventory remains sellable after a same-day top-up; the top-up itself does not.
    g=Gate.open({'X':1000}); held=1500
    f,b=g.request('X',1200,held,'protect'); assert (f,b)==(1000,200)
    # Repeated same-session requests cannot consume fresh inventory and do not add duplicate obligation.
    f2,b2=g.request('X',500,500,'protect'); assert (f2,b2)==(0,500); assert g.deferred['X'][0]==500
    # Partial fill then next-session release/retry.
    g2=Gate.open({'X':500},g.deferred); f3,b3=g2.request('X',g2.deferred['X'][0],500,'protect'); assert (f3,b3)==(500,0)
    # Brand-new same-day inventory has zero sellable quantity.
    g3=Gate.open({}); f4,b4=g3.request('Y',400,400,'hard'); assert (f4,b4)==(0,400)
    return {'old_inventory_after_topup':'PASS','repeated_attempts':'PASS','partial_fill':'PASS',
            'next_session_retry':'PASS','fresh_inventory_locked':'PASS'}

def main(root: Path):
    control=root/'native-control'; corrected=root/'native-corrected2'
    c,cv,ch=audit_t1(control/'trades.csv'); n,nv,nh=audit_t1(corrected/'trades.csv')
    cs=json.load(open(control/'summary.json')); ns=json.load(open(corrected/'summary.json'))
    settings=json.load(open(corrected/'settings.json'))
    cash,hold=replay_cash(n,settings)
    final=ns['wealth']*2_000_000.0
    ce=pd.read_csv(control/'equity.csv'); ne=pd.read_csv(corrected/'equity.csv')
    prefix_mask=ce.date.astype(str).str[:10]<'2025-04-28'
    prefix_equity_equal=ce.loc[prefix_mask].reset_index(drop=True).equals(ne.loc[prefix_mask].reset_index(drop=True))
    prefix_trades_equal=c.iloc[:57].drop(columns=['day']).reset_index(drop=True).equals(n.iloc[:57].drop(columns=['day']).reset_index(drop=True))
    shifted=n[(n.date.astype(str).str.startswith('2025-04-29')) & (n.symbol=='sz300502') & (n.side=='SELL')]
    result={
      'control':{'wealth':cs['wealth'],'max_drawdown':cs['max_drawdown'],'rows':len(c),'t1_violations':cv},
      'corrected':{'wealth':ns['wealth'],'max_drawdown':ns['max_drawdown'],'rows':len(n),'t1_violations':nv},
      'delta':{'wealth_multiple':ns['wealth']-cs['wealth'],'relative_wealth':ns['wealth']/cs['wealth']-1,
               'max_drawdown_points':(ns['max_drawdown']-cs['max_drawdown'])*100},
      'specific_case':{'control_day':'2025-04-28','symbol':'sz300502','quantity':52100,
                       'corrected_retry_day':'2025-04-29','retry_found': bool(len(shifted)==1 and int(shifted.iloc[0].quantity)==52100 and shifted.iloc[0].reason=='dd_hard_limit')},
      'unchanged_prefix':{'trade_rows_before_divergence':57,'trades_exact':bool(prefix_trades_equal),
                          'equity_through_2025_04_27_exact':bool(prefix_equity_equal)},
      'cash_reconciliation':{'replayed_cash':cash,'reported_final':final,'difference':cash-final,
                             'ending_holdings':hold,'pass':abs(cash-final)<1e-6 and not hold},
      'semantic_cases':semantic_cases(),
      'classification':'EXECUTION_CORRECTNESS_ONLY_NOT_ECONOMIC_ACCEPTANCE',
      'limitations':['Only workbuddy common-five inventory T+1 was corrected.',
                     'Intraday information timing, fee/liquidity semantics, risk-state resets beyond the necessary deferred breaker completion, corporate actions, and the other references remain non-normalized.',
                     'Private reference source is not published.']
    }
    assert len(cv)==1 and cv[0]['excess']==52100 and cv[0]['day']=='2025-04-28'
    assert not nv and not nh and result['specific_case']['retry_found']
    assert prefix_trades_equal and prefix_equity_equal and result['cash_reconciliation']['pass']
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main(Path(sys.argv[1] if len(sys.argv)>1 else '.'))

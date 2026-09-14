"""Reconstruct old close decisions and classify actual campaign-exit causes."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from techquant.data import load_market,file_hash
from techquant.evidence import load_result
from techquant.policy import CloseObservation
from research.observed_admission_completion import Owner,Parameters
from research.ledger_attribution import attribute

def diagnose(parent,output,data,supplement=None):
    BASE=parent
    OUT=output;OUT.mkdir(parents=True,exist_ok=True)
    market=load_market(data,supplement=supplement,sectors=json.loads(Path('research/catalog.json').read_text())['sectors']).prefix('2025-12-31')
    summary=[]
    for scope in ['union','chatgpt_5','joint_optical_leader_removal']:
     p=BASE/'selection/runs'/('candidate0_'+scope);identity=json.loads((p/'identity.json').read_text());r=load_result(p,expected=identity);m=market.subset(identity['universe']);owner=Owner(m,Parameters());inn=owner.inner
     units=np.zeros(len(m.symbols));anchor=np.zeros(len(m.symbols));fills={str(d.date()):[] for d in m.calendar}
     for o in r.orders:
      if o['status']=='FILLED':fills[o['date']].append(o)
     records=[];maxdiff=0.;prior_pending=np.zeros(len(m.symbols),bool)
     for i,date in enumerate(m.calendar):
      day=str(date.date());prior=units.copy()
      for order in fills[day]:
       j=m.symbols.index(order['symbol']);units[j]+=float(order['units'])*(1 if order['side']=='BUY' else -1)
       if units[j]<1e-8:units[j]=0.
      price=np.nan_to_num(inn.features.close[i],nan=0.);opened=(units>1e-10)&(prior<=1e-10);anchor[opened]=price[opened];anchor[units<=1e-10]=0.
      e=r.equity.loc[date];obs=CloseObservation.from_inventory(i,day,float(e.nav),float(e.cash),units,units*price/e.nav);d=owner.decide(obs)
      diff=float(np.max(np.abs(d.weights-r.targets.loc[date].to_numpy())));maxdiff=max(maxdiff,diff);assert diff<1e-13,(scope,day,diff)
      stop_broken=(price<=inn.stop)&(units>1e-10);price_exit=inn.features.exit[i];stale=~inn.ready[i]
      full=(units>1e-10)&(d.unit_targets<=1e-10)
      first=full &~prior_pending
      for j in np.flatnonzero(first):
       maturity=bool(inn.stop[j]>=anchor[j] and anchor[j]>0)
       stop_only=bool(stop_broken[j] and not price_exit[j] and not stale[j] and inn.risk.cap>0)
       if i+40<len(m.calendar) and i+1<len(m.calendar):
        nextopen=m.panel('open').iloc[i+1,j];after=m.panel('close').ffill().iloc[i+1:i+41,j]
        forward=float(after.iloc[-1]/nextopen-1);maxloss=float((1-after/after.cummax()).max())
       else:forward=maxloss=None
       records.append(dict(date=day,symbol=m.symbols[j],actual_units=units[j],weight=units[j]*price[j]/e.nav,price=price[j],stop=inn.stop[j],entry_close_anchor=anchor[j],stop_broken=bool(stop_broken[j]),price_exit=bool(price_exit[j]),stale=bool(stale[j]),mature=maturity,stop_only=stop_only,risk_cap=inn.risk.cap,reason=d.reason,subsequent40_price_return=forward,subsequent40_price_drawdown=maxloss))
      prior_pending=full.copy()
     df=pd.DataFrame(records);df.to_csv(OUT/(scope+'_full_exit_intents.csv'),index=False,float_format='%.17g')
     days,episodes,ledger=attribute(m,r);episodes.to_csv(OUT/(scope+'_episodes.csv'),index=False,float_format='%.17g')
     for mask,label in [(df.stop_only&df.mature,'mature_stop_only'),(df.stop_only&~df.mature,'immature_stop_only'),(~df.stop_only,'other_exits')]:
      q=df[mask];values=q.subsequent40_price_return.dropna();summary.append(dict(scope=scope,kind=label,count=len(q),fully_matured_outcomes=len(values),mean_forward=float(values.mean()) if len(values) else None,positive_count=int((values>0).sum()),max_target_error=maxdiff))
     print('\n',scope,'ALL EXITS',len(df),'MATURE STOP ONLY');print(df[df.stop_only&df.mature][['date','symbol','weight','price','stop','entry_close_anchor','subsequent40_price_return']].to_string(index=False))
    print(json.dumps(summary,indent=2));(OUT/'summary.json').write_text(json.dumps({'source':identity['source']['commit'],'status':'OBSERVATIONAL_NOT_CAUSAL_RETURN_OR_ACCEPTANCE','new_portfolio_runs':0,'selection_end':'2025-12-31','summary':summary},indent=2)+'\n');(OUT/'MANIFEST.json').write_text(json.dumps({p.name:file_hash(p) for p in OUT.iterdir() if p.name!='MANIFEST.json'},indent=2)+'\n')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('parent','output','data'):parser.add_argument('--'+field,type=Path,required=True)
    parser.add_argument('--supplement',type=Path)
    a=parser.parse_args();diagnose(a.parent,a.output,a.data,a.supplement)

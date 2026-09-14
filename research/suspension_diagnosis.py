"""Reconstruct risk-only suspensions on immutable observed-admission accounts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from techquant.data import load_market,file_hash
from techquant.evidence import load_result
from techquant.policy import CloseObservation
from research.observed_admission_completion import Owner,Parameters

def diagnose(parent,output,data,supplement=None):
    if (parent/'source-commit.txt').read_text().strip() != SOURCE:
        raise ValueError('wrong historical parent source')
    for name,digest in json.loads((parent/'MANIFEST.json').read_text()).items():
        if file_hash(parent/name) != digest:
            raise ValueError('historical parent manifest mismatch: '+name)
    BASE=parent
    OUT=output;OUT.mkdir(parents=True,exist_ok=True)
    market=load_market(data,supplement=supplement,sectors=json.loads(Path('research/catalog.json').read_text())['sectors']).prefix('2025-12-31')
    summaries=[]
    for scope in ['union','chatgpt_5','joint_optical_leader_removal']:
     path=BASE/'selection/runs'/('candidate0_'+scope);identity=json.loads((path/'identity.json').read_text());r=load_result(path,expected=identity);m=market.subset(identity['universe']);owner=Owner(m,Parameters());p=owner.inner;n=len(m.symbols)
     units=np.zeros(n);suspended=np.zeros(n);frozen_stop=np.zeros(n);signal=['']*n;events=[];states=[];error=0.;previous_full=np.zeros(n,bool)
     fast=m.panel('close').ffill().rolling(p.config.fast,min_periods=p.config.fast).mean().to_numpy();healthy=np.zeros(n,int)
     fills={str(d.date()):[] for d in m.calendar}
     for o in r.orders:
      if o['status']=='FILLED':fills[o['date']].append(o)
     for i,date in enumerate(m.calendar):
      day=str(date.date());previous=units.copy()
      for order in fills[day]:
       j=m.symbols.index(order['symbol']);units[j]+=float(order['units'])*(1 if order['side']=='BUY' else -1)
       if units[j]<1e-8:units[j]=0.
      held=units>1e-10;opened=held&(previous<=1e-10);price=np.nan_to_num(p.features.close[i],nan=0.)
      invalid=(price<=frozen_stop)|p.features.exit[i]|~p.ready[i]
      for j in np.flatnonzero((suspended>0)&(opened|invalid)):
       events.append(dict(date=day,symbol=m.symbols[j],signal=signal[j],kind='ACTUAL_REACQUIRED' if opened[j] else 'TREND_INVALID',price=price[j],stop=frozen_stop[j],suspended_units=suspended[j]))
      suspended[opened|invalid]=0.;frozen_stop[opened|invalid]=0.
      health=p.ready[i]&~p.features.exit[i]&(price>fast[i]);healthy=np.where(health,healthy+1,0)
      e=r.equity.loc[date];o=CloseObservation.from_inventory(i,day,float(e.nav),float(e.cash),units,units*price/e.nav);d=owner.decide(o)
      diff=float(np.max(np.abs(d.weights-r.targets.loc[date].to_numpy())));error=max(error,diff);assert diff<1e-13,(scope,day,diff)
      full=held&(d.unit_targets<=1e-10);fresh=full&~previous_full
      risk_only=fresh&(p.risk.cap==0)&(price>p.stop)&~p.features.exit[i]&p.ready[i]
      for j in np.flatnonzero(risk_only):
       suspended[j]=units[j];frozen_stop[j]=p.stop[j];signal[j]=day
       events.append(dict(date=day,symbol=m.symbols[j],signal=day,kind='RISK_ONLY_FULL_CUT',price=price[j],stop=frozen_stop[j],suspended_units=suspended[j]))
      for j in np.flatnonzero(suspended>0):
       veto=bool(p.readmit[j]);entry=bool(p.features.entry[i,j]);score=float(p.features.score[i,j]);cut=bool(np.any(d.unit_targets<units-1e-10));cash=float(e.cash/e.nav);slots=int(held.sum())
       candidate=bool(~held[j] and health[j] and healthy[j]>=p.config.recovery and score>0 and p.risk.cap>0 and not cut and slots<2 and cash>.01 and not p.exit_pending[j] and not np.isfinite(p.reduction_ceiling[j]))
       states.append(dict(date=day,symbol=m.symbols[j],signal=signal[j],actual_units=units[j],target_units=d.unit_targets[j],stop=frozen_stop[j],price=price[j],health=bool(health[j]),health_closes=int(healthy[j]),entry=entry,readmit=veto,score=score,risk_cap=p.risk.cap,cash_fraction=cash,occupied=slots,cut=cut,restoration_witness=candidate,held_symbols=' '.join(np.asarray(m.symbols)[held])))
      previous_full=full.copy()
     df=pd.DataFrame(states);ev=pd.DataFrame(events);df.to_csv(OUT/(scope+'_states.csv'),index=False,float_format='%.17g');ev.to_csv(OUT/(scope+'_events.csv'),index=False,float_format='%.17g')
     q=df[df.restoration_witness & (df.target_units<=1e-10)];q.to_csv(OUT/(scope+'_witnesses.csv'),index=False,float_format='%.17g')
     print(scope,'risk-only cuts',int((ev.kind=='RISK_ONLY_FULL_CUT').sum()),'all structural witnesses',int(df.restoration_witness.sum()),'missed buys',len(q),'events',len(q.groupby(['signal','symbol'])))
     print(q.groupby(['signal','symbol']).first()[['date','health_closes','entry','readmit','cash_fraction','held_symbols']].to_string())
     summaries.append(dict(scope=scope,max_target_error=error,cuts=int((ev.kind=='RISK_ONLY_FULL_CUT').sum()),structural_witnesses=int(df.restoration_witness.sum()),no_purchase_witnesses=len(q),campaigns_with_no_purchase_witness=len(q.groupby(['signal','symbol'])),entry_missing=int((~q.entry).sum()),readmit_veto=int(q.readmit.sum())))
    (OUT/'summary.json').write_text(json.dumps(dict(status='HISTORICAL_STATE_ATTRIBUTION_NOT_TREATMENT_RETURN',source=identity['source']['commit'],new_portfolio_runs=0,research_end='2025-12-31',scopes=summaries,method_sha256=file_hash(Path(__file__)),interpreter_source_sha256=file_hash(BASE/'source.tar.gz')),indent=2)+'\n')
    (OUT/'MANIFEST.json').write_text(json.dumps({p.name:file_hash(p) for p in sorted(OUT.iterdir()) if p.name!='MANIFEST.json'},indent=2)+'\n')


SOURCE='18901a431247a4f73aff08e73f2d0a7b3025de44'
ARCHIVE='c836b4b4ea4c8d22f4a8a168a97732e70b0fb48b3c136d4077767a65b6121285'


def hosted(output, data, supplement):
    """Interpret old observations using the exact archived old source, not HEAD."""
    import os,sys,subprocess,tarfile,urllib.request
    restore=output.parent/'suspension-parent';restore.mkdir(parents=True,exist_ok=True)
    archive=restore/'parent.tar.gz'
    url=f'https://raw.githubusercontent.com/ychenracing/quant/research/evidence/nonlinear/{SOURCE}/34868483244/nonlinear-evidence.tar.gz'
    with urllib.request.urlopen(url,timeout=60) as response:archive.write_bytes(response.read())
    if file_hash(archive)!=ARCHIVE:raise ValueError('historical parent archive mismatch')
    with tarfile.open(archive) as bundle:bundle.extractall(restore,filter='data')
    parent=restore/'nonlinear';old_source=restore/'source';old_source.mkdir(exist_ok=True)
    with tarfile.open(parent/'source.tar.gz') as bundle:bundle.extractall(old_source,filter='data')
    env=dict(os.environ,PYTHONPATH=str(old_source/'src')+os.pathsep+str(old_source))
    command=[sys.executable,str(Path(__file__).resolve()),'--parent',str(parent),
        '--output',str(output/'suspension_diagnosis'),'--data',str(data)]
    if supplement is not None:command+=['--supplement',str(supplement)]
    subprocess.run(command,cwd=old_source,env=env,check=True,timeout=90)
    command[1]=str(Path(__file__).with_name('holding_diagnosis.py').resolve())
    command[command.index('--output')+1]=str(output/'holding_diagnosis')
    subprocess.run(command,cwd=old_source,env=env,check=True,timeout=90)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('parent','output','data'):parser.add_argument('--'+field,type=Path,required=True)
    parser.add_argument('--supplement',type=Path)
    args=parser.parse_args();diagnose(args.parent,args.output,args.data,args.supplement)

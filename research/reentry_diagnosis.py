"""Reconstruct an immutable parent's pre-2026 reentry witnesses, not new trades."""
from pathlib import Path
import hashlib
import json
import tarfile
import urllib.request
import numpy as np
import pandas as pd
from techquant.data import file_hash, load_market
from techquant.evidence import load_result
from techquant.policy import CloseObservation
from research.ledger_attribution import attribute
from research.observed_admission_completion import Owner, Parameters

SOURCE = '18901a431247a4f73aff08e73f2d0a7b3025de44'
RUN = '34868483244'
ARCHIVE = 'c836b4b4ea4c8d22f4a8a168a97732e70b0fb48b3c136d4077767a65b6121285'


def diagnose(root, market, output):
    output.mkdir(parents=True, exist_ok=True)
    for name, digest in json.loads((root/'MANIFEST.json').read_text()).items():
        if file_hash(root/name) != digest:
            raise ValueError('parent manifest mismatch: '+name)
    if (root/'source-commit.txt').read_text().strip() != SOURCE:
        raise ValueError('wrong parent source')
    # The accounting and owner code used to interpret historical observations
    # must match the archived source. Do not demand the old execution engine:
    # no portfolio run is performed, and that engine has a preserved later fix.
    names = ['research/'+n+'.py' for n in ('ledger_attribution','observed_admission_completion',
        'observed_readiness','admission_budget_completion','quantity_obligation','support_budget','funded_risk')]
    with tarfile.open(root/'source.tar.gz') as source:
        for name in names:
            stream=source.extractfile(name)
            if stream is None or hashlib.sha256(stream.read()).hexdigest()!=file_hash(Path(name)):
                raise ValueError('historical interpreter mismatch: '+name)
    market=market.prefix('2025-12-31');summaries=[]
    for scope in ('union','chatgpt_5','joint_optical_leader_removal'):
        p=root/'selection/runs'/('candidate0_'+scope)
        identity=json.loads((p/'identity.json').read_text())
        if identity['source']['commit']!=SOURCE or identity['end']!='2025-12-31':
            raise ValueError('wrong candidate identity')
        result=load_result(p,expected=identity);m=market.subset(identity['universe'])
        days,episodes,ledger=attribute(m,result)
        days.to_csv(output/(scope+'_symbol_days.csv'),index=False,float_format='%.17g')
        episodes.to_csv(output/(scope+'_episodes.csv'),index=False,float_format='%.17g')
        owner=Owner(m,Parameters());units=np.zeros(len(m.symbols));jmap={s:j for j,s in enumerate(m.symbols)}
        fills={str(d.date()):[] for d in m.calendar}
        for order in result.orders:
            if order['status']=='FILLED':fills[order['date']].append(order)
        fast=m.panel('close').ffill().rolling(owner.inner.config.fast,min_periods=owner.inner.config.fast).mean().to_numpy()
        healthy=np.zeros(len(m.symbols),dtype=int);records=[];max_error=0.
        for i,date in enumerate(m.calendar):
            day=str(date.date());prior=units.copy()
            for order in fills[day]:
                j=jmap[order['symbol']];q=float(order['units'])
                units[j]=units[j]+q if order['side']=='BUY' else max(0.,units[j]-q)
            e=result.equity.loc[date];f=owner.inner.features;price=np.nan_to_num(f.close[i],nan=0.)
            obs=CloseObservation.from_inventory(i,day,float(e.nav),float(e.cash),units,units*price/e.nav)
            dec=owner.decide(obs);error=float(np.abs(dec.weights-result.targets.loc[date].to_numpy()).max());max_error=max(max_error,error)
            if error>=1e-13:raise ValueError('historical decision mismatch: '+day)
            sold=(prior>1e-10)&(units<=1e-10);healthy[sold]=0
            health=owner.inner.ready[i]&~f.exit[i]&(price>fast[i]);healthy=np.where(health,healthy+1,0)
            protected=bool(np.any(dec.unit_targets<units-1e-9))
            for j,symbol in enumerate(m.symbols):
                witness=bool(owner.inner.readmit[j] and f.entry[i,j] and not f.exit[i,j]
                    and owner.inner.ready[i,j] and f.score[i,j]>0 and units[j]<=1e-10 and healthy[j]>=3
                    and owner.inner.risk.cap>0 and e.cash/e.nav>.01 and np.sum(units>1e-10)<2 and not protected)
                records.append(dict(date=day,symbol=symbol,actual_units=units[j],declared_units=dec.unit_targets[j],
                    health=bool(health[j]),healthy_closes=int(healthy[j]),readmit=bool(owner.inner.readmit[j]),
                    entry=bool(f.entry[i,j]),score=float(f.score[i,j]),cash_fraction=e.cash/e.nav,
                    risk_cap=owner.inner.risk.cap,protected=protected,structural_witness=witness))
        states=pd.DataFrame(records);states.to_csv(output/(scope+'_decisions.csv'),index=False,float_format='%.17g')
        states[states.structural_witness].to_csv(output/(scope+'_witnesses.csv'),index=False,float_format='%.17g')
        summaries.append(dict(scope=scope,source=SOURCE,ledger=ledger,max_target_error=max_error,
            structural_witnesses=int(states.structural_witness.sum()),wealth=float(result.equity.nav.iloc[-1]/identity['config']['initial_cash'])))
    report=dict(status='HISTORICAL_ATTRIBUTION_NOT_ECONOMIC_ACCEPTANCE',source=SOURCE,run=RUN,
        archive_sha256=ARCHIVE,research_end='2025-12-31',new_portfolio_runs=0,
        method='Reconcile actual filled inventory/PnL and reproduce recorded parent decisions; structural witnesses are not guaranteed rank/risk/cash/sector/execution-funded orders or causal treatment returns.',scopes=summaries)
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'MANIFEST.json').write_text(json.dumps({p.name:file_hash(p) for p in sorted(output.iterdir()) if p.name!='MANIFEST.json'},indent=2)+'\n')
    return report


def hosted(output, data, supplement):
    folder=output.parent/'reentry-parent';folder.mkdir(parents=True,exist_ok=True)
    archive=folder/'evidence.tar.gz'
    url=f'https://raw.githubusercontent.com/ychenracing/quant/research/evidence/nonlinear/{SOURCE}/{RUN}/nonlinear-evidence.tar.gz'
    with urllib.request.urlopen(url,timeout=60) as response:archive.write_bytes(response.read())
    if file_hash(archive)!=ARCHIVE:raise ValueError('parent archive identity mismatch')
    with tarfile.open(archive) as bundle:bundle.extractall(folder,filter='data')
    catalog=json.loads(Path(__file__).with_name('catalog.json').read_text())
    market=load_market(data,supplement=supplement,sectors=catalog['sectors'])
    return diagnose(folder/'nonlinear',market,output/'reentry_diagnosis')

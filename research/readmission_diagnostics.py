"""Reconstruct pinned prior decisions; do not refit or rerun the old strategy grid."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import urllib.request

PARENT='965f30c64f3230644beb31109a757cdd716486a9'
RUN='34836507473'
ARCHIVE='c64e6ee882dbca9583573a5fc2da8f1e36bd7bca830b19a205c43ae76189e02f'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_pinned(data, supplement, cache, output):
    cache=Path(cache).resolve();output=Path(output).resolve()
    cache.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    archive=cache/'completion.tar.gz'
    if not archive.exists():
        url=('https://raw.githubusercontent.com/ychenracing/quant/research/evidence/nonlinear/'
             +PARENT+'/'+RUN+'/nonlinear-evidence.tar.gz')
        with urllib.request.urlopen(url,timeout=60) as response:
            archive.write_bytes(response.read())
    if digest(archive)!=ARCHIVE:
        raise ValueError('pinned completion archive hash mismatch')
    with tarfile.open(archive) as bundle:
        bundle.extractall(cache,filter='data')
    evidence=cache/'nonlinear'
    manifest=json.loads((evidence/'MANIFEST.json').read_text())
    for name,expected in manifest.items():
        path=(evidence/name).resolve()
        if not path.is_relative_to(evidence) or digest(path)!=expected:
            raise ValueError('pinned completion manifest mismatch: '+name)
    if (evidence/'source-commit.txt').read_text().strip()!=PARENT:
        raise ValueError('pinned completion source mismatch')
    source=cache/'source';source.mkdir(exist_ok=True)
    with tarfile.open(evidence/'source.tar.gz') as bundle:
        bundle.extractall(source,filter='data')
    env=dict(os.environ,PYTHONPATH=os.pathsep.join(str(source/p) for p in ('src','tests','.')),
             QUANT_SOURCE_COMMIT=PARENT)
    # Import the original source in an isolated interpreter, not today's edited
    # parent. New generator identity is recorded separately from old policy identity.
    with (output/'reconstruction.txt').open('w') as log:
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--parent',str(evidence),
            '--data',str(Path(data).resolve()),'--supplement',str(Path(supplement).resolve()),
            '--output',str(output)],cwd=source,env=env,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=90)
    receipt={'parent_source':PARENT,'run_id':RUN,'archive_sha256':ARCHIVE,
             'verified_manifest_files':len(manifest),'generator_sha256':digest(Path(__file__)),
             'generator_source':os.environ.get('QUANT_SOURCE_COMMIT','UNBOUND_LOCAL_SNAPSHOT'),
             'status':'DIAGNOSTIC_NOT_ECONOMIC_ACCEPTANCE'}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


def reconstruct(root,data,supplement,out):
    from dataclasses import asdict
    import json,numpy as np,pandas as pd
    from techquant.data import load_market,file_hash
    from techquant.evidence import load_result,source_identity
    from techquant.policy import CloseObservation
    from research.expectation_study import scopes
    from research.ledger_attribution import attribute
    from research.completion import Owner,Parameters
    cat=json.loads(Path('research/catalog.json').read_text());market=load_market(data,supplement=supplement,sectors=cat['sectors']).prefix('2025-12-31');out.mkdir(exist_ok=True)
    summary={}
    reentry_witness=[]
    for scope,names in scopes(market,cat).items():
     m=market.subset(names);folder=root/'selection/runs'/('candidate1_'+scope);meta=json.loads((folder/'identity.json').read_text());wrapper=Owner(m,Parameters('support'));owner=wrapper.owner
     assert wrapper.identity()==meta['policy'];assert m.fingerprint()==meta['data_sha256'];assert meta['source']['package_sha256']==source_identity()['package_sha256']
     assert all(file_hash(Path('research')/n)==h for n,h in meta['study']['dependencies'].items())
     r=load_result(folder,expected=meta);daily,episodes,att=attribute(m,r);episodes.to_csv(out/(scope+'_episodes.csv'),index=False,float_format='%.17g')
     daily.to_csv(out/(scope+'_pnl.csv'),index=False,float_format='%.17g')
     u=np.zeros(len(m.symbols));idx={s:j for j,s in enumerate(m.symbols)};fills={};entered=np.zeros(len(u));entry_risk=np.zeros(len(u));owned_peak=np.zeros(len(u));witness=[];rows=[]
     for f in r.orders:
      if f['status']=='FILLED':fills.setdefault(f['date'],[]).append(f)
     price=m.panel('close').ffill().to_numpy();eligible_cash=0; recovery_health=np.zeros(len(u),dtype=int); fast=m.panel("close").ffill().rolling(owner.config.fast,min_periods=owner.config.fast).mean().to_numpy()
     for i,date in enumerate(m.calendar):
      previous=u.copy()
      for f in fills.get(str(date.date()),[]):
       j=idx[f['symbol']];u[j]+=(1 if f['side']=='BUY' else -1)*f['units']
      u[np.abs(u)<1e-7]=0.;new=(u>1e-10)&(previous<=1e-10);flat=u<=1e-10
      entered[new]=price[i,new];entry_risk[new]=np.maximum(price[i,new]-owner.pending_stop[new],.02*price[i,new]);owned_peak[flat]=entered[flat]=entry_risk[flat]=0.;owned_peak=np.maximum(owned_peak,np.where(flat,0,price[i]));mfe=np.divide(owned_peak-entered,entry_risk,out=np.zeros_like(u),where=entry_risk>0)
      nav,cash=map(float,r.equity.loc[date,['nav','cash']]);value=np.nan_to_num(u*price[i],nan=0.);o=CloseObservation.from_inventory(i,str(date.date()),nav,cash,u,value/nav)
      recovery_health[(previous>1e-10)&flat]=0
      healthy=owner.ready[i]&~owner.features.exit[i]&(price[i]>fast[i])
      recovery_health=np.where(healthy,recovery_health+1,0)
      old_exit=owner.exit_pending.copy();old_cap=owner.risk.cap;d=owner.decide(o);np.testing.assert_allclose(d.weights,r.targets.iloc[i].to_numpy(),rtol=0,atol=1e-13)
      eligible=owner.ready[i]&owner.features.entry[i]&~owner.features.exit[i]&(owner.features.score[i]>0)
      delayed=eligible&owner.readmit&(recovery_health>=owner.config.recovery)&flat
      bought=np.flatnonzero(d.unit_targets>u+1e-10)
      protected=np.any(d.unit_targets<u-1e-10)
      for j in np.flatnonzero(delayed):
       actionable=(not protected and owner.risk.cap>0 and ((u>1e-10).sum()<2))
       row=dict(scope=scope,date=str(date.date()),symbol=m.symbols[j],cash_fraction=cash/nav,cap=owner.risk.cap,healthy_closes=int(recovery_health[j]),entry_closes=int(owner.healthy[j]),actionable=bool(actionable),higher_than_new_request=bool(any(flat[k] and owner.features.score[i,j]>owner.features.score[i,k] for k in bought)),actual_new_requests=','.join(m.symbols[k] for k in bought if flat[k]))
       if i+20<len(price):row['matured_forward20_return']=price[i+20,j]/price[i,j]-1
       reentry_witness.append(row)

      newly=owner.exit_pending&~old_exit&(u>1e-10);stop_only=newly&~owner.features.exit[i]&owner.ready[i]&(price[i]<=owner.stop)
      for j in np.flatnonzero(newly):
       w=dict(date=str(date.date()),symbol=m.symbols[j],stock_feature_exit=bool(owner.features.exit[i,j]),stop_only=bool(stop_only[j]),mfe_r=float(mfe[j]),weight=value[j]/nav,observed_return=price[i,j]/entered[j]-1,cap_cut=owner.risk.cap<old_cap)
       if i+20<len(price):w['matured_forward20_return']=price[i+20,j]/price[i,j]-1
       witness.append(w)
      ready=owner.ready[i]&owner.features.entry[i]&~owner.features.exit[i]&(owner.features.score[i]>0)
      rows.append(dict(date=str(date.date()),exposure=value.sum()/nav,cash_fraction=cash/nav,warning_cap=owner.risk.cap,held=int((u>1e-10).sum()),eligible=int(ready.sum()),mature_held=int(((mfe>=2)&(u>1e-10)).sum()),stop_only=int(stop_only.sum()),feature_exit=int((newly&owner.features.exit[i]).sum()),pending=bool(owner.reduction_ceiling.min()<np.inf),reason=d.reason))
     table=pd.DataFrame(rows);events=pd.DataFrame(witness);table.to_csv(out/(scope+'_decisions.csv'),index=False,float_format='%.17g');events.to_csv(out/(scope+'_exit_events.csv'),index=False,float_format='%.17g')
     mainfolder=root/'selection/runs'/('incumbent_'+scope);main=load_result(mainfolder,expected=json.loads((mainfolder/'identity.json').read_text()));md,me,ma=attribute(m,main)
     by=pd.concat([daily.groupby('symbol').pnl.sum().rename('support_pnl'),md.groupby('symbol').pnl.sum().rename('main_pnl')],axis=1).fillna(0);by['difference']=by.support_pnl-by.main_pnl
     facts=dict(att,average_exposure=float(table.exposure.mean()),mean_exposure_cap_one=float(table[table.warning_cap==1].exposure.mean()),eligible_cash_cap_one_dates=int(((table.cash_fraction>.25)&(table.warning_cap==1)&(table.eligible>0)).sum()),new_exit_events=len(events),stop_only_events=int(events.stop_only.sum()),mature_stop_only_events=int((events.stop_only&(events.mfe_r>=2)).sum()),mature_support_exits_positive_next20=int((events.stop_only&(events.mfe_r>=2)&(events.matured_forward20_return>0)).sum()))
     summary[scope]=facts;by.to_csv(out/(scope+'_symbol_pnl.csv'),float_format='%.17g')


    w=pd.DataFrame(reentry_witness);w.to_csv(out/'reentry_health_witnesses.csv',index=False,float_format='%.17g')
    for name in summary:
     a=w[w.scope==name];z=a[a.actionable]
     summary[name]['readmission_health']={'symbol_close_observations':len(a),'structural_opportunities':len(z),'distinct_names':int(z.symbol.nunique()),'higher_than_new_request':int(z.higher_than_new_request.sum())}
    record={'status':'DIAGNOSTIC_NOT_ECONOMIC_ACCEPTANCE','source':source_identity(),'generator_sha256':digest(Path(__file__)),
     'parent_archive_sha256':ARCHIVE,'scopes':summary,
     'method':'Original-source observed filled inventories and exact close target reconstruction. No alternate equity curve or new strategy execution.',
     'qualification':'Structural opportunities have a vacancy, positive cap and no protective cut. Cash/risk/sector funding is NOT established. Counts are overlapping symbol-close observations; matured future returns are diagnostic only, never policy inputs.'}
    (out/'summary.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','data','supplement','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    reconstruct(args.parent,args.data,args.supplement,args.output)

"""Read-only actual-fill reconstruction; not an alternative executed portfolio."""
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import urllib.request
from types import SimpleNamespace
import numpy as np
import pandas as pd
from techquant.data import load_market,file_hash
from techquant.evidence import verify_evidence,source_identity
from techquant.policy import CloseObservation
from research.funded_risk import Owner,Parameters

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--original-policy',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--supplement',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    audit(p.parse_args())

def audit(a):
    a.output.mkdir(parents=True,exist_ok=True)
    catalog=json.loads(Path('research/catalog.json').read_text())
    market=load_market(a.data,supplement=a.supplement,sectors=catalog['sectors'])
    assert market.fingerprint()=='d9ded1d933f292933aa09145bc36a5696421c43dbf266fce018a34ef0b9f3f9b'
    spec=importlib.util.spec_from_file_location('_archived_funded_policy',a.original_policy)
    oldmodule=importlib.util.module_from_spec(spec);sys.modules[spec.name]=oldmodule;spec.loader.exec_module(oldmodule)
    summaries=[]
    for folder in sorted(a.archive.glob('*/runs/*')):
        if not (folder/'identity.json').exists():continue
        identity=json.loads((folder/'identity.json').read_text())
        if identity.get('policy',{}).get('name')!='funded_stop_risk':continue
        verify_evidence(folder)
        assert source_identity()['package_sha256']==identity['source']['package_sha256']
        assert file_hash(a.original_policy)==identity['policy']['implementation_sha256']
        m=market.prefix(identity['end']).subset(identity['universe'])
        assert m.fingerprint()==identity['data_sha256']
        params=identity['policy']['parameters']
        old=oldmodule.Owner(m,oldmodule.Parameters(**params));new=Owner(m,Parameters(**params))
        equity=pd.read_csv(folder/'equity.csv',index_col=0,parse_dates=True,float_precision='round_trip')
        targets=pd.read_csv(folder/'targets.csv',index_col=0,parse_dates=True,float_precision='round_trip')
        orders=pd.read_csv(folder/'orders.csv',float_precision='round_trip').to_dict('records')
        grouped=defaultdict(list)
        for o in orders:
            if o['status']=='FILLED':grouped[o['date']].append(o)
        units=np.zeros(len(m.symbols));index={s:j for j,s in enumerate(m.symbols)};rows=[]
        for i,date in enumerate(m.calendar):
            day=str(date.date());opening=units.copy();sales=np.zeros(len(units))
            for order in grouped[day]:
                j,q=index[order['symbol']],float(order['units']);assert order['signal_date']<day
                if order['side']=='BUY':units[j]+=q
                else:
                    sales[j]+=q;assert sales[j]<=opening[j]+1e-7
                    units[j]=max(0.,units[j]-q)
            nav,cash=float(equity.loc[date,'nav']),float(equity.loc[date,'cash'])
            price=np.nan_to_num(old.features.close[i],nan=0.);values=units*price
            assert abs(cash+values.sum()-nav)<=max(1e-6,nav*1e-10)
            observation=CloseObservation.from_inventory(i,day,nav,cash,units,values/nav)
            before,after=old.decide(observation),new.decide(observation)
            assert before.reason==equity.loc[date,'reason']
            error=float(np.max(np.abs(before.weights-targets.loc[date].to_numpy())))
            assert error<=1e-13,(str(folder),day,error)
            # New commitments have an admission stop, not an uninitialized live stop.
            stops=np.where(units>1e-10,old.stop,np.maximum(old.admission_stop(i),old.pending_stop))
            distance=np.maximum(price-stops,.02*price)
            rows.append(dict(date=day,nav=nav,cash=cash,global_peak=old.risk.global_peak,
                warning_cap=old.risk.warnings.cap,funded_budget=old.risk.budget,
                actual_risk=float(units@distance),requested_risk=float(before.unit_targets@distance),
                original_target_error=error,changed_units=not np.array_equal(before.unit_targets,after.unit_targets),
                target_l1_change=float(np.abs(before.weights-after.weights).sum()),
                protective=bool(np.any(before.unit_targets<units-1e-10)),reason=before.reason))
        frame=pd.DataFrame(rows)
        filename=folder.parent.parent.name+'_'+folder.name+'.csv'
        frame.to_csv(a.output/filename,index=False,float_format='%.17g')
        breach=frame.requested_risk>frame.funded_budget+1e-7
        summary=dict(run=filename,source_commit=identity['source']['commit'],parameters=params,
            sessions=len(rows),changed_close_decisions=int(frame.changed_units.sum()),
            requested_risk_budget_breaches=int(breach.sum()),first_breach=frame.loc[breach,'date'].min() if breach.any() else None,
            max_archived_target_absolute_error=float(frame.original_target_error.max()),
            audit_sha256=file_hash(a.output/filename))
        summaries.append(summary);print(json.dumps(summary),flush=True)
    if len(summaries)!=9 or sum(s['sessions'] for s in summaries)!=7050:
        raise ValueError('pinned projection audit requires all nine original runs')
    report=dict(status='ACTUAL_FILL_DIAGNOSTIC_NOT_ECONOMIC_ACCEPTANCE',data_sha256=market.fingerprint(),
        original_policy_sha256=file_hash(a.original_policy),corrected_policy_sha256=file_hash(Path('research/funded_risk.py')),
        method='Reconstruct all archived actual fills, verify T+1, NAV and every original close target/reason. Feed same observations to corrected policy; changed decisions require fresh execution, not relabeling this trace as a portfolio return.',
        qualification='The corrected owner does not execute its alternative requests in this diagnostic. A requested reduction is never counted as an actual fill or spendable cash.',runs=summaries)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
def hosted(output, data, supplement):
    """Recover only the pinned measured source; never fit or rerun its grid."""
    root=output.parent/'funded-projection-origin'
    root.mkdir(parents=True,exist_ok=True)
    archive=root/'evidence.tar.gz'
    origin='e8c627e7529af833d55f516ef4d97ee309f2611e/34834159593'
    expected='3b8b967added9a8257f64e49d1eadc4031309965404261ee621614618592371b'
    url='https://raw.githubusercontent.com/ychenracing/quant/research/evidence/nonlinear/'+origin+'/nonlinear-evidence.tar.gz'
    with urllib.request.urlopen(url,timeout=60) as response:
        archive.write_bytes(response.read())
    if file_hash(archive)!=expected:
        raise ValueError('pinned funded evidence archive mismatch')
    with tarfile.open(archive) as bundle:
        bundle.extractall(root,filter='data')
    measured=root/'nonlinear'
    for name,digest in json.loads((measured/'MANIFEST.json').read_text()).items():
        if file_hash(measured/name)!=digest:
            raise ValueError('funded audit origin manifest mismatch: '+name)
    if (measured/'source-commit.txt').read_text().strip()!=origin.split('/')[0]:
        raise ValueError('funded audit origin source mismatch')
    with tarfile.open(measured/'source.tar.gz') as bundle:
        bundle.extractall(root/'source',filter='data')
    audit(SimpleNamespace(archive=measured,original_policy=root/'source/research/funded_risk.py',
                          data=data,supplement=supplement,output=output/'projection_audit'))
    (output/'audit_identity.json').write_text(json.dumps(dict(
        status='DIAGNOSTIC_NOT_ECONOMIC_ACCEPTANCE',source=source_identity(),
        original_archive_sha256=expected,original_origin=origin,
        new_model_fits=0,new_economic_grid_runs=0),indent=2)+'\n')

if __name__=='__main__':main()

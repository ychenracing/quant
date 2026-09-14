"""Measure the original ten neighborhoods, without selecting a new policy."""
from __future__ import annotations
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import tarfile
import urllib.request
import numpy as np
import pandas as pd
from techquant.config import Config
from techquant.data import file_hash, load_market
from techquant.evidence import load_result, metrics
from techquant.policy import CloseObservation
from research.finite_study import Study
from research.fixed_validation import write_json, frozen_plan, floor_audit
from research.ledger_attribution import attribute
from research.observed_admission_completion import Owner, Parameters

ROOT=Path(__file__).parent
FAMILY='observed_admission_completion'
DEFAULT_SOURCE='e7341fc52e4a11cc4d8396aa3f15819f50702877'


def neighborhood_cases(plan, market):
    result=[];seen=set()
    for original in plan:
        if original['group']!='stability_not_reselection':continue
        case=dict(original);names=case['symbols']
        if (case['name'] in seen or not names or len(names)!=len(set(names))
                or set(names)-set(market.symbols) or case['cost']!=1. or case['delay']!=1):
            raise ValueError('invalid or duplicated original neighborhood')
        cfg=Config(**case['config'])
        if asdict(cfg)!=case['config']:raise ValueError('configuration was changed while loading')
        seen.add(case['name']);result.append(case)
    return result


def saved(market,path,config,*,candidate=True,costs=1.,delay=1):
    return Study(FAMILY).saved(market,Path(path),Parameters() if candidate else None,
        costs=costs,delay=delay,configuration=config)


def default_equivalence(origin, market, destination):
    """Replay only policy decisions against old actual inventory, not portfolios."""
    origin=Path(origin)
    if (origin/'source-commit.txt').read_text().strip()!=DEFAULT_SOURCE:
        raise ValueError('unexpected measured default source')
    manifest=json.loads((origin/'MANIFEST.json').read_text())
    for name,digest in manifest.items():
        path=origin/name
        if (Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink()
                or not path.is_file() or file_hash(path)!=digest):
            raise ValueError('measured default manifest mismatch: '+name)
    catalog=json.loads((ROOT/'catalog.json').read_text())
    scopes={'union':list(market.symbols),'chatgpt_5':catalog['pools']['chatgpt_5'],
            'joint_optical_leader_removal':[s for s in market.symbols if s not in
                                            ('sz300308','sz300502','sz300394')]}
    candidates={}
    for path in sorted((origin/'runs').glob('*_candidate')):
        identity=json.loads((path/'identity.json').read_text())
        if identity['config']==asdict(Config()) and identity['cost_multiplier']==1. and identity['delay']==1:
            candidates[tuple(identity['universe'])]=(path,identity)
    reports=[]
    for name,names in scopes.items():
        sub=market.subset(names);path,identity=candidates[tuple(sub.symbols)]
        if identity['source']['commit']!=DEFAULT_SOURCE or identity['data_sha256']!=sub.fingerprint():
            raise ValueError('default comparison identity mismatch')
        result=load_result(path,expected=identity)
        ledger=attribute(sub,result)[2]
        plain=Owner(sub,Parameters());configured=Owner(sub,Parameters(),config=Config())
        units=np.zeros(len(sub.symbols));fills={};max_error=0.
        for order in result.orders:
            if order['status']=='FILLED':fills.setdefault(order['date'],[]).append(order)
        for i,date in enumerate(sub.calendar):
            day=str(date.date())
            for order in fills.get(day,[]):
                j=sub.symbols.index(order['symbol'])
                units[j]+=(1 if order['side']=='BUY' else -1)*float(order['units'])
                if abs(units[j])<1e-8:units[j]=0.
            row=result.equity.loc[date];price=np.nan_to_num(plain.inner.features.close[i],nan=0.)
            obs=CloseObservation.from_inventory(i,day,float(row.nav),float(row.cash),units,units*price/row.nav)
            a=plain.decide(obs);b=configured.decide(obs)
            np.testing.assert_array_equal(a.unit_targets,b.unit_targets)
            np.testing.assert_array_equal(a.weights,b.weights)
            if a.reason!=b.reason or a.cap!=b.cap:raise ValueError('explicit default changes a decision')
            error=float(np.max(np.abs(a.weights-result.targets.loc[date].to_numpy())))
            max_error=max(max_error,error)
            if error>1e-13 or a.reason!=row.reason or a.cap!=row.target_cap:
                raise ValueError('default differs from the old actual-inventory path: '+name+' '+day)
        if plain.trace!=configured.trace:raise ValueError('default trace differs')
        reports.append({'scope':name,'closes':len(sub.calendar),'maximum_weight_error':max_error,
            'actual_ledger':ledger,'old_identity_sha256':file_hash(path/'identity.json'),
            'new_policy':configured.identity(),'status':'RECORDED_PATH_EQUIVALENCE'})
    write_json(destination,{'old_source':DEFAULT_SOURCE,'study':Study(FAMILY).identity(),
        'new_portfolio_runs':0,'reports':reports,'economic_acceptance':'UNVERIFIED',
        'limitation':'These default recorded paths only; not a universal equivalence or acceptance claim.'})
    return reports


def evaluate(market, plan, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    neighbors=neighborhood_cases(plan,market)
    if len(neighbors)!=10:raise ValueError('the original ten neighborhoods are required')
    cases=[{'name':'default','group':'default_control','symbols':list(market.symbols),
            'config':asdict(Config()),'cost':1.,'delay':1}]+neighbors
    contract=ROOT/'neighborhood_validation_contract.json'
    windows=json.loads((ROOT/'protocol.json').read_text())['time_windows']
    identity={'study':Study(FAMILY).identity(),'runner_sha256':file_hash(Path(__file__)),
        'contract_sha256':file_hash(contract),'protocol_sha256':file_hash(ROOT/'protocol.json'),
        'data_sha256':market.fingerprint(),'cases':cases,'windows':windows,'reselection':False}
    if (output/'plan.json').exists() and json.loads((output/'plan.json').read_text())!=identity:
        raise ValueError('neighborhood resume identity mismatch')
    write_json(output/'plan.json',identity)
    results={};rows=[];audits=[];prefixes=[]
    for case in cases:
        sub=market.subset(case['symbols']);cfg=Config(**case['config'])
        for policy in ('candidate','incumbent'):
            path=output/'runs'/(case['name']+'_'+policy)
            result=saved(sub,path,cfg,candidate=policy=='candidate')
            results[(case['name'],policy)]=result
            ledger=attribute(sub,result)[2];floor=floor_audit(sub,result)
            if floor['subfloor_fills']:raise ValueError('new-source ordinary fill violates original floor')
            audits.append({'case':case['name'],'policy':policy,'ledger':ledger,'floor':floor})
            for window,(start,end) in windows.items():
                rows.append({'case':case['name'],'policy':policy,'window':window,**metrics(result,start,end)})
        # Every actual setting receives a causal prefix, rather than extrapolating a default check.
        cut=sub.calendar[500];short=saved(sub.prefix(cut),output/'prefix/runs'/case['name'],cfg)
        full=results[(case['name'],'candidate')]
        pd.testing.assert_frame_equal(full.equity.loc[:cut],short.equity,check_exact=True,check_freq=False)
        pd.testing.assert_frame_equal(full.targets.loc[:cut],short.targets,check_exact=True,check_freq=False)
        if [o for o in full.orders if o['date']<=str(cut.date())]!=short.orders:
            raise ValueError('changed-configuration prefix orders differ')
        full_trace=json.loads((output/'intents'/(case['name']+'_candidate.json')).read_text())['trace']
        short_trace=json.loads((output/'prefix/intents'/(case['name']+'.json')).read_text())['trace']
        if [r for r in full_trace if r['date']<=str(cut.date())]!=short_trace:
            raise ValueError('changed-configuration intent prefix differs')
        prefixes.append({'case':case['name'],'cut':str(cut.date()),'status':'EXACT_NAV_TARGET_ORDER_INTENT'})
        write_json(output/'progress.json',{'completed_cases':len(prefixes),'economic_acceptance':'UNVERIFIED'})
    matrix=pd.DataFrame(rows);matrix.to_csv(output/'matrix.csv',index=False,float_format='%.17g')
    comparisons=[];reference=results[('default','candidate')];default=metrics(reference)
    for case in neighbors:
        candidate=results[(case['name'],'candidate')];value=metrics(candidate)
        same=(candidate.equity.equals(reference.equity) and candidate.targets.equals(reference.targets)
              and candidate.orders==reference.orders)
        comparisons.append({'case':case['name'],'configuration':case['config'],
            'wealth':value['wealth'],'max_drawdown':value['max_drawdown'],'orders':value['orders'],
            'wealth_to_default':value['wealth']/default['wealth'],
            'drawdown_change':value['max_drawdown']-default['max_drawdown'],
            'identical_full_recorded_path':same,
            'interpretation':'INERT_ON_MEASURED_PATH_NOT_A_USEFUL_SENSITIVITY_KNOB' if same else 'MEASURED_SENSITIVITY_NOT_RESELECTION'})
    summary={'operation':'original_parameter_neighborhoods','status':'MEASURED_NOT_ACCEPTANCE',
        'scenario_count':len(cases),'full_accounts':len(results),'prefix_accounts':len(prefixes),
        'default':default,'neighbors':comparisons,'economic_acceptance':'UNVERIFIED',
        'reselection':False,'limitation':'Original joint economics and native-reference obligations remain. 2026 is retrospective.'}
    write_json(output/'account_audits.json',audits);write_json(output/'prefix_checks.json',prefixes)
    write_json(output/'summary.json',summary)
    return summary


def hosted(output):
    from research.hosted_study import inputs
    output=Path(output);checkout=ROOT.parent
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=checkout,text=True).strip() or os.environ.get('QUANT_SOURCE_COMMIT')!=sha:
        raise ValueError('neighborhood validation requires clean source binding')
    contract=json.loads((ROOT/'neighborhood_validation_contract.json').read_text())
    cache=output.parent/'neighborhood-input';data,supplement=inputs(cache)
    plan,content=frozen_plan(cache/'input.tar.gz')
    if file_hash(cache/'input.tar.gz')!=contract['baseline_archive_sha256']:
        raise ValueError('baseline archive differs')
    catalog=json.loads((ROOT/'catalog.json').read_text())
    market=load_market(data,supplement=supplement,sectors=catalog['sectors'])
    if market.fingerprint()!=contract['data_sha256']:raise ValueError('frozen market differs')
    output.mkdir(parents=True,exist_ok=True);(output/'original-case-plan.json').write_bytes(content)
    # Recover already completed original coverage; do not execute its 555 accounts again.
    archive=cache/'measured-default.tar.gz'
    prefix=('https://raw.githubusercontent.com/ychenracing/quant/research/evidence/fixed-validation/'+
            contract['measured_default_source']+'/'+contract['measured_default_run']+'/')
    with archive.open('wb') as dst:
        for part in ('000','001'):
            with urllib.request.urlopen(prefix+'fixed-validation.tar.gz.part-'+part,timeout=60) as src:
                while block:=src.read(1024*1024):dst.write(block)
    if file_hash(archive)!=contract['measured_archive_sha256']:raise ValueError('measured default archive differs')
    with tarfile.open(archive) as src:src.extractall(cache/'origin',filter='data')
    default_equivalence(cache/'origin/fixed-validation',market,output/'default_equivalence.json')
    return evaluate(market,plan,output)

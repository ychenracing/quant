"""Validate one frozen owner against original scopes; never select a new policy.

Old accounts retain their identities. Corrected-source reuse requires both a
source/runtime applicability check and a fill-by-fill no-trigger proof.
"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
import os
import subprocess
from pathlib import Path
import shutil
import tarfile
import urllib.request
import numpy as np
import pandas as pd
from techquant.config import Config
from techquant.data import Market, file_hash, load_market
from techquant.engine import Result
from techquant.evidence import load_result, metrics, source_identity
from research.finite_study import Study
from research.ledger_attribution import attribute

ROOT = Path(__file__).parent
FAMILY = 'observed_admission_completion'


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary=Path(path).with_suffix(Path(path).suffix+'.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def digest_value(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def case_key(case):
    return digest_value({'symbols':sorted(case['symbols']),'cost':case['cost'],
                         'delay':case['delay'],'config':case['config']})


def resolve_cases(plan, market):
    """Retain scenario rows and explicitly identify unsupported neighborhoods."""
    cases, missing, seen = [], [], set()
    for original in plan:
        case = dict(original)
        if case['name'] in seen:
            raise ValueError('duplicate scenario name')
        seen.add(case['name'])
        names = case['symbols']
        if (not names or len(names)!=len(set(names)) or set(names)-set(market.symbols)
                or type(case['cost']) not in (int,float) or not np.isfinite(case['cost'])
                or case['cost']<=0 or type(case['delay']) is not int or case['delay']<1):
            raise ValueError('invalid fixed scenario membership or execution assumption')
        if case['group']=='stability_not_reselection':
            missing.append({'case':case['name'],'status':'UNVERIFIED',
                'reason':'Frozen owner constructor does not accept these configuration changes; no stability claim or waiver.'})
            continue
        if case['config'] != asdict(Config()):
            raise ValueError('fixed candidate configuration may not change')
        case['symbols']=sorted(names)
        cases.append(case)
    return cases, missing


def floor_audit(market: Market, result: Result):
    """Reconstruct the exact opening account and audit the added execution guard."""
    if (list(market.symbols)!=result.metadata['universe']
            or market.fingerprint()!=result.metadata['data_sha256']):
        raise ValueError('floor audit market identity mismatch')
    opens=market.panel('open');close=market.panel('close').ffill();previous=close.shift()
    units=np.zeros(len(market.symbols));cash=result.metadata['config']['initial_cash']
    by_day={};positions={};ratios=[];violations=[];protective_count=0;worst=0.
    for order in result.orders:
        if order['status']=='FILLED':by_day.setdefault(order['date'],[]).append(order)
    for date,row in result.equity.iterrows():
        prices=opens.loc[date].to_numpy()
        marks=np.where(np.isfinite(prices),prices,previous.loc[date].to_numpy())
        opening_nav=cash+float(np.nan_to_num(units*marks,nan=0.).sum())
        for order in by_day.get(str(date.date()),[]):
            signal=pd.Timestamp(order['signal_date'])
            if signal>=date or signal not in positions:
                raise ValueError('fill precedes an executable signal')
            j=market.symbols.index(order['symbol']);pending=result.targets.loc[signal,order['symbol']]
            protective=(order['side']=='SELL' and units[j]>1e-10
                        and (pending==0 or pending<positions[signal][j]-1e-10))
            amount=order['units']*prices[j]
            if protective:protective_count+=1
            else:
                ratios.append(float(amount/opening_nav))
                if amount<.01*opening_nav:
                    violations.append(dict(order,opening_nav=opening_nav,
                        opening_notional=float(amount),floor=.01*opening_nav))
            direction=1 if order['side']=='BUY' else -1
            units[j]+=direction*order['units'];cash-=direction*order['notional']+order['fee']
            if abs(units[j])<1e-8:units[j]=0.
        holdings=np.nan_to_num(units*close.loc[date].to_numpy(),nan=0.)
        error=max(abs(cash+float(holdings.sum())-row.nav),abs(cash-row.cash))
        worst=max(worst,error)
        if error>max(1e-6,row.nav*1e-12):raise ValueError('floor audit does not reconcile to actual NAV/cash')
        positions[date]=holdings/row.nav
    return {'fills':sum(map(len,by_day.values())),'ordinary_fills':len(ratios),
            'protective_fills':protective_count,'subfloor_fills':len(violations),
            'minimum_ordinary_fraction':min(ratios) if ratios else None,
            'maximum_ledger_error':worst,'violations':violations}


def account_key(market, policy, costs=1., delay=1):
    return digest_value({'data':market.fingerprint(),'policy':policy,'cost':float(costs),'delay':delay})


def origin_accounts(origin: Path, market, output):
    """Audit immutable origin; return reusable accounts only inside the proven scope."""
    contract=json.loads((ROOT/'fixed_validation_contract.json').read_text())
    if (origin/'source-commit.txt').read_text().strip()!=contract['origin_source']:
        raise ValueError('wrong original source')
    manifest=json.loads((origin/'MANIFEST.json').read_text())
    for name,digest in manifest.items():
        path=origin/name
        if (Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink()
                or not path.is_file() or file_hash(path)!=digest):
            raise ValueError('original evidence member mismatch: '+name)
    selected=json.loads((origin/'selection/selection.json').read_text())
    if (selected['candidate']!=0 or selected['parameters']!={}
            or selected['identity']['family']!=FAMILY):
        raise ValueError('original selection is not the frozen singleton')
    current=Study(FAMILY).identity();old=selected['identity'];reasons=[]
    if current['dependencies']!=old['dependencies']:reasons.append('research dependency identity changed')
    expected=old['source']['files']|contract['corrected_source_files']
    if current['source']['files']!=expected:reasons.append('production changes exceed reviewed corrections')
    for key in ('python','platform','dependencies'):
        if current['source'][key]!=old['source'][key]:reasons.append('runtime differs: '+key)
    audited=[];candidates={}
    for directory in sorted(origin.glob('*/runs/*')):
        if not directory.is_dir():continue
        identity=json.loads((directory/'identity.json').read_text())
        if identity['source']!=old['source'] or identity['study']!=old:
            raise ValueError('mixed original source/study identity')
        subset=market.prefix(identity['end']).subset(identity['universe'])
        if (identity['config']!=asdict(Config()) or identity['data_sha256']!=subset.fingerprint()
                or identity['delay']!=1 or identity['cost_multiplier']!=1.):
            raise ValueError('original account conditions differ')
        result=load_result(directory,expected=identity)
        checked=floor_audit(subset,result)
        record={'account':str(directory.relative_to(origin)),**checked}
        audited.append(record)
        if checked['subfloor_fills']:reasons.append('guard changes '+record['account'])
        if directory.parent.parent.name=='evaluation':
            policy='candidate' if 'policy' in identity else identity['benchmark'] or 'incumbent'
            if policy=='candidate':
                owner=Study(FAMILY).module.Owner(subset,Study(FAMILY).module.Parameters())
                if identity['policy']!=owner.identity():reasons.append('owner identity changed')
            key=account_key(subset,policy)
            candidates[key]=(result,{'kind':'RECORDED_PATH_EQUIVALENCE',
                'source_commit':identity['source']['commit'],'origin_account':record['account'],
                'origin_run':contract['origin_run'],'identity_sha256':file_hash(directory/'identity.json')})
    if len(audited)!=18 or len(candidates)!=9:raise ValueError('original account coverage is incomplete')
    report={'old_source':contract['origin_source'],'current_source':current['source'],
            'old_manifest_sha256':file_hash(origin/'MANIFEST.json'),'accounts':audited,
            'reuse_permitted':not reasons,'reasons':sorted(set(reasons)),
            'limitation':'Applicability only for these recorded paths; not new-HEAD economic acceptance.'}
    write_json(output/'origin_applicability.json',report)
    return {} if reasons else candidates


def validate_cases(market, plan, output: Path, *, origin=None, include_contributor_removal=False):
    cases,missing=resolve_cases(plan,market);origin=origin or {};study=Study(FAMILY)
    if [asdict(p) for p in study.module.grid()]!=[{}]:raise ValueError('fixed singleton grid changed')
    windows=json.loads((ROOT/'protocol.json').read_text())['time_windows']
    identity={'study':study.identity(),'data_sha256':market.fingerprint(),'cases':list(cases),
              'runner_sha256':file_hash(Path(__file__)),
              'contract_sha256':file_hash(ROOT/'fixed_validation_contract.json'),
              'protocol_sha256':file_hash(ROOT/'protocol.json'),'windows':windows,
              'original_accounts':{k:v[1] for k,v in sorted(origin.items())},
              'include_contributor_removal':include_contributor_removal,'unverified':missing}
    output.mkdir(parents=True,exist_ok=True)
    if (output/'plan.json').exists() and json.loads((output/'plan.json').read_text())!=identity:
        raise ValueError('fixed validation resume identity mismatch')
    write_json(output/'plan.json',identity)
    results={};rows=[];new=0;reused=0;added=False
    for case in cases:
        subset=market.subset(case['symbols']);context=case_key(case)
        policies=['candidate','incumbent','buy_hold']
        if case['group']=='original_pool':policies.append('equal_weight')
        for policy in policies:
            key=account_key(subset,policy,case['cost'],case['delay'])
            if key not in results:
                if key in origin:
                    results[key]=origin[key];reused+=1
                else:
                    path=output/'runs'/(context+'_'+policy);new+=int(not path.exists())
                    result=study.saved(subset,path,
                        study.module.Parameters() if policy=='candidate' else None,
                        benchmark=policy if policy in ('buy_hold','equal_weight') else None,
                        costs=case['cost'],delay=case['delay'])
                    results[key]=(result,{'kind':'SOURCE_BOUND_RUN','path':str(path.relative_to(output)),
                        'source_commit':result.metadata['source']['commit']})
            result,record=results[key]
            for window,(start,end) in windows.items():
                rows.append({'case':case['name'],'group':case['group'],'policy':policy,
                    'window':window,'universe_size':len(subset.symbols),'account_key':key,
                    'evidence_kind':record['kind'],'source_commit':record['source_commit'],
                    **metrics(result,start,end)})
            if include_contributor_removal and case['name']=='union' and policy=='candidate' and not added:
                days,episodes,reconciliation=attribute(subset,result)
                pnl={s:0. for s in subset.symbols}
                if not episodes.empty:pnl.update(episodes.groupby('symbol').pnl.sum().to_dict())
                ranked=sorted(pnl,key=lambda s:(-pnl[s],s));chosen=ranked[:3]
                write_json(output/'actual_contributors.json',{'pnl':pnl,'ranked_symbols':ranked,
                    'removed':chosen,'ledger':reconciliation,'status':'EX_POST_DIAGNOSTIC_NOT_SELECTION'})
                kept=[s for s in market.symbols if s not in chosen]
                if not kept:raise ValueError('actual contributor removal leaves no universe')
                cases.append(dict(name='remove_realized_top3',group='ex_post_contributor_removal',
                    symbols=kept,cost=1.,delay=1,config=asdict(Config())))
                added=True
        pd.DataFrame(rows).to_csv(output/'matrix.csv',index=False,float_format='%.17g')
        write_json(output/'progress.json',{'last_case':case['name'],'rows':len(rows),
            'unique_accounts':len(results),'new_accounts':new,'economic_acceptance':'UNVERIFIED'})
        print(case['name'],len(results),'unique accounts',flush=True)
    if include_contributor_removal and not added:raise ValueError('required union contributor attribution is absent')
    write_json(output/'resolved_cases.json',cases)
    write_json(output/'account_registry.json',{k:v[1] for k,v in results.items()})
    frame=pd.DataFrame(rows);full=frame[frame.window=='full']
    compared=full.pivot(index='case',columns='policy',values=['wealth','max_drawdown','orders'])
    wealth=compared['wealth']['candidate']/compared['wealth']['incumbent']
    hold=compared['wealth']['candidate']/compared['wealth']['buy_hold']
    report={'status':'FINITE_FIXED_CANDIDATE_DIAGNOSTIC','scenario_count':len(cases),
        'unique_accounts':len(results),'new_accounts':new,'reused_origin_accounts':reused,
        'rows':len(frame),'economic_acceptance':'UNVERIFIED','unverified_cases':missing,
        'outstanding_requirements':json.loads((ROOT/'fixed_validation_contract.json').read_text())['outstanding_requirements'],
        'wealth_below_incumbent_cases':int((wealth<1).sum()),
        'minimum_wealth_to_incumbent':float(wealth.min()),'worst_incumbent_case':str(wealth.idxmin()),
        'minimum_wealth_to_buy_hold':float(hold.min()),'worst_buy_hold_case':str(hold.idxmin()),
        'all_subset_superiority':'NOT_ESTABLISHED','native_reference_comparison':'UNVERIFIED',
        'note':'Scenario counts include aliases; accounts are deduplicated. Comparison counts are diagnostics, not replacement acceptance rules.'}
    write_json(output/'summary.json',report)
    return report


def frozen_plan(archive: Path):
    contract=json.loads((ROOT/'fixed_validation_contract.json').read_text())
    if file_hash(archive)!=contract['baseline_archive_sha256']:raise ValueError('frozen baseline archive mismatch')
    with tarfile.open(archive) as source:
        content=source.extractfile(contract['case_plan_member']).read()
    if hashlib.sha256(content).hexdigest()!=contract['case_plan_sha256']:raise ValueError('original case plan mismatch')
    plan=json.loads(content)
    if len(plan)!=209:raise ValueError('original plan coverage changed')
    return plan,content


def prefix_checks(market, catalog, output, origin):
    contract=json.loads((ROOT/'fixed_validation_contract.json').read_text());study=Study(FAMILY)
    scopes={'union':list(market.symbols),'chatgpt_5':catalog['pools']['chatgpt_5'],
        'joint_optical_leader_removal':[s for s in market.symbols if s not in {'sz300308','sz300502','sz300394'}]}
    reports=[]
    for name,names in scopes.items():
        subset=market.subset(names);key=account_key(subset,'candidate')
        if key in origin:full=origin[key][0]
        else:
            case=dict(symbols=names,cost=1.,delay=1,config=asdict(Config()))
            full=study.saved(subset,output/'runs'/(case_key(case)+'_candidate'),study.module.Parameters())
        for index in contract['prefix_session_indices']:
            cut=subset.calendar[index];short=subset.prefix(cut)
            prefix=study.saved(short,output/'prefix/runs'/(name+'_'+str(index)),study.module.Parameters())
            pd.testing.assert_frame_equal(full.equity.loc[:cut],prefix.equity,check_exact=True,check_freq=False)
            pd.testing.assert_frame_equal(full.targets.loc[:cut],prefix.targets,check_exact=True,check_freq=False)
            if [o for o in full.orders if o['date']<=str(cut.date())]!=prefix.orders:
                raise ValueError('prefix actual orders diverge')
            reports.append({'scope':name,'cut':str(cut.date()),'status':'PASS',
                            'actual_orders_and_nav':'EXACT','source':prefix.metadata['source']})
    write_json(output/'prefix_checks.json',reports)


def hosted(output: Path):
    """Called only by the explicit short validation workflow, never ordinary CI."""
    from research.hosted_study import inputs
    checkout=ROOT.parent
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain'],cwd=checkout,text=True)
    if dirty.strip() or os.environ.get('QUANT_SOURCE_COMMIT')!=commit:
        raise ValueError('hosted validation requires a clean, exactly bound checkout')
    contract=json.loads((ROOT/'fixed_validation_contract.json').read_text())
    cache=output.parent/'fixed-validation-input';data,supplement=inputs(cache)
    plan,content=frozen_plan(cache/'input.tar.gz')
    output.mkdir(parents=True,exist_ok=True);(output/'original-case-plan.json').write_bytes(content)
    catalog=json.loads((ROOT/'catalog.json').read_text())
    market=load_market(data,supplement=supplement,sectors=catalog['sectors'])
    if market.fingerprint()!=contract['data_sha256']:raise ValueError('frozen market differs')
    archive=cache/'origin.tar.gz'
    url=('https://raw.githubusercontent.com/ychenracing/quant/'+contract['origin_evidence_commit']+
         '/nonlinear/'+contract['origin_source']+'/'+contract['origin_run']+'/nonlinear-evidence.tar.gz')
    with urllib.request.urlopen(url,timeout=45) as response,archive.open('wb') as dst:
        shutil.copyfileobj(response,dst)
    if file_hash(archive)!=contract['origin_archive_sha256']:raise ValueError('original archive mismatch')
    origin=cache/'origin'
    with tarfile.open(archive) as source:source.extractall(origin,filter='data')
    reusable=origin_accounts(origin/'nonlinear',market,output)
    report=validate_cases(market,plan,output,origin=reusable,include_contributor_removal=True)
    prefix_checks(market,catalog,output,reusable)
    return report

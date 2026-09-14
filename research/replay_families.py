"""Portable reproduction of declared independent research families, never a CI gate.

Every candidate, rejected algorithm and subperiod remains in the evidence. A
ranking selects a diagnostic for evaluation; it never changes economic gates.
"""
from __future__ import annotations
from dataclasses import asdict
import argparse
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import numpy as np
import pandas as pd
from techquant.data import file_hash, load_market
from techquant.engine import Result, run as incumbent
from techquant.evidence import metrics, save_result, source_identity, verify_evidence
from research.reproduce import restore_archives

# These sets reproduce the original pre-measurement declarations, not a new search.
GRIDS = {
    'directional': dict(exit_drawdown=[.08,.12,.16,.20], reentry_rebound=[.04,.08], positions=[2,3], lookback=[120,240]),
    'breakout': dict(slow=[20,40,80], stop=[.08,.12,.16], positions=[2,3], entry_window=[10,20]),
    'shadow_guard': dict(span=[10,20,40], loss=[.06,.10,.14], rebound=[.03,.06]),
    'record_high': dict(stop=[.06,.08,.12,.16], positions=[2,3,4,34], minimum_observations=[10,40]),
    'rotation': dict(lookback=[40,80,160], trend=[20,40,80], positions=[2,3,5], review=[10,20]),
    'passive_guard': dict(span=[10,20,40,80], confirmation=[1,2], hysteresis=[0.,.02], floor=[0.,.5]),
    'downside_guard': dict(shock_z=[1.5,2.5], regime_span=[40,80,120], recovery=[2,4], bear_cap=[0.,.25]),
    'sleeves': dict(stop=[.06,.10,.14,.18], reentry_window=[20,40,80]),
    'funded_ownership': dict(entry_window=[40,80], initial_fraction=[.20,.30], trail_atr=[2.5,3.5,4.5], distribution_guard=[False,True]),
    'adaptive_expectation': dict(horizon=[10,20], shrinkage=[.1,1.,10.], positions=[2,4]),
}


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.pending')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def aggregate_sleeves(results):
    nav = sum(r.equity.nav for r in results); cash = sum(r.equity.cash for r in results)
    equity = results[0].equity.copy()
    equity['nav'], equity['cash'], equity['holdings'] = nav, cash, nav-cash
    equity['exposure'] = (nav-cash)/nav
    equity['breadth'] = sum(r.equity.breadth for r in results)/len(results)
    equity['reason'] = 'AGGREGATE_INDEPENDENT_CASH_SLEEVES_DIAGNOSTIC'
    targets = pd.concat([r.targets.mul(r.equity.nav/nav, axis=0) for r in results], axis=1)
    metadata = dict(results[0].metadata); metadata['config'] = dict(metadata['config'])
    metadata['config']['initial_cash'] = sum(r.metadata['config']['initial_cash'] for r in results)
    metadata['universe'] = list(targets.columns)
    metadata['classification'] = 'DIAGNOSTIC_PER_SLEEVE_MATERIALITY_NOT_ACCOUNT_ACCEPTANCE'
    metadata['component_identities'] = [r.metadata for r in results]
    return Result(equity, targets, [o for r in results for o in r.orders], metadata)


def measure(module, name, market, parameters, destination, references):
    if name == 'sleeves':
        parts = []
        for symbol in market.symbols:
            r = module.run_single(market.subset([symbol]), parameters, 2_000_000/len(market.symbols))
            save_result(r, destination.parent/(destination.name+'_components')/symbol); parts.append(r)
        result = aggregate_sleeves(parts)
    else:
        kwargs = {}
        if name == 'shadow_guard': kwargs['shadow'] = references['breakout']
        if name == 'passive_guard': kwargs['shadow'] = references['buy_hold']
        if name == 'downside_guard': kwargs = dict(alpha=references['rotation'], passive=references['buy_hold'])
        result = module.run(market, parameters, **kwargs)
    save_result(result, destination)
    return result


def study(market, catalog, output, families):
    scopes = {'union': list(market.symbols), 'chatgpt_5': catalog['pools']['chatgpt_5'],
              'joint_optical_leader_removal': [s for s in market.symbols if s not in {'sz300308','sz300502','sz300394'}]}
    contexts = {}; reference_metrics = {}
    from research.breakout import run as breakout, Parameters as Breakout
    from research.rotation import run as rotation, Parameters as Rotation
    for period, full in [('training', market.prefix('2025-12-31')), ('evaluation', market)]:
        for scope, names in scopes.items():
            m = full.subset(names)
            refs = {'buy_hold': incumbent(m, benchmark='buy_hold'), 'incumbent': incumbent(m)}
            if 'shadow_guard' in families:
                refs['breakout'] = breakout(m, Breakout(slow=20,stop=.12,positions=2,entry_window=10))
            if 'downside_guard' in families:
                refs['rotation'] = rotation(m, Rotation(lookback=40,trend=80,positions=2,review=20))
            for name, result in refs.items():
                save_result(result, output/'references'/period/scope/name)
            contexts[period,scope] = (m,refs)
            reference_metrics[period,scope] = {name: metrics(r) for name,r in refs.items()}
    summary = []; all_evaluation = []
    for name in families:
        module = importlib.import_module('research.'+name)
        grid = GRIDS[name]; candidates = [dict(zip(grid,values)) for values in itertools.product(*grid.values())]
        out = output/name; out.mkdir(); ranking = []; rows = []
        write(out/'identity.json', {'source': source_identity(), 'data_sha256': market.fingerprint(),
              'algorithm_sha256': file_hash(Path(module.__file__)), 'runner_sha256': file_hash(Path(__file__)),
              'candidates': candidates, 'selection_end': '2025-12-31', 'scopes': scopes,
              'status': 'RETROSPECTIVE_RESEARCH_NOT_ACCEPTED'})
        for values in candidates:
            key = '_'.join(map(str, values.values())); scores = []; deficits = []
            for scope in scopes:
                m, refs = contexts['training',scope]
                result = measure(module,name,m,module.Parameters(**values),out/'runs'/(key+'_'+scope),refs)
                mm = metrics(result); bh = reference_metrics['training',scope]['buy_hold']
                old = reference_metrics['training',scope]['incumbent']
                score = math.log(mm['wealth']/bh['wealth'])-.75*mm['max_drawdown']-.002*max(0,mm['orders_per_year']-20)
                scores.append(score)
                deficits.extend((max(0.,math.log(old['wealth']/mm['wealth'])),
                                 max(0.,math.log(max(mm['max_drawdown'],1e-15)/max(old['max_drawdown'],1e-15))),
                                 max(0.,math.log((mm['orders']+1)/(old['orders']+1)))))
                rows.append(dict(candidate=key,pool=scope,objective=score,**values,**mm))
            ranking.append(dict(candidate=key,parameters=values,objective=float(np.mean(scores)),worst_deficit=max(deficits)))
            pd.DataFrame(rows).to_csv(out/'trials.csv',index=False,float_format='%.17g'); write(out/'ranking.json',ranking)
        if name in ('downside_guard', 'funded_ownership', 'adaptive_expectation'):
            selected = min(ranking,key=lambda r:(r['worst_deficit']>1e-12,r['worst_deficit'],-r['objective'],r['candidate']))
            selected['selection_status'] = 'TRAINING_DOMINANCE' if selected['worst_deficit']<=1e-12 else 'NO_TRAINING_DOMINANCE'
        else: selected = min(ranking,key=lambda r:(-r['objective'],r['candidate']))
        write(out/'selection.json',selected); evaluation = []
        for scope in scopes:
            m, refs = contexts['evaluation',scope]
            result = measure(module,name,m,module.Parameters(**selected['parameters']),out/'evaluation'/scope,refs)
            row = dict(family=name,pool=scope,**metrics(result),bull=metrics(result,end='2026-06-30'),
                       summer=metrics(result,start='2026-07-01',end='2026-08-31'))
            evaluation.append(row); all_evaluation.append(row)
        write(out/'evaluation.json',evaluation)
        summary.append(dict(family=name,candidate_count=len(candidates),selection=selected,
                            economic_acceptance='NOT_PASSED',evaluation=evaluation))
        write(output/'report.json',summary)
        print(name,selected,flush=True)
    return summary,all_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives', type=Path, required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--families', nargs='+', choices=sorted(GRIDS), default=list(GRIDS))
    args=parser.parse_args(); archives=args.archives.resolve(); output=args.output.resolve()
    if len(set(args.families))!=len(args.families): raise ValueError('duplicate family')
    root=Path(__file__).resolve().parents[1]; os.chdir(root)
    actual=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if os.environ.get('QUANT_SOURCE_COMMIT')!=actual or subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('require a clean checkout of declared source')
    output.mkdir(parents=True,exist_ok=False); began=time.monotonic()
    request=json.loads((root/'research/run-request.json').read_text()); restore_archives(archives,request,output/'inputs')
    market=load_market(output/'inputs/market',supplement=output/'inputs/supplement',sectors=json.loads((root/'research/catalog.json').read_text())['sectors'])
    if market.fingerprint()!=request['data_sha256']:raise ValueError('frozen market identity mismatch')
    write(output/'SOURCE_IDENTITY.json',{'source':source_identity(),'request':request,'families':args.families,
        'research_files':{str(p.relative_to(root)):file_hash(p) for p in sorted((root/'research').glob('*.py'))}})
    shutil.copytree(archives,output/'frozen_archives')
    subprocess.run(['git','archive','--format=zip','-o',str(output/'source.zip'),'HEAD'],check=True)
    summary,evaluation=study(market,json.loads((root/'research/catalog.json').read_text()),output,args.families)
    for path in output.rglob('manifest.json'):
        if (path.parent/'identity.json').exists():verify_evidence(path.parent)
    lines=['# 独立策略研究实测记录','','全部候选和失败结果保留；训练排名不等于经济验收。2026 年已被观察，不是未触碰样本外。',
           '袖珍账户试验仅诊断，单账户成交金额门槛与整组合不同，不能作为正式比较证据。',
           '', '|策略族|股票池|终值财富|最大回撤|成交笔数|七八月收益|','|---|---|---:|---:|---:|---:|']
    for r in evaluation:
        lines.append(f"|{r['family']}|{r['pool']}|{r['wealth']:.6f}|{r['max_drawdown']:.2%}|{r['orders']}|{r['summer']['total_return']:.2%}|")
    lines += ['', '没有使用旧 SHA 的测量证明当前提交；本任务重新绑定源码、runner、数据与全部回放。',
              '原生四项目对照未在本流程重跑，不引用其旧结果充当本次同执行胜出。任意股票池全面最优没有得到证明。', '']
    (output/'RUN_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    write(output/'execution.json',dict(elapsed_seconds=time.monotonic()-began,status='MEASURED_NOT_ACCEPTED',economic_acceptance='NOT_PASSED'))
    write(output/'MANIFEST.json',{str(p.relative_to(output)):file_hash(p) for p in sorted(output.rglob('*')) if p.is_file()})
    archive=output.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz',compresslevel=6) as stream:stream.add(output,arcname=output.name)
    write(output.with_suffix('.receipt.json'),{'source_commit':actual,'archive_sha256':file_hash(archive),
          'archive_bytes':archive.stat().st_size,'manifest_sha256':file_hash(output/'MANIFEST.json'),
          'economic_acceptance':'NOT_PASSED'})

if __name__=='__main__':main()

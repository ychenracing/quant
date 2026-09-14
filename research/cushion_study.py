"""Matched, finite capital-cushion study. No ordinary CI or acceptance overrides."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import itertools
import json
import math
from pathlib import Path
import time
import numpy as np
import pandas as pd
from techquant.config import Config
from techquant.data import file_hash, load_market
from techquant.engine import run
from techquant.evidence import load_result, metrics, save_result, source_identity
from research.cushion import Owner, Parameters, grid
from research.expectation_study import write_json, scopes


def identity():
    return {'source':source_identity(),'runner_sha256':file_hash(Path(__file__)),
            'helper_sha256':file_hash(Path(__file__).with_name('expectation_study.py')),
            'policy_sha256':file_hash(Path(__file__).with_name('cushion.py')),
            'admission_sha256':file_hash(Path(__file__).with_name('admission.py')),
            'ownership_sha256':file_hash(Path(__file__).with_name('leadership.py'))}


def saved(market, output, params=None, benchmark=None, costs=1., delay=1):
    cfg = Config()
    expected = {'config':asdict(cfg),'universe':list(market.symbols),'quality':market.quality,
        'data_sha256':market.fingerprint(),'source':source_identity(),'provenance':market.provenance,
        'delay':delay,'cost_multiplier':costs,'benchmark':benchmark,
        'start':str(market.calendar[0].date()),'end':str(market.calendar[-1].date()),
        'economic_acceptance':'UNVERIFIED','accounting':'adjusted economic units, not actual shares',
        'study':identity()}
    if params is not None:
        expected['policy'] = Owner(market,params).identity()
    if output.exists():
        return load_result(output, expected=expected)
    result = run(market, cfg, benchmark=benchmark, cost_multiplier=costs, delay=delay,
                 policy_factory=(lambda m,c:Owner(m,params)) if params is not None else None)
    result.metadata['study'] = identity()
    if result.metadata != expected:
        raise AssertionError('unexpected identity')
    save_result(result,output)
    return result


def select(market,catalog,out):
    market=market.prefix('2025-12-31'); scope_map=scopes(market,catalog)
    candidates=grid()
    plan={'identity':identity(),'data_sha256':market.fingerprint(),'selection_end':'2025-12-31',
          'grid':[asdict(p) for p in candidates],'scopes':scope_map}
    if (out/'plan.json').exists() and json.loads((out/'plan.json').read_text())!=plan:
        raise ValueError('cannot overwrite non-equivalent study')
    write_json(out/'plan.json',plan)
    base,hold={},{}
    for name,symbols in scope_map.items():
        m=market.subset(symbols)
        base[name]=metrics(saved(m,out/'runs'/('incumbent_'+name)))
        hold[name]=metrics(saved(m,out/'runs'/('buy_hold_'+name),benchmark='buy_hold'))
    write_json(out/'baselines.json',{'incumbent':base,'buy_hold':hold})
    rows,objectives=[],[]
    for index,p in enumerate(candidates):
        label=f'drawdown{p.drawdown_budget:g}_multiplier{p.multiplier:g}'
        deficits,scores,fills=[],[],0
        for name,symbols in scope_map.items():
            v=metrics(saved(market.subset(symbols),out/'runs'/(label+'_'+name),p))
            b=base[name]
            deficit=max(0.,math.log(b['wealth']/v['wealth']),v['max_drawdown']/max(b['max_drawdown'],1e-12)-1.,v['orders']/max(b['orders'],1)-1.)
            score=math.log(v['wealth']/hold[name]['wealth'])-.75*v['max_drawdown']-.002*max(0.,v['orders_per_year']-20.)
            rows.append({'candidate':index,'scope':name,**asdict(p),**v,'deficit':deficit,'objective':score})
            deficits.append(deficit);scores.append(score);fills+=v['orders']
        objectives.append({'candidate':index,'parameters':asdict(p),'worst_deficit':max(deficits),'objective':float(np.mean(scores)),'fills':fills})
        pd.DataFrame(rows).to_csv(out/'trials.csv',index=False,float_format='%.17g')
        write_json(out/'objectives.json',objectives)
        print(json.dumps(objectives[-1]),flush=True)
    winner=min(objectives,key=lambda r:(r['worst_deficit'],-r['objective'],r['fills'],r['candidate']))
    write_json(out/'selection.json',dict(winner,identity=identity(),data_sha256=market.fingerprint(),
        plan_sha256=file_hash(out/'plan.json'),economic_acceptance='UNVERIFIED',
        status='CORE_NONREGRESSION_ONLY' if winner['worst_deficit']<=1e-12 else 'DIAGNOSTIC_NOT_ACCEPTED'))


def evaluate(market,catalog,selection,out):
    chosen=json.loads(selection.read_text())
    if chosen['identity']!=identity() or chosen['data_sha256']!=market.prefix('2025-12-31').fingerprint() or chosen['plan_sha256']!=file_hash(selection.with_name('plan.json')):
        raise ValueError('selection identity mismatch')
    params=Parameters(**chosen['parameters'])
    windows={'full':(None,None),'bull':('2023-01-03','2026-06-30'),
             'late_june_through_august':('2026-06-22','2026-08-31'),
             'july_august':('2026-07-01','2026-08-31'),'retrospective_2026':('2026-01-01',None)}
    rows=[]
    write_json(out/'plan.json',{'identity':identity(),'data_sha256':market.fingerprint(),'selection_sha256':file_hash(selection),'windows':windows,'parameters':asdict(params)})
    for name,symbols in scopes(market,catalog).items():
        m=market.subset(symbols)
        for policy in ('cushion','incumbent','buy_hold'):
            result=saved(m,out/'runs'/(name+'_'+policy),params if policy=='cushion' else None,
                         benchmark='buy_hold' if policy=='buy_hold' else None)
            for window,(start,end) in windows.items():
                rows.append({'scope':name,'policy':policy,'window':window,**metrics(result,start,end)})
        pd.DataFrame(rows).to_csv(out/'matrix.csv',index=False,float_format='%.17g')
        print(name,'measured',flush=True)
    write_json(out/'status.json',{'identity':identity(),'runs':9,'rows':len(rows),'status':'MEASURED_NOT_UNIVERSAL_ACCEPTANCE'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--supplement',type=Path)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--selection',type=Path)
    args=parser.parse_args();catalog=json.loads(Path(__file__).with_name('catalog.json').read_text())
    market=load_market(args.data,supplement=args.supplement,sectors=catalog['sectors']);before=time.monotonic()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.selection: evaluate(market,catalog,args.selection,args.output)
    else: select(market,catalog,args.output)
    write_json(args.output/'execution.json',{'elapsed_seconds':time.monotonic()-before,'identity':identity()})


if __name__=='__main__':
    main()

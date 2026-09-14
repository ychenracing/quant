"""Bounded ownership experiment. Every trial is preserved, including failures."""
from __future__ import annotations
import argparse,itertools,json,math
from pathlib import Path
from dataclasses import asdict
import pandas as pd
from techquant.data import load_market,file_hash
from techquant.engine import run as reference
from techquant.evidence import metrics,save_result,source_identity
from research.breakout import Parameters,run

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--market',required=True)
    parser.add_argument('--supplement',required=True)
    a=parser.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    catalog=json.loads(Path('research/catalog.json').read_text())
    market=load_market(a.market,supplement=a.supplement,sectors=catalog['sectors'])
    training=market.prefix('2025-12-31')
    scopes={'union':list(market.symbols),'chatgpt_5':catalog['pools']['chatgpt_5'],
            'joint_optical_leader_removal':[s for s in market.symbols if s not in {'sz300308','sz300502','sz300394'}]}
    candidates=[Parameters(*p) for p in itertools.product((20,40,60),(1.5,2.5,3.5),(.5,.7),('trailing','confirmed'))]
    identity={'source':source_identity(),'runner_sha256':file_hash(Path(__file__)),
              'algorithm_sha256':file_hash(Path('research/breakout.py')),'data_sha256':market.fingerprint(),
              'selection_data_sha256':training.fingerprint(),'candidates':[asdict(p) for p in candidates]}
    (out/'identity.json').write_text(json.dumps(identity,indent=2))
    bases={};rows=[];scores=[]
    for scope,symbols in scopes.items():
        r=reference(training.subset(symbols),benchmark='buy_hold');bases[scope]=metrics(r)['wealth']
        save_result(r,out/'train'/f'buy_hold_{scope}')
    for index,p in enumerate(candidates):
        values=[];orders=0
        for scope,symbols in scopes.items():
            r=run(training.subset(symbols),p);mt=metrics(r)
            value=math.log(mt['wealth']/bases[scope])-.75*mt['max_drawdown']-.002*max(0,mt['orders_per_year']-20)
            values.append(value);orders+=mt['orders']
            rows.append({'candidate':index,'scope':scope,'objective':value,**asdict(p),**mt})
            save_result(r,out/'train'/f'candidate_{index}_{scope}')
        item={'candidate':index,'objective':sum(values)/len(values),'orders':orders,**asdict(p)};scores.append(item)
        pd.DataFrame(rows).to_csv(out/'trials.csv',index=False)
        (out/'scores.json').write_text(json.dumps(scores,indent=2))
        print(json.dumps(item),flush=True)
    best=sorted(scores,key=lambda s:(-s['objective'],s['orders'],s['candidate']))[0]
    (out/'selection.json').write_text(json.dumps(best,indent=2));print('SELECTED',best,flush=True)
    p=candidates[best['candidate']];full=[]
    for scope,symbols in scopes.items():
        r=run(market.subset(symbols),p);save_result(r,out/'evaluation'/scope)
        full.append({'scope':scope,**metrics(r),'summer':metrics(r,'2026-07-01','2026-08-31')['total_return'],
                     'bull':metrics(r,end='2026-06-30')['wealth']})
    pd.DataFrame(full).to_csv(out/'evaluation.csv',index=False)
    print(pd.DataFrame(full)[['scope','wealth','max_drawdown','orders','average_exposure','summer','bull']].to_string(index=False),flush=True)

if __name__=='__main__':main()

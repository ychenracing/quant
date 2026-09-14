"""Finite matched study for independently declared ownership policies.

Selection is always pre-2026. Every declared candidate and raw replay is saved.
An improved diagnostic never changes the original economic acceptance contract.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import importlib
import json
import math
from pathlib import Path
import time
import numpy as np
import pandas as pd
from techquant.config import Config
from techquant.data import file_hash,load_market
from techquant.engine import run
from techquant.evidence import load_result,metrics,save_result,source_identity
from research.expectation_study import write_json,scopes


class Study:
    def __init__(self,family:str,*,issued_evidence:Path|None=None):
        if family not in {'shock_ownership','nonlinear','observed_trend','pathwise','coherent','trend_book','recovery_memory','support_budget','risk_reliability','funded_risk','quantity_obligation'}:raise ValueError('undeclared research family')
        self.family=family
        self.module=importlib.import_module('research.'+family)
        self.issued=None
        if family in {'coherent','risk_reliability'}:
            if issued_evidence is None:raise ValueError(f'{family} comparison requires pinned issued evidence')
            from research.issued_forecasts import IssuedForecasts
            self.issued=IssuedForecasts(issued_evidence)
        elif issued_evidence is not None:
            raise ValueError('issued forecast injection is not declared for this family')

    def identity(self):
        root=Path(__file__).parent
        names=['finite_study.py','expectation_study.py',self.family+'.py',
               self.family+'_contract.json','leadership.py','expectation.py']
        if self.family in {'observed_trend','pathwise','coherent','trend_book','recovery_memory'}:names+=['nonlinear.py','nonlinear_contract.json']
        if self.family in {'pathwise','coherent','trend_book','recovery_memory'}:names+=['observed_trend.py','observed_trend_contract.json']
        if self.family=='coherent':names+=['issued_forecasts.py','pathwise.py','pathwise_contract.json']
        if self.family in {'trend_book','recovery_memory'}:names+=['coherent.py','coherent_contract.json','pathwise.py','pathwise_contract.json']
        if self.family=='recovery_memory':names+=['trend_book.py','trend_book_contract.json']
        if self.family=='risk_reliability':
            names+=['issued_forecasts.py','trend_book.py','trend_book_contract.json',
                    'coherent.py','coherent_contract.json','observed_trend.py',
                    'nonlinear.py','pathwise.py','observed_trend_contract.json',
                    'nonlinear_contract.json','pathwise_contract.json']
        if self.family=='quantity_obligation':names+=['support_budget.py','support_budget_contract.json','funded_risk.py','funded_risk_contract.json']
        if self.family=='funded_risk':names+=['support_budget.py','support_budget_contract.json']
        return {'source':source_identity(),'family':self.family,
                'dependencies':{name:file_hash(root/name) for name in names},
                **({'issued_forecasts':self.issued.identity()} if self.issued else {})}

    def saved(self,market,path,parameters=None,benchmark=None,costs=1.,delay=1):
        cfg=Config()
        expected={'config':asdict(cfg),'universe':list(market.symbols),'quality':market.quality,
            'data_sha256':market.fingerprint(),'source':source_identity(),'provenance':market.provenance,
            'delay':delay,'cost_multiplier':costs,'benchmark':benchmark,
            'start':str(market.calendar[0].date()),'end':str(market.calendar[-1].date()),
            'economic_acceptance':'UNVERIFIED','accounting':'adjusted economic units, not actual shares',
            'study':self.identity()}
        owner=None
        if parameters is not None:
            if self.issued:
                forecast,origin=self.issued.load(market)
                owner=self.module.Owner(market,parameters,prediction=forecast)
            else:
                owner=self.module.Owner(market,parameters)
        if owner is not None:expected['policy']=owner.identity()
        if self.family=='risk_reliability' and owner is not None:
            owner.preserve_audit(path.parent.parent/'audits', require_existing=path.exists())
        intent_path=path.parent.parent/'intents'/(path.name+'.json')
        if path.exists():
            if self.family=='quantity_obligation' and owner is not None:
                self.module.verify_trace(intent_path,expected)
            return load_result(path,expected=expected)
        factory=(lambda m,c:owner) if owner is not None else None
        result=run(market,cfg,benchmark=benchmark,policy_factory=factory,cost_multiplier=costs,delay=delay)
        result.metadata['study']=self.identity()
        if result.metadata!=expected:raise AssertionError('unexpected study identity')
        save_result(result,path)
        if self.family=='quantity_obligation' and owner is not None:
            self.module.preserve_trace(intent_path,expected,owner.trace)
        prediction=getattr(owner,'f',None) or getattr(getattr(owner,'inner',None),'f',None)
        if prediction is not None:
            # Preserve issued arrays, not only hashes that require a potentially
            # different numerical runtime to reconstruct later diagnostics.
            folder=path.parent.parent/'forecasts'/market.fingerprint()
            target=folder/(prediction.fingerprint()+'.npz')
            arrays={key:getattr(prediction,key) for key in
                    ('expected','tail','ready','price','ema10','ema20','ema60','momentum5','ret1')}
            if hasattr(prediction,'outcome_probability'):
                arrays['outcome_probability']=prediction.outcome_probability
            if target.exists():
                with np.load(target,allow_pickle=False) as previous:
                    if any(not np.array_equal(previous[k],v,equal_nan=True) for k,v in arrays.items()):
                        raise ValueError('same forecast fingerprint has different preserved arrays')
            else:
                folder.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(target,**arrays)
                write_json(target.with_suffix('.json'),{'data_sha256':market.fingerprint(),
                    'forecast_sha256':prediction.fingerprint(),'symbols':list(market.symbols),
                    'dates':[str(d.date()) for d in market.calendar],'fits':prediction.fits,
                    'source':source_identity(),'archive_sha256':file_hash(target),
                    **({'issuing_source':origin} if self.issued else {})})
        return result

    def select(self,market,catalog,out):
        market=market.prefix('2025-12-31');scoped=scopes(market,catalog)
        grid=self.module.grid()
        plan={'identity':self.identity(),'data_sha256':market.fingerprint(),
              'selection_end':'2025-12-31','grid':[asdict(p) for p in grid],'scopes':scoped}
        if (out/'plan.json').exists() and json.loads((out/'plan.json').read_text())!=plan:
            raise ValueError('existing plan has different source/data/runtime/candidates')
        write_json(out/'plan.json',plan)
        base,hold={},{}
        for name,names in scoped.items():
            m=market.subset(names)
            base[name]=metrics(self.saved(m,out/'runs'/('incumbent_'+name)))
            hold[name]=metrics(self.saved(m,out/'runs'/('buy_hold_'+name),benchmark='buy_hold'))
        write_json(out/'baselines.json',{'incumbent':base,'buy_hold':hold})
        rows,scores=[],[]
        for number,p in enumerate(grid):
            deficits,objectives,fills=[],[],0
            for name,names in scoped.items():
                v=metrics(self.saved(market.subset(names),out/'runs'/(f'candidate{number}_'+name),p))
                b=base[name]
                deficit=max(0.,math.log(b['wealth']/v['wealth']),v['max_drawdown']/max(b['max_drawdown'],1e-12)-1.,v['orders']/max(b['orders'],1)-1.)
                objective=math.log(v['wealth']/hold[name]['wealth'])-.75*v['max_drawdown']-.002*max(0.,v['orders_per_year']-20.)
                rows.append({'candidate':number,'scope':name,**asdict(p),**v,'deficit':deficit,'objective':objective})
                deficits.append(deficit);objectives.append(objective);fills+=v['orders']
            scores.append({'candidate':number,'parameters':asdict(p),'worst_deficit':max(deficits),'objective':float(np.mean(objectives)),'fills':fills})
            pd.DataFrame(rows).to_csv(out/'trials.csv',index=False,float_format='%.17g')
            write_json(out/'objectives.json',scores)
            print(json.dumps(scores[-1]),flush=True)
        selected=min(scores,key=lambda v:(v['worst_deficit'],-v['objective'],v['fills'],v['candidate']))
        write_json(out/'selection.json',dict(selected,identity=self.identity(),data_sha256=market.fingerprint(),
            plan_sha256=file_hash(out/'plan.json'),economic_acceptance='UNVERIFIED',
            status='CORE_NONREGRESSION_ONLY' if selected['worst_deficit']<=1e-12 else 'DIAGNOSTIC_NOT_ACCEPTED'))

    def evaluate(self,market,catalog,selection,out):
        chosen=json.loads(selection.read_text())
        if (chosen['identity']!=self.identity() or chosen['data_sha256']!=market.prefix('2025-12-31').fingerprint()
            or chosen['plan_sha256']!=file_hash(selection.with_name('plan.json'))):
            raise ValueError('selection identity mismatch')
        p=self.module.Parameters(**chosen['parameters'])
        windows={'full':(None,None),'bull':('2023-01-03','2026-06-30'),
                 'late_june_through_august':('2026-06-22','2026-08-31'),
                 'july_august':('2026-07-01','2026-08-31'),'retrospective_2026':('2026-01-01',None)}
        write_json(out/'plan.json',{'identity':self.identity(),'data_sha256':market.fingerprint(),
            'selection_sha256':file_hash(selection),'parameters':asdict(p),'windows':windows})
        rows=[]
        for name,names in scopes(market,catalog).items():
            m=market.subset(names)
            for policy in (self.family,'incumbent','buy_hold'):
                result=self.saved(m,out/'runs'/(name+'_'+policy),p if policy==self.family else None,
                                  benchmark='buy_hold' if policy=='buy_hold' else None)
                for window,(start,end) in windows.items():
                    rows.append({'scope':name,'policy':policy,'window':window,**metrics(result,start,end)})
            pd.DataFrame(rows).to_csv(out/'matrix.csv',index=False,float_format='%.17g')
            print(name,'measured',flush=True)
        write_json(out/'status.json',{'identity':self.identity(),'runs':9,'rows':len(rows),
                                     'status':'MEASURED_NOT_UNIVERSAL_ACCEPTANCE'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family',required=True);parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--supplement',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--selection',type=Path)
    parser.add_argument('--issued-evidence',type=Path)
    args=parser.parse_args();before=time.monotonic()
    catalog=json.loads(Path(__file__).with_name('catalog.json').read_text())
    market=load_market(args.data,supplement=args.supplement,sectors=catalog['sectors'])
    study=Study(args.family,issued_evidence=args.issued_evidence);args.output.mkdir(parents=True,exist_ok=True)
    if args.selection:study.evaluate(market,catalog,args.selection,args.output)
    else:study.select(market,catalog,args.output)
    write_json(args.output/'execution.json',{'elapsed_seconds':time.monotonic()-before,'identity':study.identity()})


if __name__=='__main__':main()

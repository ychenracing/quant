"""One immutable paired test of simultaneous funding within unchanged resources."""
from dataclasses import asdict
import json
from pathlib import Path
import pandas as pd
from techquant.config import Config
from techquant.data import file_hash
from techquant.engine import run
from techquant.evidence import source_identity,save_result,load_result,metrics
from research.finite_study import Study as ParentStudy
from research.expectation_study import write_json,scopes
from research.decision_review import paired_screen
from research.ledger_attribution import attribute
from research import joint_funding as policy

REGISTRATION='637fdb539580c1ecddbff9958d74ba79c27ce1c1'


class Study:
    def __init__(self,family='joint_funding',*,issued_evidence=None):
        if family!='joint_funding' or issued_evidence is not None:
            raise ValueError('only the fixed joint-funding pair is supported')
        self.module=policy

    def identity(self):
        root=Path(__file__).parent
        files=('joint_resources.py','joint_funding.py','joint_funding_study.py',
               'joint_funding_contract.json','decision_review.py','ledger_attribution.py')
        return {'family':'joint_funding','source':source_identity(),
            'parent_study':ParentStudy('observed_admission_completion').identity(),
            'dependencies':{name:file_hash(root/name) for name in files}}

    def saved(self,market,path,parameters,*,costs=1.,delay=1):
        path=Path(path);cfg=Config();owner=policy.Owner(market,parameters,config=cfg)
        expected={'config':asdict(cfg),'universe':list(market.symbols),'quality':market.quality,
            'data_sha256':market.fingerprint(),'source':source_identity(),'provenance':market.provenance,
            'delay':delay,'cost_multiplier':costs,'benchmark':None,
            'start':str(market.calendar[0].date()),'end':str(market.calendar[-1].date()),
            'economic_acceptance':'UNVERIFIED','accounting':'adjusted economic units, not actual shares',
            'study':self.identity(),'policy':owner.identity()}
        trace=path.parent.parent/'intents'/(path.name+'.json')
        if path.exists():
            policy.verify_trace(trace,expected)
            return load_result(path,expected=expected)
        result=run(market,cfg,policy_factory=lambda m,c:owner,cost_multiplier=costs,delay=delay)
        result.metadata['study']=self.identity()
        if result.metadata!=expected:raise AssertionError('joint-funding identity mismatch')
        save_result(result,path);policy.preserve_trace(trace,expected,owner.trace)
        return result

    def select(self,market,catalog,out):
        out=Path(out);contract=json.loads(Path(__file__).with_name('joint_funding_contract.json').read_text())
        if market.fingerprint()!=contract['data']['full_sha256']:
            raise ValueError('joint funding requires the frozen full market')
        market=market.prefix(contract['data']['selection_end']);scope_map=scopes(market,catalog)
        plan={'identity':self.identity(),'data_sha256':market.fingerprint(),'scopes':scope_map,
              'parameters':[asdict(p) for p in policy.grid()],'registration_commit':REGISTRATION}
        if (out/'plan.json').exists() and json.loads((out/'plan.json').read_text())!=plan:
            raise ValueError('existing joint funding plan is not equivalent')
        write_json(out/'plan.json',plan);rows=[]
        for scope,names in scope_map.items():
            m=market.subset(names);row={'scope':scope}
            for enabled,label in ((False,'control'),(True,'treatment')):
                name=scope+'_'+label
                result=self.saved(m,out/'runs'/name,policy.Parameters(enabled))
                days,episodes,audit=attribute(m,result)
                target=out/'attribution'/name;target.mkdir(parents=True,exist_ok=True)
                days.to_csv(target/'pnl.csv',index=False,float_format='%.17g')
                episodes.to_csv(target/'episodes.csv',index=False,float_format='%.17g')
                row[label]=dict(metrics(result),average_exposure=float(result.equity.exposure.mean()))
                row[label+'_ledger']=audit
                if enabled:
                    trace=json.loads((out/'intents'/(name+'.json')).read_text())['trace']
                    calls=[r for r in trace if r.get('kind')=='JOINT_RESOURCE_FUNDING']
                    row['funding']={'calls':len(calls),'held_fresh_competition':sum(r['held_fresh_competition'] for r in calls),
                        'multi_leg_funding':sum(r['funded_legs']>1 for r in calls),
                        'cash_violation':sum(sum(r['notional_additions'])>r['cash_authority']+1e-7 for r in calls),
                        'risk_violation':sum(r['funded_stop_risk']>r['remaining_risk']+1e-7 for r in calls)}
            rows.append(row);write_json(out/'paired-progress.json',rows);print(json.dumps(row),flush=True)
        decision=paired_screen(rows)
        decision.update(status='PAIRED_SCREEN_ADVANCE' if decision['advance'] else 'REJECTED_PAIRED_SCREEN',
            rows=rows,identity=self.identity(),data_sha256=market.fingerprint(),new_accounts=6,
            candidate=int(decision['advance']),plan_sha256=file_hash(out/'plan.json'),
            economic_acceptance='NOT_ESTABLISHED',historical_exposure='RETROSPECTIVE_NOT_UNSEEN_OUT_OF_SAMPLE')
        write_json(out/'selection.json',decision)

    def evaluate(self,market,catalog,selection,out):
        selection,out=Path(selection),Path(out);chosen=json.loads(selection.read_text())
        if (chosen['identity']!=self.identity() or chosen['data_sha256']!=market.prefix('2025-12-31').fingerprint()
                or chosen['plan_sha256']!=file_hash(selection.with_name('plan.json'))):
            raise ValueError('joint funding selection identity mismatch')
        if not chosen['advance']:
            write_json(out.parent/'evaluation-decision.json',{'status':'NOT_RUN_REJECTED_PAIR',
                'selection_sha256':file_hash(selection),'treatment_2026_runs':0})
            return
        windows={'full':(None,None),'bull':('2023-01-03','2026-06-30'),
                 'late_june_through_august':('2026-06-22','2026-08-31'),
                 'july_august':('2026-07-01','2026-08-31'),'retrospective_2026':('2026-01-01',None)}
        write_json(out/'plan.json',{'identity':self.identity(),'data_sha256':market.fingerprint(),
                                  'selection_sha256':file_hash(selection),'windows':windows})
        rows=[]
        for scope,names in scopes(market,catalog).items():
            for enabled,label in ((False,'control'),(True,'treatment')):
                result=self.saved(market.subset(names),out/'runs'/(scope+'_'+label),policy.Parameters(enabled))
                for window,(start,end) in windows.items():
                    rows.append({'scope':scope,'policy':label,'window':window,**metrics(result,start,end)})
            pd.DataFrame(rows).to_csv(out/'matrix.csv',index=False,float_format='%.17g')
        write_json(out/'status.json',{'identity':self.identity(),'runs':6,'rows':len(rows),
                    'status':'MEASURED_NOT_ORIGINAL_ACCEPTANCE','final_matrix':'NOT_RUN'})

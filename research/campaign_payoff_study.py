"""A paired cash-ledger test of settled-campaign payoff priority, with no grid."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
from techquant.config import Config
from techquant.data import file_hash
from techquant.engine import run
from techquant.evidence import source_identity,save_result,load_result,metrics
from research.finite_study import Study as ParentStudy
from research.expectation_study import write_json,scopes
from research.decision_review import paired_screen
from research.ledger_attribution import attribute
from research.observed_admission_completion import Parameters as ParentParameters
from research import campaign_payoff_policy as policy

REGISTRATION='9755458a87692c490f13452f7d9173b05dc5f2c7'


class Study:
    def __init__(self,family='campaign_payoff',*,issued_evidence=None):
        if family!='campaign_payoff' or issued_evidence is not None:
            raise ValueError('only the registered campaign-payoff pair is supported')
        self.module=policy
        self.parent=ParentStudy('observed_admission_completion')

    def identity(self):
        root=Path(__file__).parent
        names=('campaign_payoff.py','campaign_payoff_policy.py','campaign_payoff_study.py',
               'campaign_payoff_contract.json','relative_rank.py','ledger_attribution.py','decision_review.py')
        return {'family':'campaign_payoff','source':source_identity(),
                'parent_study':self.parent.identity(),
                'dependencies':{name:file_hash(root/name) for name in names}}

    def saved(self,market,path,parameters,*,reference=None,costs=1.,delay=1):
        path=Path(path)
        if not parameters.learn_payoff:
            if reference is not None:raise ValueError('control does not consume reference labels')
            return self.parent.saved(market,path,ParentParameters(),costs=costs,delay=delay)
        if reference is None:raise ValueError('reuse the matched control rather than dispatch a second shadow')
        cfg=Config();owner=policy.Owner(market,parameters,config=cfg,reference=reference,costs=costs,delay=delay)
        expected={'config':asdict(cfg),'universe':list(market.symbols),'quality':market.quality,
            'data_sha256':market.fingerprint(),'source':source_identity(),'provenance':market.provenance,
            'delay':delay,'cost_multiplier':costs,'benchmark':None,
            'start':str(market.calendar[0].date()),'end':str(market.calendar[-1].date()),
            'economic_acceptance':'UNVERIFIED','accounting':'adjusted economic units, not actual shares',
            'study':self.identity(),'policy':owner.identity()}
        intent=path.parent.parent/'intents'/(path.name+'.json')
        folder=path.parent.parent/'payoffs';arrays=folder/(path.name+'.npz');meta_path=folder/(path.name+'.json')
        samples=[asdict(r) for r in owner.samples]
        if path.exists():
            policy.verify_trace(intent,expected);meta=json.loads(meta_path.read_text())
            if (meta['identity']!=expected or meta['fits']!=owner.fits or meta['samples']!=samples
                    or meta['npz_sha256']!=file_hash(arrays)):
                raise ValueError('saved campaign learning evidence does not match')
            with np.load(arrays,allow_pickle=False) as old:
                if not np.array_equal(old['payoffs'],owner.payoffs,equal_nan=True):
                    raise ValueError('saved issued payoffs differ')
            return load_result(path,expected=expected)
        result=run(market,cfg,policy_factory=lambda m,c:owner,cost_multiplier=costs,delay=delay)
        result.metadata['study']=self.identity()
        if result.metadata!=expected:raise AssertionError('unexpected campaign study identity')
        save_result(result,path);policy.preserve_trace(intent,expected,owner.trace)
        folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(arrays,payoffs=owner.payoffs)
        calls=[r for r in owner.trace if r.get('kind')=='SETTLED_CAMPAIGN_PAYOFF_PRIORITY']
        write_json(meta_path,{'identity':expected,'fits':owner.fits,'samples':samples,
            'label_meta':owner.label_meta,'npz_sha256':file_hash(arrays),
            'priority_calls':len(calls),'changed_priority_sessions':len({r['session'] for r in calls if r['changed']}),
            'trained_sessions':int(np.isfinite(owner.payoffs).any(axis=1).sum())})
        return result

    def select(self,market,catalog,out):
        out=Path(out);root=Path(__file__).parent
        contract=json.loads((root/'campaign_payoff_contract.json').read_text())
        if market.fingerprint()!=contract['data']['full_sha256']:
            raise ValueError('only the original frozen full market is admissible')
        market=market.prefix(contract['data']['selection_end']);scope_map=scopes(market,catalog)
        plan={'identity':self.identity(),'data_sha256':market.fingerprint(),'scopes':scope_map,
              'parameters':[asdict(p) for p in policy.grid()],'registration_commit':REGISTRATION,
              'contract_sha256':file_hash(root/'campaign_payoff_contract.json')}
        if (out/'plan.json').exists() and json.loads((out/'plan.json').read_text())!=plan:
            raise ValueError('existing campaign plan differs')
        write_json(out/'plan.json',plan);rows=[]
        for scope,names in scope_map.items():
            m=market.subset(names);row={'scope':scope};reference=None
            for enabled,label in ((False,'control'),(True,'treatment')):
                name=scope+'_'+label
                result=self.saved(m,out/'runs'/name,policy.Parameters(enabled),reference=reference)
                if not enabled:reference=result
                days,episodes,audit=attribute(m,result)
                folder=out/'attribution'/name;folder.mkdir(parents=True,exist_ok=True)
                days.to_csv(folder/'pnl.csv',index=False,float_format='%.17g')
                episodes.to_csv(folder/'episodes.csv',index=False,float_format='%.17g')
                row[label]=dict(metrics(result),average_exposure=float(result.equity.exposure.mean()));row[label+'_ledger']=audit
                if enabled:
                    meta=json.loads((out/'payoffs'/(name+'.json')).read_text())
                    row['learning']={k:meta[k] for k in ('priority_calls','changed_priority_sessions','trained_sessions')}
                    row['learning'].update(settled_campaigns=len(meta['samples']),censored_open=meta['label_meta']['censored_open'])
            rows.append(row);write_json(out/'paired-progress.json',rows);print(json.dumps(row),flush=True)
        decision=paired_screen(rows)
        decision.update(status='PAIRED_SCREEN_ADVANCE' if decision['advance'] else 'REJECTED_PAIRED_SCREEN',
            rows=rows,identity=self.identity(),data_sha256=market.fingerprint(),candidate=int(decision['advance']),
            new_accounts=6,additional_shadow_accounts=0,plan_sha256=file_hash(out/'plan.json'),
            economic_acceptance='NOT_ESTABLISHED',historical_exposure='RETROSPECTIVE_NOT_UNSEEN_OUT_OF_SAMPLE')
        write_json(out/'selection.json',decision)

    def evaluate(self,market,catalog,selection,out):
        selection,out=Path(selection),Path(out);chosen=json.loads(selection.read_text())
        if (chosen['identity']!=self.identity() or chosen['data_sha256']!=market.prefix('2025-12-31').fingerprint()
                or chosen['plan_sha256']!=file_hash(selection.with_name('plan.json'))):
            raise ValueError('campaign selection identity mismatch')
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
            reference=None;m=market.subset(names)
            for enabled,label in ((False,'control'),(True,'treatment')):
                result=self.saved(m,out/'runs'/(scope+'_'+label),policy.Parameters(enabled),reference=reference)
                if not enabled:reference=result
                for window,(start,end) in windows.items():
                    rows.append({'scope':scope,'policy':label,'window':window,**metrics(result,start,end)})
            pd.DataFrame(rows).to_csv(out/'matrix.csv',index=False,float_format='%.17g')
        write_json(out/'status.json',{'identity':self.identity(),'runs':6,'rows':len(rows),
                    'status':'MEASURED_NOT_ORIGINAL_ACCEPTANCE','final_matrix':'NOT_RUN'})

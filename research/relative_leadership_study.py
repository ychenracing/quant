"""One source-bound relative-ranking pair; rejected treatments never see 2026."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
from techquant.config import Config
from techquant.data import file_hash
from techquant.engine import run
from techquant.evidence import source_identity, save_result, load_result, metrics
from research.finite_study import Study as ParentStudy
from research.expectation_study import write_json, scopes
from research.decision_review import paired_screen
from research.ledger_attribution import attribute
from research import relative_leadership as policy


class Study:
    def __init__(self, family='relative_leadership', *, issued_evidence=None):
        if family!='relative_leadership' or issued_evidence is not None:
            raise ValueError('only the fixed relative-leadership pair is supported')
        self.module=policy

    def identity(self):
        root=Path(__file__).parent
        names=('relative_rank.py','relative_leadership.py','relative_leadership_study.py',
               'relative_leadership_contract.json','decision_review.py','ledger_attribution.py')
        return {'family':'relative_leadership', 'source':source_identity(),
            'parent_study':ParentStudy('observed_admission_completion').identity(),
            'relative_dependencies':{name:file_hash(root/name) for name in names}}

    def saved(self, market, path, parameters, *, costs=1., delay=1):
        path=Path(path);cfg=Config();owner=policy.Owner(market,parameters,config=cfg)
        identity=self.identity()
        expected={'config':asdict(cfg),'universe':list(market.symbols),'quality':market.quality,
            'data_sha256':market.fingerprint(),'source':source_identity(),'provenance':market.provenance,
            'delay':delay,'cost_multiplier':costs,'benchmark':None,
            'start':str(market.calendar[0].date()),'end':str(market.calendar[-1].date()),
            'economic_acceptance':'UNVERIFIED','accounting':'adjusted economic units, not actual shares',
            'study':identity,'policy':owner.identity()}
        intent=path.parent.parent/'intents'/(path.name+'.json')
        ranks=path.parent.parent/'rankings'/(path.name+'.npz')
        rank_meta=ranks.with_suffix('.json')
        if path.exists():
            policy.verify_trace(intent,expected)
            if owner.relative is not None:
                meta=json.loads(rank_meta.read_text())
                if (meta['identity']!=expected or meta['fits']!=owner.fits
                        or meta['npz_sha256']!=file_hash(ranks)):
                    raise ValueError('saved relative ranks have incompatible identity or bytes')
                with np.load(ranks,allow_pickle=False) as prior:
                    if not np.array_equal(prior['relative'],owner.relative,equal_nan=True):
                        raise ValueError('issued relative ranks cannot be relabelled')
            return load_result(path,expected=expected)
        result=run(market,cfg,policy_factory=lambda m,c:owner,cost_multiplier=costs,delay=delay)
        result.metadata['study']=identity
        if result.metadata!=expected:
            raise AssertionError('unexpected relative study source/config/account identity')
        save_result(result,path);policy.preserve_trace(intent,expected,owner.trace)
        if owner.relative is not None:
            ranks.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(ranks,relative=owner.relative)
            calls=[row for row in owner.trace if row.get('kind')=='RELATIVE_LEADERSHIP_PRIORITY']
            write_json(rank_meta,{'identity':expected,'fits':owner.fits,'npz_sha256':file_hash(ranks),
                'priority_calls':len(calls),'changed_priority_calls':sum(row['changed'] for row in calls),
                'changed_priority_sessions':len({row['session'] for row in calls if row['changed']}),
                'trained_rank_sessions':int(np.isfinite(owner.relative).any(axis=1).sum())})
        return result

    def select(self, market, catalog, out):
        out=Path(out);root=Path(__file__).parent
        contract=json.loads((root/'relative_leadership_contract.json').read_text())
        if market.fingerprint()!=contract['data']['full_sha256']:
            raise ValueError('the fixed frozen full market is required')
        market=market.prefix(contract['data']['selection_end'])
        plan={'identity':self.identity(),'data_sha256':market.fingerprint(),
              'scopes':scopes(market,catalog),'parameters':[asdict(p) for p in policy.grid()],
              'registration_commit':'98341651d0320cb7be37c653ac32a03f2b581985',
              'contract_sha256':file_hash(root/'relative_leadership_contract.json')}
        if (out/'plan.json').exists() and json.loads((out/'plan.json').read_text())!=plan:
            raise ValueError('paired plan is not source/input equivalent')
        write_json(out/'plan.json',plan)
        rows=[]
        for scope,names in scopes(market,catalog).items():
            m=market.subset(names);row={'scope':scope}
            for enabled,label in ((False,'control'),(True,'treatment')):
                name=scope+'_'+label;result=self.saved(m,out/'runs'/name,policy.Parameters(enabled))
                days,episodes,audit=attribute(m,result)
                destination=out/'attribution'/name;destination.mkdir(parents=True,exist_ok=True)
                days.to_csv(destination/'pnl.csv',index=False,float_format='%.17g')
                episodes.to_csv(destination/'episodes.csv',index=False,float_format='%.17g')
                row[label]=dict(metrics(result),average_exposure=float(result.equity.exposure.mean()))
                row[label+'_ledger']=audit
                if enabled:
                    meta=json.loads((out/'rankings'/(name+'.json')).read_text())
                    row['priority_diagnostics']={key:meta[key] for key in
                        ('priority_calls','changed_priority_calls','changed_priority_sessions','trained_rank_sessions')}
            rows.append(row);write_json(out/'paired-progress.json',rows)
            print(json.dumps(row),flush=True)
        decision=paired_screen(rows)
        decision.update(status='PAIRED_SCREEN_ADVANCE' if decision['advance'] else 'REJECTED_PAIRED_SCREEN',
            rows=rows,identity=self.identity(),data_sha256=market.fingerprint(),
            candidate=1 if decision['advance'] else 0,new_accounts=6,
            parameters=asdict(policy.Parameters(bool(decision['advance']))),
            plan_sha256=file_hash(out/'plan.json'),economic_acceptance='NOT_ESTABLISHED',
            historical_exposure='RETROSPECTIVE_NOT_UNSEEN_OUT_OF_SAMPLE',
            research_budget='One newly registered relative-leadership hypothesis; no neighboring retries')
        write_json(out/'selection.json',decision)

    def evaluate(self, market, catalog, selection, out):
        selection,out=Path(selection),Path(out);chosen=json.loads(selection.read_text())
        if (chosen['identity']!=self.identity()
                or chosen['data_sha256']!=market.prefix('2025-12-31').fingerprint()
                or chosen['plan_sha256']!=file_hash(selection.with_name('plan.json'))):
            raise ValueError('paired selection identity mismatch')
        if not chosen['advance']:
            write_json(out.parent/'evaluation-decision.json',{'status':'NOT_RUN_REJECTED_PAIR',
                'selection_sha256':file_hash(selection),'treatment_2026_runs':0})
            return
        windows={'full':(None,None),'bull':('2023-01-03','2026-06-30'),
            'late_june_through_august':('2026-06-22','2026-08-31'),
            'july_august':('2026-07-01','2026-08-31'),
            'retrospective_2026':('2026-01-01',None)}
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

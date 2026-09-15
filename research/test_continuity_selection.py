"""Causal information priority cannot change eligibility, funding or protection."""
from importlib import import_module, util
from dataclasses import replace
from pathlib import Path
import unittest,tempfile
import numpy as np,pandas as pd
from techquant.config import Config
from techquant.data import Market
from techquant.engine import run
from techquant.policy import CloseObservation
from research.test_allocation_auction import market
from research.observed_admission_completion import Owner as Parent,Parameters as ParentParameters


class ContinuityTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(util.find_spec('research.continuity_selection'),'registered implementation is absent')
        return import_module('research.continuity_selection')

    def test_registered_boolean_and_invalid_window(self):
        mod=self.module();self.assertEqual([x.continuous_information_priority for x in mod.grid()],[False,True])
        for value in (0,1,'True',None):
            with self.assertRaises(ValueError):mod.Parameters(value)
        for value in (0,1.5,True):
            with self.assertRaises(ValueError):mod.information_continuity(market(),value)

    def test_direction_counts_zero_and_missing_are_distinct(self):
        mod=self.module();m=market(days=15,names=2)
        frames={s:f.copy() for s,f in m.frames.items()};s=m.symbols[0]
        prices=[10,11,11,10,20,21,20,19,18,17,18,19,20,21,22]
        for col in ('open','high','low','close','raw_open','raw_close'):frames[s][col]=prices
        frames[s].loc[m.calendar[4],'volume']=0
        adjusted=Market.from_frames(frames,m.calendar,sectors=m.sectors,quality='synthetic')
        a=mod.information_continuity(adjusted,40)
        np.testing.assert_allclose(a[:7,0],[1.,1.,.75,.5,.5,.5,.375],rtol=0,atol=1e-14)
        self.assertTrue(np.isfinite(a).all());self.assertTrue(((0<=a)&(a<=1)).all())

    def test_priority_changes_only_order_not_original_scores_or_eligibility(self):
        mod=self.module();owner=mod.Owner(market(names=3),mod.Parameters(True));p=owner.inner;i=50
        p.features.score[i]=[2.,1.,.5];p.continuity[i]=[.2,.8,1.]
        before=p.features.score.copy();entry=p.features.entry.copy();exits=p.features.exit.copy()
        self.assertEqual(p._allocation_order(i,np.array([0,1,2])),[1,2,0])
        np.testing.assert_array_equal(p.features.score,before);np.testing.assert_array_equal(p.features.entry,entry);np.testing.assert_array_equal(p.features.exit,exits)
        self.assertEqual(p._allocation_order(i,np.array([0])),[0])
        p.continuity[i]=1.
        self.assertEqual(p._allocation_order(i,np.array([0,1,2])),[0,1,2])

    def test_neutral_hook_exactly_matches_original_order_and_control(self):
        mod=self.module();m=market(days=120,names=5);p=Parent(m,ParentParameters()).inner
        for i in (20,50,100):
            indices=np.arange(len(m.symbols));expected=sorted(indices,key=lambda j:(-p.features.score[i,j],m.symbols[j]))
            self.assertEqual(p._allocation_order(i,indices),expected)
        a=run(m,policy_factory=lambda m,c:Parent(m,ParentParameters(),config=c))
        b=run(m,policy_factory=lambda m,c:mod.Owner(m,mod.Parameters(False),config=c))
        pd.testing.assert_frame_equal(a.equity,b.equity,check_exact=True);pd.testing.assert_frame_equal(a.targets,b.targets,check_exact=True);self.assertEqual(a.orders,b.orders)

    def test_cap_cash_stops_and_partial_exit_survive(self):
        mod=self.module();m=market(names=5);owner=mod.Owner(m,mod.Parameters(True));p=owner.inner
        p.features.entry[:]=True;p.features.exit[:]=False;p.ready[:]=True
        p.risk.cap=1.;price=p.features.close[40];units=np.zeros(5);units[:2]=.2*2e6/price[:2];p.previous_units=units.copy();p.stop[0]=1e6
        for i,amount in ((40,units[0]),(41,units[0]/2)):
            actual=units.copy();actual[0]=amount;px=p.features.close[i];obs=CloseObservation.from_inventory(i,str(m.calendar[i].date()),2e6,float(2e6-actual@px),actual,actual*px/2e6)
            d=owner.decide(obs);self.assertEqual(d.unit_targets[0],0);self.assertTrue(np.all(d.unit_targets<=actual+1e-10));self.assertEqual(len(p.history),i-39)
        self.assertEqual(p.params.positions,2);self.assertEqual(p.params.risk_budget,.10)

    def test_prefix_removed_inputs_config_and_actual_engine(self):
        mod=self.module();m=market(days=105,names=5);cfg=Config(slow=50);date=m.calendar[75]
        a=mod.information_continuity(m,cfg.slow);b=mod.information_continuity(m.prefix(date),cfg.slow);np.testing.assert_array_equal(a[:76],b)
        factory=lambda m,c:mod.Owner(m,mod.Parameters(True),config=c)
        full=run(m,cfg,policy_factory=factory,delay=2);short=run(m.prefix(date),cfg,policy_factory=factory,delay=2)
        pd.testing.assert_frame_equal(full.equity.loc[:date],short.equity,check_exact=True);pd.testing.assert_frame_equal(full.targets.loc[:date],short.targets,check_exact=True)
        self.assertEqual([o for o in full.orders if o['date']<=str(date.date())],short.orders);self.assertGreaterEqual(float(full.equity.cash.min()),-1e-7)
        names=m.symbols[:3];first=run(m.subset(names),cfg,policy_factory=factory)
        frames={s:f.copy() for s,f in m.frames.items()}
        for col in ('open','high','low','close','raw_open','raw_close'):frames[m.symbols[-1]][col]*=2
        changed=Market.from_frames(frames,m.calendar,sectors=m.sectors,quality='synthetic');second=run(changed.subset(names),cfg,policy_factory=factory)
        pd.testing.assert_frame_equal(first.equity,second.equity,check_exact=True);self.assertEqual(first.orders,second.orders)

    def test_completion_uses_the_same_priority_without_new_authority(self):
        mod=self.module();owner=mod.Owner(market(names=5),mod.Parameters(True));p=owner.inner
        calls=[];original=p._allocation_order
        def record(i,indices):
            result=original(i,indices);calls.append((i,tuple(indices),tuple(result)));return result
        p._allocation_order=record
        result=run(owner.market,policy_factory=lambda m,c:owner)
        self.assertTrue(calls);self.assertGreater(len(calls),len(set(i for i,_,_ in calls)))
        self.assertTrue(owner.trace);self.assertGreaterEqual(float(result.equity.cash.min()),-1e-7)
        self.assertTrue(any(x.get('kind')=='INFORMATION_CONTINUITY_PRIORITY' for x in owner.trace))

    def test_saved_trace_and_transitive_sources(self):
        mod=self.module()
        from research.finite_study import Study
        study=Study('continuity_selection')
        for name in ('continuity_selection.py','continuity_selection_contract.json','opportunity_allocation_study.py','support_budget.py','admission_budget_completion.py','decision_review.py'):
            self.assertIn(name,study.identity()['dependencies'])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'runs'/'pair';a=study.saved(market(),path,mod.Parameters(True));b=study.saved(market(),path,mod.Parameters(True))
            pd.testing.assert_frame_equal(a.equity,b.equity,check_dtype=False,check_exact=True)
            self.assertTrue((Path(tmp)/'intents/pair.json').is_file())


if __name__=='__main__':unittest.main()

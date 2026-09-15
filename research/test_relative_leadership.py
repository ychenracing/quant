"""The new objective has priority authority, never execution/risk authority."""
from dataclasses import asdict
import importlib.util
import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from techquant.config import Config
from techquant.data import Market
from techquant.engine import run
from techquant.policy import CloseObservation
from research.test_allocation_auction import market
from research.observed_admission_completion import Owner as Parent, Parameters as ParentParameters


class RelativeLeadershipTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research.relative_leadership'))
        from research import relative_leadership
        return relative_leadership

    def test_only_registered_boolean(self):
        mod=self.module()
        self.assertEqual([p.learn_relative for p in mod.grid()], [False,True])
        for value in (0,1,'True',None):
            with self.assertRaises(ValueError): mod.Parameters(value)

    def test_neutral_adapter_is_exact_control(self):
        mod=self.module(); m=market(days=160,names=5)
        a=run(m,policy_factory=lambda m,c:Parent(m,ParentParameters(),config=c))
        b=run(m,policy_factory=lambda m,c:mod.Owner(m,mod.Parameters(False),config=c))
        pd.testing.assert_frame_equal(a.equity,b.equity,check_exact=True)
        pd.testing.assert_frame_equal(a.targets,b.targets,check_exact=True)
        self.assertEqual(a.orders,b.orders)

    def test_priority_never_changes_eligibility_or_issues_prediction_sell(self):
        mod=self.module(); owner=mod.Owner(market(days=160,names=3),mod.Parameters(True))
        p=owner.inner;i=50;p.features.score[i]=[3.,2.,1.];p.relative[i]=[-.3,-.2,-.1]
        before=p.features.score.copy(); entries=p.features.entry.copy(); exits=p.features.exit.copy()
        self.assertEqual(p._allocation_order(i,np.array([0,1,2])),[2,1,0])
        np.testing.assert_array_equal(before,p.features.score)
        np.testing.assert_array_equal(entries,p.features.entry)
        np.testing.assert_array_equal(exits,p.features.exit)
        self.assertEqual(p._allocation_order(i,np.array([0,2])),[2,0])

    def test_pending_exit_and_partial_fill_are_not_revoked(self):
        mod=self.module();m=market(days=160,names=5);owner=mod.Owner(m,mod.Parameters(True));p=owner.inner
        p.features.entry[:]=True;p.features.exit[:]=False;p.ready[:]=True;p.risk.cap=1.
        price=p.features.close[60]; units=np.zeros(5);units[:2]=.2*2e6/price[:2]
        p.previous_units=units.copy();p.stop[0]=1e6
        for i,amount in ((60,units[0]),(61,units[0]/2)):
            actual=units.copy();actual[0]=amount;px=p.features.close[i]
            o=CloseObservation.from_inventory(i,str(m.calendar[i].date()),2e6,float(2e6-actual@px),actual,actual*px/2e6)
            decision=owner.decide(o)
            self.assertEqual(decision.unit_targets[0],0)
            self.assertTrue(np.all(decision.unit_targets<=actual+1e-10))
        self.assertEqual(p.params.positions,2);self.assertEqual(p.params.risk_budget,.10)

    def test_real_engine_prefix_exact_and_cash_funded(self):
        mod=self.module();m=market(days=170,names=5);date=m.calendar[125]
        factory=lambda m,c:mod.Owner(m,mod.Parameters(True),config=c)
        a=run(m,policy_factory=factory,delay=2)
        b=run(m.prefix(date),policy_factory=factory,delay=2)
        pd.testing.assert_frame_equal(a.equity.loc[:date],b.equity,check_exact=True)
        pd.testing.assert_frame_equal(a.targets.loc[:date],b.targets,check_exact=True)
        self.assertEqual([o for o in a.orders if o['date']<=str(date.date())],b.orders)
        self.assertGreaterEqual(float(a.equity.cash.min()),-1e-7)

    def test_removed_name_is_absent_from_training_and_results(self):
        mod=self.module();m=market(days=160,names=5);keep=m.symbols[:3]
        frames={s:f.copy() for s,f in m.frames.items()}
        for col in ('open','high','low','close','raw_open','raw_close'):
            frames[m.symbols[-1]][col]*=3
        changed=Market.from_frames(frames,m.calendar,sectors=m.sectors,quality='synthetic')
        factory=lambda m,c:mod.Owner(m,mod.Parameters(True),config=c)
        a=run(m.subset(keep),policy_factory=factory);b=run(changed.subset(keep),policy_factory=factory)
        pd.testing.assert_frame_equal(a.equity,b.equity,check_exact=True);self.assertEqual(a.orders,b.orders)

    def test_source_bound_saved_outputs_cannot_be_relabelled(self):
        from research.relative_leadership_study import Study
        mod=self.module();study=Study('relative_leadership')
        identity=study.identity()
        for name in ('relative_rank.py','relative_leadership.py','relative_leadership_study.py','relative_leadership_contract.json'):
            self.assertIn(name,identity['relative_dependencies'])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'runs'/'test'
            a=study.saved(market(days=150,names=3),path,mod.Parameters(True))
            b=study.saved(market(days=150,names=3),path,mod.Parameters(True))
            pd.testing.assert_frame_equal(a.equity,b.equity,check_dtype=False,check_exact=True)
            self.assertTrue((path.parent.parent/'intents/test.json').is_file())
            self.assertTrue((path.parent.parent/'rankings/test.npz').is_file())
            with self.assertRaises(ValueError):
                study.saved(market(days=150,names=3),path,mod.Parameters(False))


if __name__=='__main__':unittest.main()

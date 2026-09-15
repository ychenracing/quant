"""Joint sizing cannot mutate eligibility, promised protection or the cash ledger."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from techquant.config import Config
from techquant.engine import run
from techquant.data import Market
from techquant.policy import CloseObservation
from research.test_allocation_auction import market
from research.observed_admission_completion import Owner as Parent,Parameters as ParentParameters
from research.joint_funding import Owner,Parameters,grid


class JointFundingTests(unittest.TestCase):
    def test_disabled_control_is_exact_and_only_boolean_pair(self):
        self.assertEqual([p.joint for p in grid()],[False,True])
        for value in (0,1,None,'true'):
            with self.assertRaises(ValueError):Parameters(value)
        m=market(days=150,names=5)
        a=run(m,policy_factory=lambda m,c:Parent(m,ParentParameters(),config=c))
        b=run(m,policy_factory=lambda m,c:Owner(m,Parameters(False),config=c))
        pd.testing.assert_frame_equal(a.equity,b.equity,check_exact=True)
        pd.testing.assert_frame_equal(a.targets,b.targets,check_exact=True);self.assertEqual(a.orders,b.orders)

    def test_original_risk_cash_and_slots_bound_every_joint_request(self):
        m=market(days=180,names=5);owner=Owner(m,Parameters(True))
        result=run(m,policy_factory=lambda m,c:owner)
        records=[r for r in owner.trace if r.get('kind')=='JOINT_RESOURCE_FUNDING']
        self.assertTrue(records)
        for row in records:
            values=np.array(row['notional_additions']);held=np.array(row['actual_units'])>1e-10
            self.assertLessEqual(values.sum(),row['cash_authority']+1e-7)
            self.assertLessEqual(row['funded_stop_risk'],row['remaining_risk']+1e-7)
            self.assertLessEqual(np.count_nonzero((values>0)&~held),row['available_slots'])
            self.assertTrue(np.all(values[(values>0)&held]>=.08*row['nav']))
            self.assertTrue(np.all(values[(values>0)&~held]>=.01*row['nav']))
        self.assertGreaterEqual(float(result.equity.cash.min()),-1e-7)
        self.assertFalse(any(r.get('kind')=='ADMISSION_BUDGET_COMPLETION' for r in owner.trace))

    def test_pending_full_and_partial_protection_cannot_be_offset(self):
        m=market(days=100,names=5);owner=Owner(m,Parameters(True));p=owner.inner
        p.features.entry[:]=True;p.features.exit[:]=False;p.ready[:]=True;p.risk.cap=1.
        price=p.features.close[40];units=np.zeros(5);units[:2]=.2*2e6/price[:2]
        p.previous_units=units.copy();p.stop[0]=1e6
        for i,amount in ((40,units[0]),(41,units[0]/2)):
            actual=units.copy();actual[0]=amount;px=p.features.close[i]
            obs=CloseObservation.from_inventory(i,str(m.calendar[i].date()),2e6,float(2e6-actual@px),actual,actual*px/2e6)
            d=owner.decide(obs)
            self.assertEqual(d.unit_targets[0],0);self.assertTrue(np.all(d.unit_targets<=actual+1e-10))
        self.assertEqual(len(p.history),2)
        self.assertFalse(any(r.get('kind')=='JOINT_RESOURCE_FUNDING' for r in owner.trace))

    def test_unqualified_held_leg_never_adds_without_breakout(self):
        m=market(days=120,names=3);owner=Owner(m,Parameters(True));p=owner.inner;i=60
        price=p.features.close[i];units=np.zeros(3);units[0]=.1*2e6/price[0]
        p.stop=np.maximum(0.,price*.99);p.breakout[i]=False;p.features.score[i]=[3.,2.,1.]
        obs=CloseObservation.from_inventory(i,str(m.calendar[i].date()),2e6,float(2e6-units@price),units,units*price/2e6)
        d=p._allocate(obs,units.copy(),price,p.admission_stop(i),np.ones(3,dtype=bool),1.)
        self.assertEqual(d[0],units[0]);self.assertLessEqual(np.count_nonzero(d>0),2)
        self.assertTrue(np.all(p.pending_stop>=0))
        with self.assertRaises(AssertionError):p._allocate(obs,units*.5,price,p.admission_stop(i),np.ones(3,dtype=bool),1.)

    def test_prefix_removal_delayed_execution_and_single_name(self):
        m=market(days=180,names=5);date=m.calendar[129]
        factory=lambda m,c:Owner(m,Parameters(True),config=c)
        a=run(m,policy_factory=factory,delay=2,cost_multiplier=2.)
        b=run(m.prefix(date),policy_factory=factory,delay=2,cost_multiplier=2.)
        pd.testing.assert_frame_equal(a.equity.loc[:date],b.equity,check_exact=True)
        pd.testing.assert_frame_equal(a.targets.loc[:date],b.targets,check_exact=True)
        self.assertEqual([r for r in a.orders if r['date']<=str(date.date())],b.orders)
        selected=m.symbols[:3];base=run(m.subset(selected),policy_factory=factory)
        frames={s:f.copy() for s,f in m.frames.items()}
        for col in ('open','high','low','close','raw_open','raw_close'):frames[m.symbols[-1]][col]*=100
        changed=Market.from_frames(frames,m.calendar,sectors=m.sectors,quality='synthetic')
        altered=run(changed.subset(selected),policy_factory=factory)
        pd.testing.assert_frame_equal(base.equity,altered.equity,check_exact=True)
        single=run(m.subset(m.symbols[:1]),policy_factory=factory)
        self.assertGreaterEqual(float(single.equity.cash.min()),-1e-7)

    def test_saved_trace_has_bound_sources_and_is_reusable(self):
        from research.joint_funding_study import Study
        study=Study();m=market(days=100,names=5)
        for name in ('joint_resources.py','joint_funding.py','joint_funding_contract.json','decision_review.py'):
            self.assertIn(name,study.identity()['dependencies'])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'runs/joint'
            a=study.saved(m,path,Parameters(True));b=study.saved(m,path,Parameters(True))
            pd.testing.assert_frame_equal(a.equity,b.equity,check_dtype=False,check_exact=True)
            self.assertTrue((Path(tmp)/'intents/joint.json').is_file())


if __name__=='__main__':unittest.main()

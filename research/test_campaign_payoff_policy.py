"""Shadow labels cannot change cash, protection, causal timing or input identity."""
import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from techquant.config import Config
from techquant.data import Market
from techquant.engine import run
from techquant.policy import CloseObservation
from research.test_allocation_auction import market
from research.observed_admission_completion import Owner as Parent,Parameters as ParentParameters
from research.campaign_payoff_policy import Owner,Parameters,grid,control_account,settled_samples


class CampaignPolicyTests(unittest.TestCase):
    def test_boolean_pair_and_disabled_control_are_exact(self):
        self.assertEqual([p.learn_payoff for p in grid()],[False,True])
        for bad in (1,0,None,'yes'):
            with self.assertRaises(ValueError):Parameters(bad)
        m=market(days=140,names=5)
        old=control_account(m,Config())
        new=run(m,policy_factory=lambda m,c:Owner(m,Parameters(False),config=c))
        pd.testing.assert_frame_equal(old.equity,new.equity,check_exact=True)
        pd.testing.assert_frame_equal(old.targets,new.targets,check_exact=True)
        self.assertEqual(old.orders,new.orders)

    def test_settlement_samples_are_actual_closed_episodes(self):
        m=market(days=180,names=5);cfg=Config();reference=control_account(m,cfg)
        records,meta=settled_samples(m,reference,cfg)
        from research.ledger_attribution import attribute
        _,episodes,_=attribute(m,reference)
        self.assertEqual(meta['censored_open'],sum(episodes.exit=='OPEN') if len(episodes) else 0)
        self.assertEqual(len(records),sum(episodes.exit!='OPEN') if len(episodes) else 0)
        self.assertTrue(all(r.formation+1<r.settled for r in records))
        owner=Owner(m,Parameters(True),reference=reference)
        self.assertTrue(all(f['latest_settlement_session']<=f['session'] for f in owner.fits))

    def test_reference_configuration_source_policy_cost_and_delay_are_bound(self):
        m=market(days=110,names=5);cfg=Config();reference=control_account(m,cfg)
        for key,value in (('delay',2),('cost_multiplier',2.),('source',{}),('policy',{}),
                          ('data_sha256','wrong'),('universe',[]),('config',{})):
            changed=copy.deepcopy(reference);changed.metadata[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                Owner(m,Parameters(True),reference=changed)
        stress=control_account(m,cfg,costs=2.,delay=2)
        with self.assertRaises(ValueError):Owner(m,Parameters(True),reference=stress)
        Owner(m,Parameters(True),reference=stress,costs=2.,delay=2)
        with self.assertRaises(ValueError):Owner(m,Parameters(False),reference=reference)

    def test_rank_can_only_reorder_the_original_eligible_set(self):
        owner=Owner(market(days=110,names=3),Parameters(True));p=owner.inner;i=60
        p.features.score[i]=[3.,2.,1.];p.payoffs[i]=[-1.,10.,0.]
        entry=p.features.entry.copy();exits=p.features.exit.copy();scores=p.features.score.copy()
        self.assertEqual(p._allocation_order(i,[0,1,2]),[1,2,0])
        self.assertEqual(p._allocation_order(i,[0,2]),[2,0])
        np.testing.assert_array_equal(p.features.entry,entry);np.testing.assert_array_equal(p.features.exit,exits)
        np.testing.assert_array_equal(p.features.score,scores)

    def test_pending_exit_and_actual_risk_history_survive_partial_fill(self):
        m=market(days=110,names=5);owner=Owner(m,Parameters(True));p=owner.inner
        p.features.entry[:]=True;p.features.exit[:]=False;p.ready[:]=True;p.risk.cap=1.
        px=p.features.close[40];units=np.zeros(5);units[:2]=.2*2e6/px[:2]
        p.previous_units=units.copy();p.stop[0]=1e6
        for i,amount in ((40,units[0]),(41,units[0]/2)):
            actual=units.copy();actual[0]=amount;px=p.features.close[i]
            o=CloseObservation.from_inventory(i,str(m.calendar[i].date()),2e6,
                        float(2e6-actual@px),actual,actual*px/2e6)
            d=owner.decide(o)
            self.assertEqual(d.unit_targets[0],0.)
            self.assertTrue(np.all(d.unit_targets<=actual+1e-10))
        self.assertEqual(len(p.history),2);self.assertEqual(p.params.positions,2)
        self.assertEqual(p.params.risk_budget,.10)

    def test_prefix_removed_input_and_delayed_execution(self):
        m=market(days=180,names=5);cfg=Config();date=m.calendar[139]
        factory=lambda m,c:Owner(m,Parameters(True),config=c,costs=2.,delay=2)
        full=run(m,cfg,policy_factory=factory,cost_multiplier=2.,delay=2)
        prefix=run(m.prefix(date),cfg,policy_factory=factory,cost_multiplier=2.,delay=2)
        pd.testing.assert_frame_equal(full.equity.loc[:date],prefix.equity,check_exact=True)
        pd.testing.assert_frame_equal(full.targets.loc[:date],prefix.targets,check_exact=True)
        self.assertEqual([o for o in full.orders if o['date']<=str(date.date())],prefix.orders)
        self.assertGreaterEqual(float(full.equity.cash.min()),-1e-7)
        subset=m.symbols[:3];a=run(m.subset(subset),cfg,policy_factory=factory,cost_multiplier=2.,delay=2)
        frames={s:f.copy() for s,f in m.frames.items()}
        for col in ('open','high','low','close','raw_open','raw_close'):frames[m.symbols[-1]][col]*=100
        altered=Market.from_frames(frames,m.calendar,sectors=m.sectors,quality='synthetic')
        b=run(altered.subset(subset),cfg,policy_factory=factory,cost_multiplier=2.,delay=2)
        pd.testing.assert_frame_equal(a.equity,b.equity,check_exact=True);self.assertEqual(a.orders,b.orders)

    def test_study_reuses_control_and_preserves_all_learning_evidence(self):
        from research.campaign_payoff_study import Study
        study=Study();m=market(days=120,names=5)
        for key in ('campaign_payoff.py','campaign_payoff_policy.py','campaign_payoff_study.py',
                    'relative_rank.py','ledger_attribution.py'):
            self.assertIn(key,study.identity()['dependencies'])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);reference=study.saved(m,root/'runs/control',Parameters(False))
            with patch('research.campaign_payoff_policy.control_account',side_effect=AssertionError('duplicate shadow')):
                a=study.saved(m,root/'runs/treatment',Parameters(True),reference=reference)
                b=study.saved(m,root/'runs/treatment',Parameters(True),reference=reference)
            pd.testing.assert_frame_equal(a.equity,b.equity,check_dtype=False,check_exact=True)
            self.assertTrue((root/'payoffs/treatment.json').is_file())
            self.assertTrue((root/'intents/treatment.json').is_file())
            with self.assertRaises(ValueError):study.saved(m,root/'runs/unbound',Parameters(True))


if __name__=='__main__':unittest.main()

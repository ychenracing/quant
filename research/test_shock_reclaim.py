"""A shock-exited campaign may only get one-close provisional rank-1 reclaim permission."""
from dataclasses import asdict, replace
import importlib, importlib.util, unittest
import numpy as np
import pandas as pd
from test_core import sample_market
from techquant.engine import run
from research.observed_admission_completion import Owner as Parent, Parameters as ParentParameters
from research.test_quantity_obligation import observe, control

class ShockReclaimTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research.shock_reclaim'),
                             'preregistered shock reclaim implementation is absent')
        return importlib.import_module('research.shock_reclaim')

    def make(self,n=2,enabled=True):
        mod=self.module(); owner=mod.Owner(sample_market(n,120),mod.Parameters(enabled)); p=owner.inner
        control(p,1.); p.features.close[:]=10.; p.features.entry[:]=False; p.features.exit[:]=False
        p.features.score[:]=1.; p.ready[:]=True; p.atr[:]=1.; p.support[:]=0.; p.stop[:]=0.; p.peak[:]=10.
        return owner

    def arm_flat(self,owner,j=0,reference=9.5):
        p=owner.inner; owner.reclaim_reference[j]=reference; owner.reclaim_pending[j]=False; owner.reclaim_flat[j]=True
        p.readmit[j]=True; p.previous_units[j]=0.; p.exit_pending[j]=False; p.reduction_ceiling[j]=np.inf

    def test_registered_pair_and_control_exact_parent(self):
        mod=self.module(); self.assertEqual([asdict(x) for x in mod.grid()],[{'enabled':False},{'enabled':True}])
        market=sample_market(2,80)
        a=mod.Owner(market,mod.Parameters(False)); b=Parent(market,ParentParameters())
        for i in range(40):
            o=observe(a.inner,i,[0.,0.]); x=a.decide(o); y=b.decide(o)
            np.testing.assert_array_equal(x.unit_targets,y.unit_targets); self.assertEqual(x.reason,y.reason)

    def test_rank1_reclaim_is_provisional_and_parent_funded(self):
        owner=self.make(); p=owner.inner; self.arm_flat(owner)
        p.features.entry[40]=[True,True]; p.features.score[40]=[3.,2.]
        d=owner.decide(observe(p,40,[0.,0.],cash=2_000_000.))
        self.assertGreater(d.unit_targets[0],0.)
        self.assertTrue(p.readmit[0], 'permission must remain provisional until an actual fill is observed')
        self.assertIn('SHOCK_RECLAIM',d.reason)

    def test_rank2_or_unreclaimed_price_cannot_release(self):
        owner=self.make(); p=owner.inner; self.arm_flat(owner,reference=9.5)
        p.features.entry[40]=[True,True]; p.features.score[40]=[2.,3.]
        d=owner.decide(observe(p,40,[0.,0.])); self.assertEqual(d.unit_targets[0],0.); self.assertTrue(p.readmit[0])
        owner=self.make(); p=owner.inner; self.arm_flat(owner,reference=10.5)
        p.features.entry[40]=[True,False]; p.features.score[40]=[3.,1.]
        d=owner.decide(observe(p,40,[0.,0.])); self.assertEqual(d.unit_targets[0],0.); self.assertTrue(p.readmit[0])

    def test_actual_reacquisition_consumes_memory_and_veto(self):
        owner=self.make(); p=owner.inner; self.arm_flat(owner)
        p.features.entry[40]=[True,False]; p.features.score[40]=[3.,1.]
        owner.decide(observe(p,40,[0.,0.]))
        d=owner.decide(observe(p,41,[5000.,0.]))
        self.assertFalse(owner.reclaim_flat[0]); self.assertFalse(p.readmit[0]); self.assertGreaterEqual(d.unit_targets[0],0.)

    def test_natural_parent_release_expires_special_memory(self):
        owner=self.make(); p=owner.inner; self.arm_flat(owner)
        p.readmit[0]=False
        owner.decide(observe(p,40,[0.,0.]))
        self.assertFalse(owner.reclaim_flat[0])

    def test_capture_requires_pure_market_shock_and_actual_flat(self):
        owner=self.make(); p=owner.inner
        # use the module's capture hook directly so this test does not alter parent risk arithmetic
        owner._remember_shock_exit(40,np.array([10000.,0.]),np.array([0.,0.]),'CROSS_SECTION_SHOCK',False,False)
        self.assertTrue(owner.reclaim_pending[0]); self.assertFalse(owner.reclaim_flat[0])
        owner._observe_reclaim_inventory(np.array([5000.,0.])); self.assertTrue(owner.reclaim_pending[0]); self.assertFalse(owner.reclaim_flat[0])
        owner._observe_reclaim_inventory(np.array([0.,0.])); self.assertFalse(owner.reclaim_pending[0]); self.assertTrue(owner.reclaim_flat[0])
        other=self.make(); other._remember_shock_exit(40,np.array([10000.,0.]),np.array([0.,0.]),'PORTFOLIO_DRAWDOWN_SHOCK',True,False)
        self.assertFalse(other.reclaim_pending.any())

    def test_prefix_and_delayed_execution_remain_causal(self):
        mod=self.module(); market=sample_market(3,120); factory=lambda m,c:mod.Owner(m,mod.Parameters(True))
        full=run(market,policy_factory=factory,delay=2); cut=market.calendar[85]
        short=run(market.prefix(cut),policy_factory=factory,delay=2)
        pd.testing.assert_frame_equal(full.equity.loc[:cut],short.equity)
        pd.testing.assert_frame_equal(full.targets.loc[:cut],short.targets)
        self.assertEqual([o for o in full.orders if o['date']<=str(cut.date())],short.orders)

if __name__=='__main__': unittest.main()

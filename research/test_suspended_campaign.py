"""A completed protective sale is never treated as an unfunded restoration."""
from dataclasses import asdict
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from test_core import sample_market
from techquant.data import Market
from techquant.engine import run
from research.observed_admission_completion import Owner as Parent, Parameters as ParentParameters
from research.test_quantity_obligation import observe, control


class SuspendedCampaignTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research.suspended_campaign'),
                             'registered suspended campaign owner is absent')
        return importlib.import_module('research.suspended_campaign')

    def make(self,n=2):
        mod=self.module();owner=mod.Owner(sample_market(n,120),mod.Parameters());p=owner.inner
        control(p);p.features.close[:]=10.;p.features.entry[:]=False;p.features.exit[:]=False
        p.features.score[:]=1.;p.ready[:]=True;p.atr[:]=1.;p.support[:]=0.;owner.fast[:]=9.
        p.stop[:]=9.;p.peak[:]=10.
        return owner

    def suspend(self,owner,units=None):
        p=owner.inner
        if units is None:units=np.array([10000.]+[0.]*(len(owner.market.symbols)-1))
        p.previous_units=np.array(units,dtype=float);control(p,0.)
        d=owner.decide(observe(p,40,units));self.assertTrue((d.unit_targets==0).all())
        return d

    def test_singleton_exact_identity_and_no_extra_parameters(self):
        mod=self.module();self.assertEqual([asdict(x) for x in mod.grid()],[{}])
        with self.assertRaises(ValueError):mod.Owner(sample_market(),ParentParameters())
        with self.assertRaises(TypeError):mod.Parameters(recovery=1)
        from research.finite_study import Study
        identity=Study('suspended_campaign').identity()
        for name in ('observed_admission_completion.py','quantity_obligation.py','support_budget.py','suspended_campaign_contract.json'):
            self.assertIn(name,identity['dependencies'])

    def test_only_unbroken_actual_zero_cap_exit_creates_suspension(self):
        owner=self.make();p=owner.inner;self.suspend(owner)
        np.testing.assert_array_equal(owner.suspended_units,[10000.,0.]);self.assertEqual(owner.stop_floor[0],9.)
        for invalid in ('price','stale','pending','partial'):
            other=self.make();q=other.inner;q.previous_units[:]=[10000.,0.];control(q,0.)
            if invalid=='price':q.features.exit[40,0]=True
            if invalid=='stale':q.ready[40,0]=False
            if invalid=='pending':q.exit_pending[0]=True
            if invalid=='partial':control(q,.5)
            other.decide(observe(q,40,[10000.,0.],cash=1000.))
            self.assertFalse(other.suspended_units.any(),invalid)

    def test_blocked_partial_and_actual_flat_clock(self):
        owner=self.make();p=owner.inner;self.suspend(owner);control(p,1.)
        for i,units in ((41,[10000.,0.]),(42,[5000.,0.])):
            d=owner.decide(observe(p,i,units));self.assertEqual(d.unit_targets[0],0.)
            self.assertTrue(p.exit_pending[0]);self.assertEqual(owner.flat_health[0],0)
        for i in (43,44):
            d=owner.decide(observe(p,i,[0.,0.]));self.assertEqual(d.unit_targets[0],0.)
        d=owner.decide(observe(p,45,[0.,0.]));self.assertEqual(d.unit_targets[0],10000.)
        self.assertEqual(owner.suspended_units[0],10000.);self.assertTrue(p.readmit[0])
        self.assertGreaterEqual(p.pending_stop[0],9.);self.assertEqual(len(p.history),6)

    def test_present_health_and_old_floor_invalidate_without_canceling_sales(self):
        for invalid in ('price','stale','exit'):
            owner=self.make();p=owner.inner;self.suspend(owner);control(p,1.)
            if invalid=='price':p.features.close[41,0]=8.9
            if invalid=='stale':p.ready[41,0]=False
            if invalid=='exit':p.features.exit[41,0]=True
            d=owner.decide(observe(p,41,[5000.,0.]));self.assertEqual(d.unit_targets[0],0.)
            self.assertTrue(p.exit_pending[0]);self.assertEqual(owner.suspended_units[0],0.)
        owner=self.make();p=owner.inner;self.suspend(owner);control(p,1.)
        owner.decide(observe(p,41,[0.,0.]));owner.decide(observe(p,42,[0.,0.]))
        owner.fast[43,0]=11.;d=owner.decide(observe(p,43,[0.,0.]))
        self.assertEqual(d.unit_targets[0],0.);self.assertEqual(owner.flat_health[0],0)

    def test_actual_acquisition_consumes_only_opportunity_preserving_stop(self):
        owner=self.make();p=owner.inner;self.suspend(owner);control(p,1.)
        for i in (41,42,43):d=owner.decide(observe(p,i,[0.,0.]))
        self.assertGreater(d.unit_targets[0],0.);self.assertGreater(owner.suspended_units[0],0.)
        d=owner.decide(observe(p,44,[5000.,0.]))
        self.assertEqual(owner.suspended_units[0],0.);self.assertFalse(p.readmit[0])
        self.assertGreaterEqual(p.stop[0],9.);self.assertEqual(d.unit_targets[0],5000.)
        self.assertEqual(len(p.history),5)

    def test_real_cash_risk_cap_and_quantity_bound(self):
        owner=self.make();p=owner.inner;self.suspend(owner);control(p,.5)
        for i in (41,42,43):d=owner.decide(observe(p,i,[0.,0.],cash=100000.))
        self.assertLessEqual(d.unit_targets[0],10000.);self.assertLessEqual(d.unit_targets[0]*10.,50000.)
        self.assertLessEqual(d.unit_targets[0]*(10.-p.pending_stop[0]),p.params.risk_budget*100000.*.5/2+1e-8)
        zero=self.make();q=zero.inner;self.suspend(zero)
        for i in (41,42,43):d=zero.decide(observe(q,i,[0.,0.]))
        self.assertFalse(d.unit_targets.any())

    def test_parent_purchases_and_full_book_are_reserved(self):
        owner=self.make(3);p=owner.inner;self.suspend(owner)
        control(p,1.);p.features.entry[43:,1:]=True;p.features.score[:,1:]=[3.,2.]
        for i in (41,42,43):d=owner.decide(observe(p,i,[0.,0.,0.]))
        self.assertGreater(d.unit_targets[1],0.);self.assertGreater(d.unit_targets[2],0.)
        self.assertEqual(d.unit_targets[0],0.)
        owner=self.make();p=owner.inner;self.suspend(owner);control(p,1.)
        p.features.entry[43:,1]=True
        for i in (41,42,43):d=owner.decide(observe(p,i,[0.,0.],cash=100000.))
        self.assertLessEqual(float(d.unit_targets@np.array([10.,10.])),99000.+1e-8)
        self.assertLessEqual(float((d.unit_targets*np.array([10.,10.])).max()),55000.+1e-8)

    def test_duplicate_observation_leaves_all_state_unchanged(self):
        owner=self.make();p=owner.inner;self.suspend(owner)
        state=(owner.suspended_units.copy(),owner.stop_floor.copy(),owner.flat_health.copy(),p.readmit.copy(),list(p.history),list(owner.trace))
        with self.assertRaises(ValueError):owner.decide(observe(p,40,[0.,0.]))
        for a,b in zip(state[:4],(owner.suspended_units,owner.stop_floor,owner.flat_health,p.readmit)):np.testing.assert_array_equal(a,b)
        self.assertEqual(state[4],p.history);self.assertEqual(state[5],owner.trace)

    def test_no_suspension_preserves_parent_decisions(self):
        mod=self.module();m=sample_market(2,90);a=mod.Owner(m,mod.Parameters());b=Parent(m,ParentParameters())
        for i in range(50):
            o=observe(a.inner,i,[0.,0.]);x=a.decide(o);y=b.decide(o)
            np.testing.assert_array_equal(x.unit_targets,y.unit_targets);self.assertEqual(x.reason,y.reason)
            self.assertEqual(a.inner.history,b.inner.history)

    def test_prefix_removal_delayed_cash_and_source_bound_trace(self):
        mod=self.module();m=sample_market(3,120);factory=lambda market,c:mod.Owner(market,mod.Parameters())
        full=run(m,policy_factory=factory,delay=2);cut=m.calendar[85];short=run(m.prefix(cut),policy_factory=factory,delay=2)
        pd.testing.assert_frame_equal(full.equity.loc[:cut],short.equity);pd.testing.assert_frame_equal(full.targets.loc[:cut],short.targets)
        self.assertEqual([o for o in full.orders if o['date']<=str(cut.date())],short.orders)
        changed={s:f.copy() for s,f in m.frames.items()};changed[m.symbols[-1]].loc[:,['open','high','low','close','raw_open','raw_close']]*=10.
        other=Market.from_frames(changed,m.calendar,quality='synthetic')
        x=run(m.subset(m.symbols[:-1]),policy_factory=factory);y=run(other.subset(m.symbols[:-1]),policy_factory=factory)
        pd.testing.assert_frame_equal(x.equity,y.equity);self.assertEqual(x.orders,y.orders)
        from research.ledger_attribution import attribute
        self.assertLess(attribute(m,full)[2]['max_reconciliation_error'],1e-6)
        from research.finite_study import Study
        study=Study('suspended_campaign')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'runs'/'owned';study.saved(m,path,mod.Parameters());study.saved(m,path,mod.Parameters())
            trace=Path(tmp)/'intents'/'owned.json';data=json.loads(trace.read_text());data['trace'].append({'corrupted':True});trace.write_text(json.dumps(data))
            with self.assertRaises(ValueError):study.saved(m,path,mod.Parameters())

if __name__=='__main__':unittest.main()

"""Only the registered held-addition timing interlock may change."""
from importlib import import_module, util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from techquant.engine import run
from techquant.policy import CloseObservation
from research.test_allocation_auction import market
from research.observed_admission_completion import Owner as Parent, Parameters as ParentParameters


class HealthyHoldFundingTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(util.find_spec('research.healthy_hold_funding'), 'registered implementation is absent')
        return import_module('research.healthy_hold_funding')

    def test_only_two_boolean_candidates(self):
        mod = self.module()
        self.assertEqual([p.continuous_held_funding for p in mod.grid()],[False,True])
        for bad in (0,1,None,'true'):
            with self.assertRaises(ValueError): mod.Parameters(bad)

    def test_extra_held_breakout_is_the_only_changed_mask(self):
        mod = self.module(); m=market(); a=mod.Owner(m,mod.Parameters(False));b=mod.Owner(m,mod.Parameters(True))
        i=40;held=np.array([True,False,True]);allowed=np.array([True,True,False]);price=np.array([10.,10.,10.]);stops=np.array([9.,9.,9.])
        units=held.astype(float)
        o=CloseObservation.from_inventory(i,str(m.calendar[i].date()),2e6,2e6-20,units,units*price/2e6)
        for owner in (a,b): owner.inner.breakout[i]=False
        np.testing.assert_array_equal(a.inner._funding_eligible(o,held,allowed,price,stops),[False,True,False])
        np.testing.assert_array_equal(b.inner._funding_eligible(o,held,allowed,price,stops),[True,True,False])
        stops[0]=10.
        np.testing.assert_array_equal(b.inner._funding_eligible(o,held,allowed,price,stops),[False,True,False])

    def test_default_is_exact_existing_parent_account(self):
        mod=self.module();m=market(days=120)
        a=run(m,policy_factory=lambda m,c:Parent(m,ParentParameters(),config=c))
        b=run(m,policy_factory=lambda m,c:mod.Owner(m,mod.Parameters(False),config=c))
        pd.testing.assert_frame_equal(a.equity,b.equity);pd.testing.assert_frame_equal(a.targets,b.targets)
        self.assertEqual(a.orders,b.orders)

    def test_actual_addition_still_obeys_cash_risk_and_name_limits(self):
        mod=self.module();m=market(names=1);i=40
        owner=mod.Owner(m,mod.Parameters(True));p=owner.inner;price=p.features.close[i]
        units=np.array([10000.]);nav=2e6;cash=float(nav-units@price)
        o=CloseObservation.from_inventory(i,str(m.calendar[i].date()),nav,cash,units,units*price/nav)
        p.stop[:]=.95*price;p.breakout[i]=False
        wanted=p._allocate(o,units.copy(),price,.95*price,np.ones(1,bool),1.)
        self.assertGreater(wanted[0],units[0])
        self.assertLessEqual(float((wanted-units)@price),.99*cash+1e-7)
        self.assertLessEqual(float(wanted@(price-p.stop)),p._risk_limit(o,1.)+1e-7)
        self.assertLessEqual(float(wanted@price),nav+1e-7)
        self.assertGreaterEqual(p.pending_stop[0],p.stop[0])

    def test_partial_or_blocked_protection_cannot_be_topped_up(self):
        mod=self.module();m=market();owner=mod.Owner(m,mod.Parameters(True));p=owner.inner
        p.previous_units=np.array([1000.,0.,0.]);p.stop[0]=1e6;p.risk.cap=1.
        for i,q in ((40,1000.),(41,500.)):
            units=np.array([q,0.,0.]);price=p.features.close[i]
            o=CloseObservation.from_inventory(i,str(m.calendar[i].date()),2e6,float(2e6-units@price),units,units*price/2e6)
            d=owner.decide(o)
            self.assertEqual(d.unit_targets[0],0.)
            self.assertTrue(np.all(d.unit_targets<=units+1e-10))
            self.assertTrue(p.exit_pending[0])
            self.assertEqual(len(p.history),i-39)

    def test_prefix_invariance_and_real_cash_under_delayed_fills(self):
        mod=self.module();m=market(days=110);factory=lambda m,c:mod.Owner(m,mod.Parameters(True),config=c)
        full=run(m,policy_factory=factory,delay=2);cut=m.calendar[75]
        prefix=run(m.prefix(cut),policy_factory=factory,delay=2)
        pd.testing.assert_frame_equal(full.equity.loc[:cut],prefix.equity)
        pd.testing.assert_frame_equal(full.targets.loc[:cut],prefix.targets)
        self.assertEqual([o for o in full.orders if o['date']<=str(cut.date())],prefix.orders)
        self.assertGreaterEqual(float(full.equity.cash.min()),-1e-7)

    def test_existing_runner_preserves_transitive_identity_and_cache(self):
        mod=self.module()
        from research.finite_study import Study
        study=Study('healthy_hold_funding'); identity=study.identity()
        for name in ('healthy_hold_funding.py','healthy_hold_funding_contract.json','support_budget.py',
                     'quantity_obligation.py','observed_admission_completion.py','observed_readiness.py','admission_budget_completion.py'):
            self.assertIn(name,identity['dependencies'])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'runs'/'candidate';a=study.saved(market(),path,mod.Parameters(True))
            b=study.saved(market(),path,mod.Parameters(True))
            pd.testing.assert_frame_equal(a.equity,b.equity,check_dtype=False,check_exact=True)


if __name__ == '__main__': unittest.main()

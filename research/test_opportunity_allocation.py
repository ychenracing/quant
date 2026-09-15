"""Residual opportunity funding never changes an incumbent protection obligation."""
from importlib import import_module, util
from pathlib import Path
from dataclasses import replace
import tempfile
import unittest
import numpy as np
import pandas as pd
from techquant.engine import run
from techquant.policy import CloseObservation, CloseDecision
from techquant.data import Market
from research.test_allocation_auction import market
from research.observed_admission_completion import Owner as Parent, Parameters as ParentParameters


class OpportunityAllocationTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(util.find_spec('research.opportunity_allocation'), 'registered implementation is absent')
        return import_module('research.opportunity_allocation')

    def fixture(self, names=5):
        mod=self.module();m=market(names=names);owner=mod.Owner(m,mod.Parameters(True));p=owner.inner;i=40
        price=p.features.close[i];nav=2e6
        units=np.zeros(names);units[:2]=.15*nav/price[:2]
        p.stop[:]=.9*price;p.risk.cap=1.;p.ready[i]=True;p.features.entry[i]=True;p.features.exit[i]=False;p.features.score[i]=np.arange(names,0,-1)
        p.readmit[:]=False;p.pending_stop[:]=0.
        o=CloseObservation.from_inventory(i,str(m.calendar[i].date()),nav,float(nav-units@price),units,units*price/nav)
        d=CloseDecision(units*price/nav,'UNCHANGED_PARENT',1.,units.copy())
        return mod,m,owner,o,d,price

    def test_registered_boolean_only(self):
        mod=self.module();self.assertEqual([p.fund_residual_opportunities for p in mod.grid()],[False,True])
        for bad in (0,1,None,'true'):
            with self.assertRaises(ValueError):mod.Parameters(bad)

    def test_original_control_is_bit_exact(self):
        mod=self.module();m=market(days=120)
        a=run(m,policy_factory=lambda m,c:Parent(m,ParentParameters(),config=c))
        b=run(m,policy_factory=lambda m,c:mod.Owner(m,mod.Parameters(False),config=c))
        pd.testing.assert_frame_equal(a.equity,b.equity,check_exact=True)
        pd.testing.assert_frame_equal(a.targets,b.targets,check_exact=True);self.assertEqual(a.orders,b.orders)

    def test_two_existing_slots_do_not_erase_residual_authority(self):
        mod,m,owner,o,d,price=self.fixture()
        result=owner._fund_residual(o,d)
        self.assertGreater(np.count_nonzero(result.unit_targets),2)
        np.testing.assert_array_equal(result.unit_targets[:2],o.units[:2])
        self.assertTrue(owner.trace)
        self.assertGreaterEqual(owner.inner.pending_stop[2],owner.inner.admission_stop(o.session)[2])

    def test_parent_purchases_are_reserved_not_spent_twice(self):
        mod,m,owner,o,d,price=self.fixture()
        wanted=d.unit_targets.copy();wanted[0]+=.4*o.nav/price[0]
        parent=replace(d,unit_targets=wanted,weights=wanted*price/o.nav)
        result=owner._fund_residual(o,parent);p=owner.inner
        self.assertEqual(result.unit_targets[0],wanted[0])
        self.assertLessEqual(float((result.unit_targets-o.units)@price),.99*o.cash+1e-7)
        stops=np.where(o.units>1e-10,p.stop,np.maximum(p.admission_stop(o.session),p.pending_stop))
        self.assertLessEqual(float(result.unit_targets@np.maximum(price-stops,.02*price)),p._risk_limit(o,1.)+1e-7)
        self.assertLessEqual(float(result.unit_targets@price),o.nav+1e-7)
        self.assertTrue(np.all(result.weights<=.55+1e-12))

    def test_no_readmission_or_qualification_bypass(self):
        mod,m,owner,o,d,price=self.fixture()
        owner.inner.readmit[2]=True;owner.inner.features.entry[o.session,3]=False;owner.inner.features.exit[o.session,4]=True
        self.assertIs(owner._fund_residual(o,d),d)
        owner.inner.readmit[2]=False
        result=owner._fund_residual(o,d)
        self.assertGreater(result.unit_targets[2],0.)
        np.testing.assert_array_equal(result.unit_targets[3:],[0.,0.])

    def test_all_protective_authority_wins(self):
        mod,m,owner,o,d,price=self.fixture()
        p=owner.inner
        for blocked in ('cap','ceiling','exit','reduction'):
            p.risk.cap=1.;p.reduction_ceiling[:]=np.inf;p.exit_pending[:]=False;parent=d
            if blocked=='cap':p.risk.cap=0.
            elif blocked=='ceiling':p.reduction_ceiling[0]=o.units[0]*.5
            elif blocked=='exit':p.exit_pending[0]=True
            else:
                q=d.unit_targets.copy();q[0]=0.;parent=replace(d,unit_targets=q,weights=q*price/o.nav)
            self.assertIs(owner._fund_residual(o,parent),parent)
        p.risk.cap=1.;p.exit_pending[:]=False;p.reduction_ceiling[:]=np.inf
        p.previous_units=o.units.copy();p.stop[0]=1e6
        for i,qty in ((40,o.units[0]),(41,o.units[0]/2)):
            units=o.units.copy();units[0]=qty;px=p.features.close[i]
            obs=CloseObservation.from_inventory(i,str(m.calendar[i].date()),o.nav,float(o.nav-units@px),units,units*px/o.nav)
            result=owner.decide(obs)
            self.assertEqual(result.unit_targets[0],0.)
            self.assertTrue(np.all(result.unit_targets<=units+1e-10))
            self.assertEqual(len(p.history),i-39)

    def test_sector_cap_and_original_single_slot_path(self):
        mod,m,owner,o,d,price=self.fixture()
        owner.inner.features=replace(owner.inner.features,sectors=('a','a','a','b','b'))
        result=owner._fund_residual(o,d)
        self.assertLessEqual(float(result.weights[:3].sum()),owner.inner.config.sector_cap+1e-12)
        units=o.units.copy();units[1]=0.
        one=CloseObservation.from_inventory(o.session,o.date,o.nav,float(o.nav-units@price),units,units*price/o.nav)
        parent=replace(d,weights=one.weights,unit_targets=units)
        self.assertIs(owner._fund_residual(one,parent),parent)

    def test_actual_engine_delay_cash_prefix_and_removed_inputs(self):
        mod=self.module();m=market(days=105,names=5);factory=lambda m,c:mod.Owner(m,mod.Parameters(True),config=c)
        full=run(m,policy_factory=factory,delay=2);date=m.calendar[75]
        short=run(m.prefix(date),policy_factory=factory,delay=2)
        pd.testing.assert_frame_equal(full.equity.loc[:date],short.equity,check_exact=True)
        pd.testing.assert_frame_equal(full.targets.loc[:date],short.targets,check_exact=True)
        self.assertEqual([r for r in full.orders if r['date']<=str(date.date())],short.orders)
        self.assertGreaterEqual(float(full.equity.cash.min()),-1e-7)
        names=m.symbols[:3];a=run(m.subset(names),policy_factory=factory)
        frames={s:f.copy() for s,f in m.frames.items()}
        frames[m.symbols[-1]][['open','high','low','close','raw_open','raw_close']]*=4
        changed=Market.from_frames(frames,m.calendar,sectors=m.sectors,quality='synthetic')
        b=run(changed.subset(names),policy_factory=factory)
        pd.testing.assert_frame_equal(a.equity,b.equity,check_exact=True);self.assertEqual(a.orders,b.orders)

    def test_registered_runner_trace_and_transitive_identity(self):
        mod=self.module()
        from research.finite_study import Study
        study=Study('opportunity_allocation')
        for name in ('opportunity_allocation.py','opportunity_allocation_contract.json','support_budget.py','quantity_obligation.py','observed_readiness.py','admission_budget_completion.py','observed_admission_completion.py','decision_review.py','opportunity_allocation_study.py','ledger_attribution.py'):
            self.assertIn(name,study.identity()['dependencies'])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'runs'/'pair';a=study.saved(market(),path,mod.Parameters(True));b=study.saved(market(),path,mod.Parameters(True))
            pd.testing.assert_frame_equal(a.equity,b.equity,check_dtype=False,check_exact=True)
            self.assertTrue((Path(tmp)/'intents/pair.json').is_file())


if __name__=='__main__':unittest.main()

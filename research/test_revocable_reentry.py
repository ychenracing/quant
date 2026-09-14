"""A weaker health observation cannot become an unfilled permanent buy right."""
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from test_core import sample_market
from research.test_quantity_obligation import observe, control
from research.funded_reentry import Owner as Prior, Parameters as PriorParameters
from techquant.engine import run


class RevocableReentryTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research.revocable_reentry'),
                             'the registered revocable permission is absent')
        return importlib.import_module('research.revocable_reentry')

    def make(self):
        module=self.module();owner=module.Owner(sample_market(1,120),module.Parameters())
        p=owner.inner;control(p);p.features.close[:]=10.;p.features.entry[:]=False
        p.features.exit[:]=False;p.features.score[:]=1.;p.ready[:]=True
        p.atr[:]=1.;p.support[:]=0.;owner.fast[:]=9.;p.readmit[:]=True
        return owner

    def test_health_without_acquisition_expires_after_deterioration(self):
        owner=self.make();p=owner.inner
        for i in range(40,43):
            d=owner.decide(observe(p,i,[0.]));self.assertEqual(d.unit_targets[0],0.)
        self.assertTrue(p.readmit[0])
        p.ready[43,0]=False;owner.decide(observe(p,43,[0.]))
        p.features.entry[44,0]=True
        d=owner.decide(observe(p,44,[0.]));self.assertEqual(d.unit_targets[0],0.)
        self.assertTrue(p.readmit[0]);self.assertEqual(owner.health_closes[0],1)

    def test_fresh_health_allows_present_trigger_not_permanent_flat_permission(self):
        owner=self.make();p=owner.inner
        owner.decide(observe(p,40,[0.]));owner.decide(observe(p,41,[0.]))
        p.features.entry[42,0]=True;d=owner.decide(observe(p,42,[0.]))
        self.assertGreater(d.unit_targets[0],0.);self.assertTrue(p.readmit[0])
        p.ready[43,0]=False;d=owner.decide(observe(p,43,[0.]))
        self.assertEqual(d.unit_targets[0],0.);self.assertTrue(p.readmit[0])

    def test_actual_acquisition_ends_only_the_flat_admission_veto(self):
        owner=self.make();p=owner.inner
        owner.decide(observe(p,40,[0.]));owner.decide(observe(p,41,[0.]))
        p.features.entry[42,0]=True;owner.decide(observe(p,42,[0.]))
        d=owner.decide(observe(p,43,[100.]))
        self.assertFalse(p.readmit[0]);self.assertEqual(d.unit_targets[0],100.)
        self.assertEqual(len(p.history),4);self.assertGreater(p.stop[0],0.)
        self.assertTrue(owner.trace[-1]['revocable_reentry']['actual_open'][0])

    def test_original_stronger_confirmation_keeps_its_durable_behavior(self):
        owner=self.make();p=owner.inner;p.features.entry[40:43,0]=True
        for i in range(40,43):owner.decide(observe(p,i,[0.]))
        self.assertFalse(p.readmit[0]);p.ready[43,0]=False
        owner.decide(observe(p,43,[0.]));self.assertFalse(p.readmit[0])

    def test_protective_obligation_and_inventory_survive(self):
        owner=self.make();p=owner.inner;p.previous_units[:]=100.;p.exit_pending[:]=True
        owner.health_closes[:]=10
        d=owner.decide(observe(p,40,[50.]));self.assertEqual(d.unit_targets[0],0.)
        self.assertTrue(p.exit_pending[0]);self.assertTrue(p.readmit[0])
        self.assertEqual(p.reduction_ceiling[0],0.)

    def test_duplicate_rejected_before_actual_open_can_mutate_veto(self):
        owner=self.make();p=owner.inner;owner.decide(observe(p,40,[0.]));before=(p.readmit.copy(),owner.health_closes.copy(),list(p.history),list(owner.trace))
        with self.assertRaises(ValueError):owner.decide(observe(p,40,[100.]))
        np.testing.assert_array_equal(before[0],p.readmit);np.testing.assert_array_equal(before[1],owner.health_closes)
        self.assertEqual(before[2],p.history);self.assertEqual(before[3],owner.trace)

    def test_source_bound_singleton_and_transitive_cache(self):
        mod=self.module()
        from research.finite_study import Study
        self.assertEqual(len(mod.grid()),1)
        with self.assertRaises(ValueError):mod.Owner(sample_market(),PriorParameters())
        study=Study('revocable_reentry');identity=study.identity()
        for name in ['funded_reentry.py','funded_reentry_contract.json','observed_admission_completion.py','reentry_diagnosis.py']:
            self.assertIn(name,identity['dependencies'])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'runs'/'candidate';m=sample_market(1,70)
            result=study.saved(m,path,mod.Parameters());same=study.saved(m,path,mod.Parameters())
            pd.testing.assert_frame_equal(result.equity,same.equity,check_dtype=False)
            self.assertEqual(result.orders,same.orders)

    def test_prefix_and_delayed_fill_cash_accounting(self):
        mod=self.module();m=sample_market(3,120);factory=lambda market,c:mod.Owner(market,mod.Parameters())
        full=run(m,policy_factory=factory,delay=2);cut=m.calendar[85]
        prefix=run(m.prefix(cut),policy_factory=factory,delay=2)
        pd.testing.assert_frame_equal(full.equity.loc[:cut],prefix.equity)
        pd.testing.assert_frame_equal(full.targets.loc[:cut],prefix.targets)
        self.assertEqual([o for o in full.orders if o['date']<=str(cut.date())],prefix.orders)
        from research.ledger_attribution import attribute
        self.assertLess(attribute(m,full)[2]['max_reconciliation_error'],1e-6)
        self.assertTrue((full.equity.cash>=-1e-6).all())

if __name__=='__main__':unittest.main()

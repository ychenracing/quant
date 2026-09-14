"""Health releases a completed-campaign veto, never cash or protection authority."""
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


class FundedReentryTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research.funded_reentry'),
                             'preregistered funded reentry implementation is absent')
        return importlib.import_module('research.funded_reentry')

    def make(self, names=1):
        module=self.module();owner=module.Owner(sample_market(names,120),module.Parameters())
        p=owner.inner;control(p);p.features.close[:]=10.;p.features.entry[:]=False
        p.features.exit[:]=False;p.features.score[:]=1.;p.ready[:]=True
        p.atr[:]=1.;p.support[:]=0.;owner.fast[:]=9.
        return owner

    def test_singleton_exact_parent_and_transitive_identity(self):
        module=self.module();self.assertEqual([asdict(p) for p in module.grid()],[{}])
        with self.assertRaises(TypeError):module.Parameters(recovery=1)
        with self.assertRaises(ValueError):module.Owner(sample_market(),ParentParameters())
        from research.finite_study import Study
        identity=Study('funded_reentry').identity()
        for stem in ('funded_reentry','observed_admission_completion','observed_readiness','admission_budget_completion','quantity_obligation','support_budget','funded_risk'):
            self.assertIn(stem+'.py',identity['dependencies'])
            self.assertIn(stem+'_contract.json',identity['dependencies'])

    def test_health_never_substitutes_for_current_entry_trigger(self):
        owner=self.make();p=owner.inner;p.readmit[:]=True
        for i in range(40,43):
            result=owner.decide(observe(p,i,[0.]));self.assertEqual(result.unit_targets[0],0.)
        self.assertFalse(p.readmit[0]);self.assertEqual(owner.health_closes[0],3)
        p.features.entry[43,0]=True
        result=owner.decide(observe(p,43,[0.]));self.assertGreater(result.unit_targets[0],0.)
        self.assertEqual(len(p.history),4)

    def test_actual_liquidation_restarts_confirmation_not_requested_exit(self):
        owner=self.make();p=owner.inner;p.previous_units[:]=100.;p.exit_pending[:]=True
        owner.health_closes[:]=20;p.readmit[:]=True
        for i,units in ((40,[100.]),(41,[50.])):
            result=owner.decide(observe(p,i,units));self.assertEqual(result.unit_targets[0],0.)
            self.assertTrue(p.exit_pending[0]);self.assertTrue(p.readmit[0])
            self.assertFalse(owner.trace[-1]['funded_reentry']['released'][0])
        result=owner.decide(observe(p,42,[0.]));self.assertTrue(p.readmit[0])
        self.assertEqual(owner.health_closes[0],1)
        owner.decide(observe(p,43,[0.]));self.assertTrue(p.readmit[0])
        p.features.entry[44,0]=True
        result=owner.decide(observe(p,44,[0.]));self.assertFalse(p.readmit[0])
        self.assertGreater(result.unit_targets[0],0.)

    def test_unhealthy_quote_resets_counter_and_no_protective_release(self):
        owner=self.make(2);p=owner.inner;p.readmit[:]=True
        owner.decide(observe(p,40,[0.,0.]));owner.decide(observe(p,41,[0.,0.]))
        p.ready[42,0]=False;p.features.exit[42,1]=True
        owner.decide(observe(p,42,[0.,0.]));np.testing.assert_array_equal(owner.health_closes,[0,0])
        p.exit_pending[0]=True;owner.health_closes[:]=5
        owner.decide(observe(p,43,[0.,0.]))
        self.assertFalse(owner.trace[-1]['funded_reentry']['released'][0])

    def test_duplicate_observation_rejected_before_any_mutation(self):
        owner=self.make();p=owner.inner;o=observe(p,40,[0.]);owner.decide(o)
        before=(owner.health_closes.copy(),p.readmit.copy(),list(p.history),list(owner.trace))
        with self.assertRaises(ValueError):owner.decide(o)
        np.testing.assert_array_equal(owner.health_closes,before[0]);np.testing.assert_array_equal(p.readmit,before[1])
        self.assertEqual(p.history,before[2]);self.assertEqual(owner.trace,before[3])

    def test_parent_equivalence_when_no_veto_is_released(self):
        module=self.module();market=sample_market(2,120)
        owner=module.Owner(market,module.Parameters());parent=Parent(market,ParentParameters());inner=owner.inner
        for i in range(40):
            o=observe(inner,i,[0.,0.]);a,b=owner.decide(o),parent.decide(o)
            np.testing.assert_array_equal(a.unit_targets,b.unit_targets);self.assertEqual(a.reason,b.reason)
            self.assertEqual(inner.history,parent.inner.history);self.assertIs(owner.inner,inner)

    def test_prefix_removal_delayed_cash_and_inventory_reconciliation(self):
        module=self.module();market=sample_market(3,120);factory=lambda m,c:module.Owner(m,module.Parameters())
        full=run(market,policy_factory=factory,delay=2);cut=market.calendar[85]
        short=run(market.prefix(cut),policy_factory=factory,delay=2)
        pd.testing.assert_frame_equal(full.equity.loc[:cut],short.equity)
        pd.testing.assert_frame_equal(full.targets.loc[:cut],short.targets)
        self.assertEqual([o for o in full.orders if o['date']<=str(cut.date())],short.orders)
        frames={s:f.copy() for s,f in market.frames.items()};frames[market.symbols[-1]].loc[:,['open','high','low','close','raw_open','raw_close']]*=100
        changed=Market.from_frames(frames,market.calendar,quality='synthetic')
        a=run(market.subset(market.symbols[:-1]),policy_factory=factory)
        b=run(changed.subset(market.symbols[:-1]),policy_factory=factory)
        pd.testing.assert_frame_equal(a.equity,b.equity);self.assertEqual(a.orders,b.orders)
        self.assertTrue((full.equity.cash>=-1e-6).all());self.assertTrue((full.equity.exposure<=1+1e-9).all())
        from research.ledger_attribution import attribute
        self.assertLess(attribute(market,full)[2]['max_reconciliation_error'],1e-6)

    def test_immutable_source_bound_health_trace_cache(self):
        self.module()
        from research.finite_study import Study
        market=sample_market(1,70);study=Study('funded_reentry')
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'runs'/'candidate';study.saved(market,p,study.module.Parameters())
            same=study.saved(market,p,study.module.Parameters());self.assertIsNotNone(same)
            path=Path(temp)/'intents'/'candidate.json';data=json.loads(path.read_text())
            self.assertTrue(path.exists());path.write_text(json.dumps({'corrupted':data}))
            with self.assertRaises(ValueError):study.saved(market,p,study.module.Parameters())

if __name__=='__main__':unittest.main()

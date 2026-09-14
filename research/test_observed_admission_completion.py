"""Registered interaction: one initialized owner, no state or evidence splicing."""
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
from research.observed_readiness import Owner as Readiness, Parameters as ReadinessParameters
from research.admission_budget_completion import Owner as Completion, Parameters as CompletionParameters
from research.test_quantity_obligation import observe, control


class ObservedAdmissionCompletionTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research.observed_admission_completion'),
                             'preregistered interaction implementation is absent')
        return importlib.import_module('research.observed_admission_completion')

    def make(self, market):
        module = self.module()
        return module.Owner(market, module.Parameters())

    def test_singleton_and_transitive_component_identity(self):
        module = self.module()
        self.assertEqual([asdict(p) for p in module.grid()], [{}])
        with self.assertRaises(TypeError): module.Parameters(window=5)
        with self.assertRaises(ValueError): module.Owner(sample_market(), CompletionParameters())
        from research.finite_study import Study
        identity = Study('observed_admission_completion').identity()
        for stem in ('observed_readiness','admission_budget_completion','quantity_obligation','support_budget','funded_risk'):
            for suffix in ('.py','_contract.json'):
                self.assertIn(stem+suffix, identity['dependencies'])
        owner = self.make(sample_market())
        self.assertEqual(owner.identity()['name'], 'observed_admission_completion')
        self.assertEqual(owner.identity()['readiness_origin']['name'], 'expanding_observed_support_readiness')
        self.assertEqual(owner.identity()['completion_origin']['name'], 'admission_budget_completion')

    def test_estimators_exactly_reuse_observed_history_without_initial_range(self):
        market = sample_market(2, 100)
        owner = self.make(market)
        parent = Readiness(market, ReadinessParameters())
        full = Completion(market, CompletionParameters())
        for field in ('atr','support','ready'):
            np.testing.assert_array_equal(getattr(owner.inner,field), getattr(parent.inner,field))
            np.testing.assert_array_equal(getattr(owner.inner,field)[20:], getattr(full.inner,field)[20:])
        self.assertFalse(owner.inner.ready[:10].any())
        self.assertTrue(owner.inner.ready[10].all())
        self.assertFalse(full.inner.ready[10].any())
        frames = {s:f.copy() for s,f in market.frames.items()}
        frames[market.symbols[0]].iloc[12,frames[market.symbols[0]].columns.get_loc('volume')] = 0.
        missing = self.make(Market.from_frames(frames, market.calendar, quality='synthetic'))
        self.assertFalse(missing.inner.ready[12,0])

    def test_one_initialized_account_owner_and_one_risk_update_per_close(self):
        owner = self.make(sample_market(3,100)); inner = owner.inner
        for i in range(35):
            owner.decide(observe(inner,i,[0,0,0]))
            self.assertIs(owner.inner, inner)
            self.assertEqual(len(inner.history), i+1)
            self.assertEqual(inner.last_session, i)
        self.assertEqual(inner.history, [2_000_000.]*35)

    def test_joint_decisions_equal_completion_when_estimators_and_history_match(self):
        market = sample_market(3,100)
        owner = self.make(market); parent = Completion(market,CompletionParameters())
        for i in range(30,70):
            obs = observe(parent.inner,i,[0,0,0])
            a,b = owner.decide(obs), parent.decide(obs)
            np.testing.assert_array_equal(a.unit_targets,b.unit_targets)
            self.assertEqual(a.reason,b.reason)
            self.assertEqual(owner.trace,parent.trace)
            self.assertEqual(owner.inner.history,parent.inner.history)

    def test_fresh_completion_reserves_original_requests_and_preserves_membership(self):
        for selected in ((0,), (0,1)):
            market=sample_market(3,100)
            owner=self.make(market); parent=Readiness(market,ReadinessParameters())
            for policy in (owner,parent):
                p=policy.inner; control(p)
                p.features.close[:]=10.;p.features.entry[:]=False
                p.features.entry[:,list(selected)]=True;p.features.exit[:]=False
                p.features.score[:]=1.;p.ready[:]=True;p.atr[:]=1.;p.support[:]=0.
            obs=observe(parent.inner,40,[0,0,0]);original=parent.decide(obs);done=owner.decide(obs)
            self.assertEqual(list(done.unit_targets>0),list(original.unit_targets>0))
            self.assertTrue(np.all(done.unit_targets>=original.unit_targets))
            self.assertAlmostEqual(float(done.unit_targets@np.full(3,3.)),200_000.)
            self.assertLessEqual(float(done.unit_targets@np.full(3,10.)),.99*obs.cash)
            if len(selected)==2: np.testing.assert_array_equal(done.unit_targets,original.unit_targets)
            else: self.assertGreater(done.unit_targets[0],original.unit_targets[0])
            for field in ('stop','pending_stop','reduction_ceiling','previous_units'):
                np.testing.assert_array_equal(getattr(owner.inner,field),getattr(parent.inner,field))

    def test_protective_obligation_partial_retry_and_actual_completion_remain_parent_owned(self):
        market=sample_market(1,100); owner=self.make(market);parent=Readiness(market,ReadinessParameters())
        for policy in (owner,parent):
            control(policy.inner);policy.inner.reduction_ceiling[0]=999.95
        for i,units in ((40,[1000.05]),(41,[1000.05]),(42,[900.05001])):
            obs=observe(parent.inner,i,units);a,b=owner.decide(obs),parent.decide(obs)
            np.testing.assert_array_equal(a.unit_targets,b.unit_targets)
            np.testing.assert_array_equal(owner.inner.reduction_ceiling,parent.inner.reduction_ceiling)
            self.assertEqual(a.reason,b.reason);self.assertEqual(owner.trace,parent.trace)
        self.assertTrue(np.isinf(owner.inner.reduction_ceiling[0]))

    def test_prefix_excluded_symbol_and_delayed_actual_cash(self):
        module=self.module();market=sample_market(3,120)
        factory=lambda m,c:module.Owner(m,module.Parameters())
        full=run(market,policy_factory=factory,delay=2);cut=market.calendar[85]
        short=run(market.prefix(cut),policy_factory=factory,delay=2)
        pd.testing.assert_frame_equal(full.equity.loc[:cut],short.equity)
        pd.testing.assert_frame_equal(full.targets.loc[:cut],short.targets)
        self.assertEqual([o for o in full.orders if o['date']<=str(cut.date())],short.orders)
        changed={s:f.copy() for s,f in market.frames.items()}
        changed[market.symbols[-1]].loc[:,['open','high','low','close','raw_open','raw_close']]*=50
        other=Market.from_frames(changed,market.calendar,quality='synthetic')
        a=run(market.subset(market.symbols[:-1]),policy_factory=factory)
        b=run(other.subset(market.symbols[:-1]),policy_factory=factory)
        pd.testing.assert_frame_equal(a.equity,b.equity);self.assertEqual(a.orders,b.orders)
        self.assertTrue((full.equity.cash>=-1e-6).all())
        self.assertTrue((full.equity.exposure<=1+1e-9).all())
        for order in full.orders:
            if order['status']=='FILLED':
                self.assertGreater(order['date'],order['signal_date'])
        from research.ledger_attribution import attribute
        self.assertLess(attribute(market,full)[2]['max_reconciliation_error'],1e-6)

    def test_cached_component_identity_and_trace_cannot_be_spliced(self):
        module=self.module()
        from research.finite_study import Study
        study=Study('observed_admission_completion')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'runs'/'candidate';market=sample_market(2,90)
            study.saved(market,p,module.Parameters());study.saved(market,p,module.Parameters())
            trace=root/'intents'/'candidate.json';value=json.loads(trace.read_text())
            value['identity']['study']['dependencies']['observed_readiness.py']='0'*64
            trace.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'trace'):
                study.saved(market,p,module.Parameters())


if __name__=='__main__': unittest.main()

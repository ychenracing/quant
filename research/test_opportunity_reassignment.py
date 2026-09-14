"""Scheduled voluntary exits may only use mature actual holdings and real cash."""
from dataclasses import asdict
import copy
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from test_core import sample_market
from test_quantity_obligation import observe, control
from techquant.data import Market
from techquant.engine import run
from research.quantity_obligation import Owner as Parent, Parameters as ParentParameters


class OpportunityReassignmentTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research.opportunity_reassignment'),
                             'preregistered scheduled reassignment implementation is absent')
        return importlib.import_module('research.opportunity_reassignment')

    def owner(self):
        module = self.module()
        owner = module.Owner(sample_market(4, 140), module.Parameters())
        control(owner.inner)
        owner.inner.features.score[:] = [1., 3., 4., .1]
        owner.inner.features.entry[:] = True
        owner.inner.features.exit[:] = False
        return owner

    def close(self, owner, i, units=(10000., 10000., 0., 0.), cash=2_000_000.):
        return owner.decide(observe(owner.inner, i, units, cash))

    def events(self, owner):
        return [r for r in owner.trace if r.get('kind') == 'OPPORTUNITY_REASSIGNMENT']

    def test_single_candidate_and_transitive_trace_identity(self):
        module = self.module()
        self.assertEqual([asdict(p) for p in module.grid()], [{}])
        with self.assertRaises(TypeError): module.Parameters(margin=1.1)
        from research.finite_study import Study
        deps = Study('opportunity_reassignment').identity()['dependencies']
        for name in ('quantity_obligation.py', 'support_budget.py', 'funded_risk.py',
                     'opportunity_reassignment_contract.json'):
            self.assertIn(name, deps)

    def test_mature_actual_holding_and_only_scheduled_review(self):
        owner = self.owner()
        for i in range(40, 50):
            np.testing.assert_array_equal(self.close(owner, i).unit_targets,
                                          [10000., 10000., 0., 0.])
        decision = self.close(owner, 50)
        np.testing.assert_array_equal(decision.unit_targets, [0., 10000., 0., 0.])
        event, = self.events(owner)
        self.assertEqual(event['session'], 50)
        self.assertEqual(event['held_sessions'], 10)
        self.assertGreaterEqual(event['held_rank'], 3)
        self.assertEqual(event['authority'], 1.)
        self.assertEqual(event['incumbent'], owner.market.symbols[0])
        self.assertEqual(event['economic_target'], 0.)

    def test_young_holding_nonreview_competitive_rank_and_small_margin_do_not_sell(self):
        for reason in ('young', 'nonreview', 'rank', 'margin'):
            with self.subTest(reason=reason):
                owner = self.owner()
                if reason == 'young':
                    for i in range(40, 46): self.close(owner, i, (0., 10000., 0., 0.))
                    first = 46
                else: first = 40
                for i in range(first, 50): self.close(owner, i)
                if reason == 'nonreview':
                    owner.inner.features.score[50, 2] = 1.
                    self.close(owner, 50); decision = self.close(owner, 51)
                else:
                    if reason == 'rank': owner.inner.features.score[50] = [3., 4., 2., .1]
                    if reason == 'margin': owner.inner.features.score[50] = [1., 3., 1.25, .1]
                    decision = self.close(owner, 50)
                self.assertFalse(self.events(owner))
                self.assertEqual(decision.unit_targets[0], 10000.)

    def test_parent_changes_protection_warning_or_ineligible_challenger_preempt(self):
        for reason in ('buy', 'cut', 'pending', 'warning', 'eligibility'):
            with self.subTest(reason=reason):
                owner = self.owner()
                for i in range(40, 50): self.close(owner, i)
                if reason == 'buy': owner.inner.breakout[50, 1] = True
                if reason == 'cut': owner.inner.stop[1] = owner.inner.features.close[50, 1]*1.1
                if reason == 'pending': owner.inner.reduction_ceiling[1] = 9999.95
                if reason == 'warning': control(owner.inner, .5)
                if reason == 'eligibility': owner.inner.readmit[2] = True; owner.inner.healthy[2] = 0
                parent = Parent(owner.market, ParentParameters('support'))
                parent.inner = copy.deepcopy(owner.inner)
                observation = observe(owner.inner, 50, [10000., 10000., 0., 0.])
                actual, expected = owner.decide(observation), parent.decide(observation)
                np.testing.assert_array_equal(actual.unit_targets, expected.unit_targets)
                self.assertFalse(self.events(owner))

    def test_blocked_partial_sale_is_an_actual_zero_obligation_not_a_reservation(self):
        owner = self.owner()
        for i in range(40, 51): self.close(owner, i)
        owner.inner.features.score[51:] = [100., 1., 0., 8.]
        for i, remainder in ((51, 10000.), (52, 5000.), (53, .05)):
            decision = self.close(owner, i, (remainder, 10000., 0., 0.))
            np.testing.assert_array_equal(decision.unit_targets, [0., 10000., 0., 0.])
            self.assertTrue(owner.inner.exit_pending[0])
            self.assertEqual(owner.inner.reduction_ceiling[0], 0.)
        parent = Parent(owner.market, ParentParameters('support'))
        parent.inner = copy.deepcopy(owner.inner)
        observation = observe(owner.inner, 54, [0., 10000., 0., 0.], cash=200000.)
        actual, expected = owner.decide(observation), parent.decide(observation)
        np.testing.assert_array_equal(actual.unit_targets, expected.unit_targets)
        self.assertEqual(actual.unit_targets[2], 0.)
        self.assertGreater(actual.unit_targets[3], 0.)
        spent = np.maximum(actual.unit_targets-observation.units, 0.) @ owner.inner.features.close[54]
        self.assertLessEqual(spent, .99*observation.cash+1e-8)

    def test_actual_exit_and_reentry_reset_age_but_addition_does_not(self):
        owner = self.owner()
        for i in range(40, 51): self.close(owner, i)
        self.assertEqual(len(self.events(owner)), 1)
        owner.inner.features.score[51:] = [.1, 1., 4., 5.]
        self.close(owner, 51, (0., 10000., 0., 0.))
        for i in range(52, 60): self.close(owner, i, (0., 10000., 10000., 0.))
        self.close(owner, 60, (0., 15000., 10000., 0.))
        second = self.events(owner)[1]
        self.assertEqual(second['incumbent'], owner.market.symbols[1])
        self.assertEqual(second['held_sessions'], 20)

    def test_no_reassignment_is_exact_parent_with_unchanged_risk_history(self):
        owner = self.owner(); owner.inner.features.score[:] = [3., 4., 2., 1.]
        parent = Parent(owner.market, ParentParameters('support'))
        parent.inner = copy.deepcopy(owner.inner)
        for i in range(40, 70):
            observation = observe(owner.inner, i, [10000., 10000., 0., 0.])
            actual, expected = owner.decide(observation), parent.decide(observation)
            np.testing.assert_array_equal(actual.unit_targets, expected.unit_targets)
            self.assertEqual(actual.reason, expected.reason)
            self.assertEqual(actual.cap, expected.cap)
        self.assertFalse(self.events(owner))
        self.assertEqual(owner.inner.history, parent.inner.history)
        for field in ('stop', 'peak', 'exit_pending', 'reduction_ceiling', 'readmit', 'healthy'):
            np.testing.assert_array_equal(getattr(owner.inner, field), getattr(parent.inner, field))

    def test_prefix_removed_symbol_delay_and_ledger_invariants(self):
        module = self.module(); market = sample_market(4, 140)
        def execute(data, delay=1):
            return run(data, delay=delay, policy_factory=lambda current,c: module.Owner(current,module.Parameters()))
        full = execute(market); cut = market.calendar[95]; prefix = execute(market.prefix(cut))
        pd.testing.assert_frame_equal(full.equity.loc[:cut], prefix.equity)
        pd.testing.assert_frame_equal(full.targets.loc[:cut], prefix.targets)
        frames = {s:f.copy() for s,f in market.frames.items()}
        frames[market.symbols[-1]].loc[:, ['open','high','low','close','raw_open','raw_close']] *= 70
        changed = Market.from_frames(frames, market.calendar, quality='synthetic')
        names = market.symbols[:-1]
        pd.testing.assert_frame_equal(execute(market.subset(names)).equity,execute(changed.subset(names)).equity)
        from research.ledger_attribution import attribute
        for result in (full, execute(market, 2)):
            self.assertTrue((result.equity.cash >= 0).all())
            self.assertLess(attribute(market, result)[2]['max_reconciliation_error'], 1e-6)

    def test_cache_requires_exact_intent_trace(self):
        module = self.module()
        from research.finite_study import Study
        study = Study('opportunity_reassignment')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); market = sample_market(3, 100); path = root/'runs/candidate'
            first = study.saved(market, path, module.Parameters())
            second = study.saved(market, path, module.Parameters())
            pd.testing.assert_frame_equal(first.targets.rename_axis('date'),second.targets,check_freq=False)
            trace = root/'intents/candidate.json'; content = json.loads(trace.read_text())
            content['identity']['delay'] = 99; trace.write_text(json.dumps(content))
            with self.assertRaises(ValueError): study.saved(market, path, module.Parameters())
            trace.unlink()
            with self.assertRaises(ValueError): study.saved(market, path, module.Parameters())


if __name__ == '__main__': unittest.main()

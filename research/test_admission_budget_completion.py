"""One registered completion: preserve the parent and exhaust only residual authority."""
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
from research.quantity_obligation import Owner as Parent, Parameters as ParentParameters
from research.test_quantity_obligation import observe, control


class AdmissionCompletionTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research.admission_budget_completion'),
                             'registered fresh-admission completion is absent')
        return importlib.import_module('research.admission_budget_completion')

    def pair(self, n=3, entries=(0,), sectors=None, atr=1., cap=1.):
        module = self.module()
        market = sample_market(n, 100)
        if sectors is not None:
            market = Market.from_frames(market.frames, market.calendar,
                dict(zip(market.symbols, sectors)), quality='synthetic')
        candidate = module.Owner(market, module.Parameters())
        parent = Parent(market, ParentParameters('support'))
        for policy in (parent, candidate):
            inner = policy.inner
            control(inner, cap)
            inner.features.close[:] = 10.
            inner.features.entry[:] = False
            inner.features.entry[:, list(entries)] = True
            inner.features.exit[:] = False
            inner.features.score[:] = 1.
            inner.ready[:] = True
            inner.atr[:] = atr
            inner.support[:] = 0.
        return parent, candidate

    def decide_pair(self, parent, candidate, units=None, cash=2_000_000., day=40):
        units = [0.] * len(parent.market.symbols) if units is None else units
        observation = observe(parent.inner, day, units, cash)
        return observation, parent.decide(observation), candidate.decide(observation)

    def test_fixed_singleton_and_complete_dependency_identity(self):
        module = self.module()
        self.assertEqual([asdict(p) for p in module.grid()], [{}])
        with self.assertRaises(TypeError): module.Parameters(extra=1)
        from research.finite_study import Study
        identity = Study('admission_budget_completion').identity()
        for name in ('quantity_obligation.py', 'support_budget.py', 'funded_risk.py',
                     'quantity_obligation_contract.json', 'support_budget_contract.json',
                     'admission_budget_completion_contract.json'):
            self.assertIn(name, identity['dependencies'])

    def test_only_selected_fresh_order_gets_residual_global_risk(self):
        parent, candidate = self.pair()
        o, original, completed = self.decide_pair(parent, candidate)
        np.testing.assert_allclose(original.unit_targets, [100_000/3, 0, 0])
        np.testing.assert_allclose(completed.unit_targets, [200_000/3, 0, 0])
        self.assertAlmostEqual(completed.weights.sum(), 1/3)
        for name in ('stop', 'peak', 'pending_stop', 'previous_units', 'exit_pending',
                     'readmit', 'healthy', 'reduction_ceiling'):
            np.testing.assert_array_equal(getattr(parent.inner, name), getattr(candidate.inner, name))
        self.assertEqual(parent.inner.history, candidate.inner.history)
        self.assertEqual(parent.inner.last_session, candidate.inner.last_session)
        trace = candidate.trace[-1]
        self.assertEqual(trace['kind'], 'ADMISSION_BUDGET_COMPLETION')
        self.assertEqual(trace['actual_units'], list(o.units))
        self.assertEqual(trace['original_units'], list(original.unit_targets))
        self.assertEqual(trace['completed_units'], list(completed.unit_targets))
        self.assertAlmostEqual(trace['remaining_risk'], 0.)

    def test_all_original_orders_reserved_before_any_completion(self):
        parent, candidate = self.pair(entries=(0, 1))
        _, original, completed = self.decide_pair(parent, candidate)
        np.testing.assert_array_equal(completed.unit_targets, original.unit_targets)
        self.assertTrue(np.all(completed.unit_targets[:2] > 0.))
        self.assertEqual(completed.unit_targets[2], 0.)
        self.assertAlmostEqual(candidate.trace[-1]['remaining_risk'], 0.)

    def test_no_extra_symbol_or_actual_holding_enlargement(self):
        parent, candidate = self.pair(entries=(0, 1, 2))
        _, original, completed = self.decide_pair(parent, candidate, [10_000, 0, 0], 1_900_000.)
        self.assertEqual(completed.unit_targets[0], original.unit_targets[0])
        self.assertEqual(completed.unit_targets[2], 0.)
        self.assertGreater(completed.unit_targets[1], original.unit_targets[1])
        self.assertAlmostEqual(completed.unit_targets.sum()*3, 200_000.)

    def test_single_name_and_sector_limits_apply_to_whole_desired_book(self):
        parent, candidate = self.pair(entries=(1,), sectors=('optical', 'optical', 'chip'), atr=.5)
        for policy in (parent, candidate): policy.inner.atr[:,0] = .1
        _, original, completed = self.decide_pair(parent, candidate, [60_000, 0, 0], 1_400_000.)
        # With a held 30% in this sector, completion stops at the original 75% sector cap.
        self.assertGreater(completed.unit_targets[1], original.unit_targets[1])
        self.assertLessEqual(float(completed.weights[:2].sum()), .75+1e-12)
        self.assertLessEqual(completed.weights[1], .55+1e-12)
        self.assertEqual(completed.unit_targets[0], original.unit_targets[0])
        parent, candidate = self.pair(atr=.3)
        _, original, completed = self.decide_pair(parent, candidate)
        np.testing.assert_array_equal(completed.unit_targets, original.unit_targets)
        self.assertAlmostEqual(completed.weights[0], .55)

    def test_risk_cap_scaled_and_cash_reserve_never_spends_sale_proceeds(self):
        parent, candidate = self.pair(entries=(1,), atr=.4, cap=.6)
        # Make the observation start under the same cap: not a cap-cut close.
        for policy in (parent, candidate):
            policy.inner.risk.cap = .6
            policy.inner.atr[:,0] = .1
        o, original, completed = self.decide_pair(parent, candidate, [50_000, 0, 0], 1_500_000.)
        self.assertGreater(completed.unit_targets[1], original.unit_targets[1])
        self.assertLessEqual(float(completed.unit_targets@np.full(3,10.)), .6*o.nav+1e-8)
        self.assertLessEqual(float((completed.unit_targets-o.units)@np.full(3,10.)), .99*o.cash+1e-8)
        self.assertLessEqual(float(completed.unit_targets@np.array([.3,1.2,1.2])), .10*o.nav*.6+1e-8)
        self.assertGreaterEqual(candidate.trace[-1]['remaining_cash'], -1e-8)

    def test_parent_protective_exit_and_partial_retry_are_unchanged(self):
        for partial in (False, True):
            parent, candidate = self.pair(entries=(1,))
            for policy in (parent, candidate):
                if partial: policy.inner.reduction_ceiling[0] = 9_900.95
                else: policy.inner.exit_pending[0] = True
            for day in (40, 41):
                _, original, completed = self.decide_pair(parent, candidate, [10_000.05,0,0], day=day)
                np.testing.assert_array_equal(original.unit_targets, completed.unit_targets)
                self.assertEqual(completed.unit_targets[1], 0.)
                self.assertEqual(original.reason, completed.reason)
                self.assertEqual(parent.trace, candidate.trace)
                np.testing.assert_array_equal(parent.inner.reduction_ceiling, candidate.inner.reduction_ceiling)

    def test_cap_cut_and_newly_sold_readmission_do_not_create_extra_orders(self):
        parent, candidate = self.pair(entries=(0,), cap=.4)
        for policy in (parent, candidate): policy.inner.risk.cap = 1.
        _, original, completed = self.decide_pair(parent, candidate)
        np.testing.assert_array_equal(original.unit_targets, completed.unit_targets)
        self.assertEqual(original.reason, completed.reason)
        parent, candidate = self.pair(entries=(0,))
        for policy in (parent, candidate): policy.inner.previous_units[0] = 1_000.
        _, original, completed = self.decide_pair(parent, candidate)
        np.testing.assert_array_equal(original.unit_targets, completed.unit_targets)
        self.assertEqual(completed.unit_targets[0], 0.)

    def test_causal_prefix_actual_fills_cash_and_next_session(self):
        module = self.module(); market = sample_market(3, 130)
        factory = lambda m,c: module.Owner(m, module.Parameters())
        full = run(market, policy_factory=factory, delay=2)
        cut = market.calendar[90]
        short = run(market.prefix(cut), policy_factory=factory, delay=2)
        pd.testing.assert_frame_equal(full.equity.loc[:cut], short.equity)
        pd.testing.assert_frame_equal(full.targets.loc[:cut], short.targets)
        self.assertEqual([o for o in full.orders if o['date']<=str(cut.date())], short.orders)
        self.assertTrue((full.equity.cash >= -1e-6).all())
        self.assertTrue((full.equity.exposure <= 1+1e-9).all())
        for order in full.orders:
            if order['status']=='FILLED':
                self.assertGreater(market.calendar.get_loc(pd.Timestamp(order['date'])),
                                   market.calendar.get_loc(pd.Timestamp(order['signal_date'])))

    def test_cached_trace_detects_tampering(self):
        module = self.module()
        from research.finite_study import Study
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); path=root/'runs'/'candidate'
            study=Study('admission_budget_completion'); market=sample_market(2,90)
            result=study.saved(market,path,module.Parameters())
            study.saved(market,path,module.Parameters())
            trace=root/'intents'/'candidate.json'
            value=json.loads(trace.read_text()); value['trace'].append({'forged':True})
            trace.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'trace'):
                study.saved(market,path,module.Parameters())
            self.assertEqual(result.metadata['economic_acceptance'],'UNVERIFIED')


if __name__ == '__main__': unittest.main()

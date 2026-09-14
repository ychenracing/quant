"""Confirmed economic handovers must remain subordinate to actual inventory."""
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


class LeadershipHandoverTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research.leadership_handover'),
                             'registered confirmed leadership owner is absent')
        return importlib.import_module('research.leadership_handover')

    def owner(self):
        m = self.module(); adapter = m.Owner(sample_market(4, 110), m.Parameters())
        inner = adapter.inner; control(inner)
        inner.features.score[:] = [1., 3., 6., .1]
        inner.features.entry[:] = True; inner.features.exit[:] = False
        return adapter

    def close(self, adapter, i, units=(10000., 10000., 0., 0.), cash=2_000_000.):
        return adapter.decide(observe(adapter.inner, i, units, cash))

    def events(self, adapter):
        return [r for r in adapter.trace if r.get('kind') == 'LEADERSHIP_HANDOVER']

    def test_one_registered_candidate_and_all_parent_identities(self):
        m = self.module()
        self.assertEqual([asdict(p) for p in m.grid()], [{}])
        with self.assertRaises(TypeError): m.Parameters(preference=1.25)
        from research.finite_study import Study
        deps = Study('leadership_handover').identity()['dependencies']
        for name in ('leadership_handover_contract.json', 'quantity_obligation.py',
                     'quantity_obligation_contract.json', 'support_budget.py',
                     'support_budget_contract.json', 'funded_risk.py'):
            self.assertIn(name, deps)

    def test_three_same_pair_closes_then_only_weaker_exit_without_calendar_gate(self):
        a = self.owner()
        for i in (40, 41):
            np.testing.assert_array_equal(self.close(a, i).unit_targets, [10000., 10000., 0., 0.])
        decision = self.close(a, 42)
        np.testing.assert_array_equal(decision.unit_targets, [0., 10000., 0., 0.])
        self.assertTrue(a.inner.exit_pending[0]); self.assertEqual(a.inner.reduction_ceiling[0], 0.)
        event, = self.events(a)
        self.assertEqual(event['confirmation_count'], 3)
        self.assertEqual(event['incumbent'], a.market.symbols[0])
        self.assertEqual(event['challenger'], a.market.symbols[2])
        self.assertEqual(event['actual_units'], [10000., 10000., 0., 0.])
        self.assertEqual(event['authority'], 1.)
        self.assertEqual(event['economic_target'], 0.)

    def test_changed_pair_nonqualifying_close_and_session_gap_reset_confirmation(self):
        for interruption in ('pair', 'threshold', 'gap'):
            with self.subTest(interruption=interruption):
                a = self.owner(); self.close(a, 40); self.close(a, 41)
                if interruption == 'pair':
                    a.inner.features.score[42:] = [1., 3., 6., 8.]
                    self.close(a, 42); self.assertFalse(self.events(a))
                    self.close(a, 43); self.assertFalse(self.events(a))
                    self.close(a, 44); self.assertEqual(len(self.events(a)), 1)
                elif interruption == 'threshold':
                    a.inner.features.score[42] = [1., 3., 2., .1]
                    self.close(a, 42); self.close(a, 43); self.close(a, 44)
                    self.assertFalse(self.events(a)); self.close(a, 45)
                    self.assertEqual(len(self.events(a)), 1)
                else:
                    self.close(a, 43); self.close(a, 44); self.assertFalse(self.events(a))
                    self.close(a, 45); self.assertEqual(len(self.events(a)), 1)

    def test_pending_partial_sale_blocks_buys_and_does_not_cancel_on_rank_change(self):
        a = self.owner()
        for i in (40, 41, 42): self.close(a, i)
        a.inner.features.score[43:] = [100., 1., 0., 10.]
        for i, remainder in ((43, 10000.), (44, 5000.), (45, .05)):
            d = self.close(a, i, (remainder, 10000., 0., 0.))
            np.testing.assert_array_equal(d.unit_targets, [0., 10000., 0., 0.])
            self.assertEqual(a.inner.reduction_ceiling[0], 0.)
            self.assertTrue(a.inner.exit_pending[0]); self.assertEqual(len(self.events(a)), 1)
        # No reservation for the original challenger. The unchanged parent uses
        # the now-observed proceeds and current eligibility after actual zero.
        expected = Parent(a.market, ParentParameters('support'))
        expected.inner = copy.deepcopy(a.inner)
        o = observe(a.inner, 46, [0., 10000., 0., 0.], cash=200000.)
        d = a.decide(o); parent = expected.decide(o)
        np.testing.assert_array_equal(d.unit_targets, parent.unit_targets)
        self.assertEqual(d.unit_targets[2], 0.); self.assertGreater(d.unit_targets[3], 0.)
        buy = np.maximum(d.unit_targets-o.units, 0.) @ a.inner.features.close[46]
        self.assertLessEqual(buy, .99*o.cash+1e-8)

    def test_parent_buys_risk_cuts_pending_obligations_and_zero_authority_have_priority(self):
        for case in ('buy', 'cut', 'pending', 'zero', 'not_eligible'):
            with self.subTest(case=case):
                a = self.owner(); self.close(a, 40); self.close(a, 41)
                if case == 'buy': a.inner.breakout[42, 1] = True
                if case == 'cut': a.inner.stop[1] = a.inner.features.close[42, 1]*1.1
                if case == 'pending': a.inner.reduction_ceiling[1] = 9999.95
                if case == 'zero': control(a.inner, 0.)
                if case == 'not_eligible': a.inner.readmit[2] = True; a.inner.healthy[2] = 0
                expected = Parent(a.market, ParentParameters('support')); expected.inner = copy.deepcopy(a.inner)
                o = observe(a.inner, 42, [10000., 10000., 0., 0.])
                d, parent = a.decide(o), expected.decide(o)
                np.testing.assert_array_equal(d.unit_targets, parent.unit_targets)
                self.assertFalse(self.events(a))
                if case == 'buy': self.assertGreater(d.unit_targets[1], o.units[1])

    def test_cooldown_uses_signal_session_and_ties_use_symbol_order(self):
        a = self.owner(); a.inner.features.score[:] = [1., 1., 6., 6.]
        for i in (40, 41, 42): self.close(a, i)
        first, = self.events(a)
        self.assertEqual(first['incumbent'], a.market.symbols[0])
        self.assertEqual(first['challenger'], a.market.symbols[2])
        a.inner.features.score[43:] = [.1, 1., 3., 8.]
        for i in range(43, 52):
            self.close(a, i, (0., 10000., 10000., 0.)); self.assertEqual(len(self.events(a)), 1)
        self.close(a, 52, (0., 10000., 10000., 0.))
        self.assertEqual(len(self.events(a)), 2)
        self.assertEqual(self.events(a)[1]['session']-first['session'], a.inner.config.rebalance)

    def test_no_handover_is_exact_parent_including_risk_and_stop_history(self):
        a = self.owner(); a.inner.features.score[:] = [3., 3., 4., 4.]
        p = Parent(a.market, ParentParameters('support')); p.inner = copy.deepcopy(a.inner)
        for i in range(40, 65):
            o = observe(a.inner, i, [10000., 10000., 0., 0.])
            d, expected = a.decide(o), p.decide(o)
            np.testing.assert_array_equal(d.unit_targets, expected.unit_targets)
            self.assertEqual(d.reason, expected.reason); self.assertEqual(d.cap, expected.cap)
        self.assertFalse(self.events(a)); self.assertEqual(a.inner.history, p.inner.history)
        for field in ('stop', 'peak', 'exit_pending', 'reduction_ceiling', 'healthy', 'readmit'):
            np.testing.assert_array_equal(getattr(a.inner, field), getattr(p.inner, field))

    def test_prefix_removed_symbol_and_delayed_cash_inventory_reconciliation(self):
        m = self.module(); market = sample_market(4, 140)
        def execute(data, delay=1):
            return run(data, delay=delay, policy_factory=lambda current,c:m.Owner(current,m.Parameters()))
        full = execute(market); cut = market.calendar[95]; prefix = execute(market.prefix(cut))
        pd.testing.assert_frame_equal(full.equity.loc[:cut], prefix.equity)
        pd.testing.assert_frame_equal(full.targets.loc[:cut], prefix.targets)
        frames = {s:f.copy() for s,f in market.frames.items()}
        frames[market.symbols[-1]].loc[:, ['open','high','low','close','raw_open','raw_close']] *= 70
        changed = Market.from_frames(frames, market.calendar, quality='synthetic')
        names = market.symbols[:-1]; a, b = execute(market.subset(names)), execute(changed.subset(names))
        self.assertEqual(a.orders, b.orders); pd.testing.assert_frame_equal(a.equity, b.equity)
        from research.ledger_attribution import attribute
        for r in (full, execute(market, 2)):
            self.assertTrue((r.equity.cash >= 0).all())
            self.assertTrue(all(o['signal_date'] < o['date'] for o in r.orders if o['status'] == 'FILLED'))
            self.assertLess(attribute(market, r)[2]['max_reconciliation_error'], 1e-6)

    def test_cache_rejects_missing_or_changed_source_bound_trace(self):
        m = self.module()
        from research.finite_study import Study
        study = Study('leadership_handover'); market = sample_market(3, 100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root/'runs'/'candidate'
            result = study.saved(market, path, m.Parameters())
            cached = study.saved(market, path, m.Parameters())
            pd.testing.assert_frame_equal(result.targets.rename_axis('date'), cached.targets, check_freq=False)
            trace = root/'intents/candidate.json'; saved = json.loads(trace.read_text())
            saved['identity']['delay'] = 99; trace.write_text(json.dumps(saved))
            with self.assertRaises(ValueError): study.saved(market, path, m.Parameters())
            trace.unlink()
            with self.assertRaises(ValueError): study.saved(market, path, m.Parameters())


if __name__ == '__main__': unittest.main()

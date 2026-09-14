"""Actual-inventory contracts for the preregistered support-risk allocator."""
from dataclasses import asdict
import importlib
import importlib.util
import unittest
import numpy as np
import pandas as pd
from test_core import sample_market
from techquant.data import Market
from techquant.engine import run
from techquant.policy import CloseObservation


class SupportBudgetTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research.support_budget'),
                             'preregistered support-risk allocator is not implemented')
        return importlib.import_module('research.support_budget')

    def owner(self, market=None, risk=.06, positions=2):
        module = self.module()
        return module.Owner(market or sample_market(2, 130), module.Parameters(risk, positions))

    def cap(self, owner, value):
        def update(*args):
            owner.risk.cap = value
            return value, 'CONTROLLED_RISK_CAP'
        owner.risk.update = update

    def observation(self, owner, i, units=None, cash=2_000_000.):
        units = np.zeros(len(owner.market.symbols)) if units is None else np.array(units, dtype=float)
        values = np.nan_to_num(units * owner.features.close[i], nan=0.)
        nav = float(cash + values.sum())
        return CloseObservation.from_inventory(i, str(owner.market.calendar[i].date()),
                                               nav, cash, units, values / nav)

    def test_runner_registers_only_the_declared_family_and_binds_its_contract(self):
        from research.finite_study import Study
        study = Study('support_budget')
        self.assertEqual(len(study.module.grid()), 4)
        self.assertIn('support_budget_contract.json', study.identity()['dependencies'])
        with self.assertRaises(ValueError):
            Study('support_budget_unregistered')

    def test_grid_and_initial_commitments_obey_risk_cash_and_capacity(self):
        module = self.module()
        self.assertEqual([asdict(p) for p in module.grid()],
            [{'risk_budget': r, 'positions': n} for r in (.06, .10) for n in (2, 4)])
        for risk, positions in [(True, 2), (.2, 2), (.06, True), (.06, 3)]:
            with self.assertRaises(ValueError):
                module.Parameters(risk, positions)
        owner = self.owner(sample_market(5, 130))
        self.cap(owner, 1.)
        o = self.observation(owner, 40)
        d = owner.decide(o)
        d.validated_weights(5)
        d.validated_unit_targets(owner.features.close[40], o.nav)
        q, price = d.unit_targets, owner.features.close[40]
        distance = np.maximum(price-owner.admission_stop(40), .02*price)
        self.assertGreater(q.sum(), 0.)
        self.assertLessEqual(np.count_nonzero(q), 2)
        self.assertLessEqual(float(q@price), .99*o.cash + 1e-7)
        self.assertLessEqual(float(q@distance), .06*o.nav + 1e-7)
        self.assertTrue(np.all(q*distance <= .06*o.nav/2 + 1e-7))
        np.testing.assert_array_equal(o.units, np.zeros(5))

    def test_stop_is_not_funded_by_a_request_and_is_inherited_by_partial_fill(self):
        owner = self.owner(sample_market(1, 130))
        self.cap(owner, 1.)
        d = owner.decide(self.observation(owner, 40))
        declared = owner.pending_stop.copy()
        self.assertGreater(d.unit_targets[0], 0.)
        self.assertEqual(owner.stop[0], 0.)
        owner.decide(self.observation(owner, 41))  # opening buy blocked
        self.assertEqual(owner.stop[0], 0.)
        self.assertGreaterEqual(owner.pending_stop[0], declared[0])
        owner.decide(self.observation(owner, 42, [1000.]))
        self.assertGreaterEqual(owner.stop[0], declared[0])

    def test_existing_units_are_not_rebalanced_and_support_never_moves_down(self):
        owner = self.owner(sample_market(1, 130))
        self.cap(owner, 1.)
        owner.breakout[:] = False
        first = owner.decide(self.observation(owner, 40, [1000.]))
        old = owner.stop.copy()
        # Wider subsequent volatility cannot loosen a funded protective stop.
        owner.atr[41:] *= 10.
        second = owner.decide(self.observation(owner, 41, [1000.]))
        np.testing.assert_array_equal(first.unit_targets, [1000.])
        np.testing.assert_array_equal(second.unit_targets, [1000.])
        self.assertTrue(np.all(owner.stop >= old))

    def test_material_additions_to_funded_winners_need_a_fresh_high(self):
        owner = self.owner(sample_market(1, 130))
        self.cap(owner, 1.)
        o = self.observation(owner, 40, [1000.])
        d = owner.decide(o)
        self.assertGreater(d.unit_targets[0], o.units[0])
        addition = (d.unit_targets[0]-o.units[0])*owner.features.close[40, 0]
        self.assertGreaterEqual(addition, owner.config.trade_band*o.nav)
        owner.breakout[41] = False
        retained = owner.decide(self.observation(owner, 41, [1000.]))
        np.testing.assert_array_equal(retained.unit_targets, [1000.])

    def test_protective_sell_latches_through_rebound_and_reentry_needs_actual_exit(self):
        owner = self.owner(sample_market(1, 130))
        self.cap(owner, 1.)
        owner.decide(self.observation(owner, 40, [50000.], 100000.))
        owner.stop[0] = owner.features.close[41, 0]*1.01
        breached = owner.decide(self.observation(owner, 41, [50000.], 100000.))
        retry = owner.decide(self.observation(owner, 42, [50000.], 100000.))
        np.testing.assert_array_equal(breached.unit_targets, [0.])
        np.testing.assert_array_equal(retry.unit_targets, [0.])
        self.assertTrue(owner.exit_pending[0])
        for i in (43, 44):
            d = owner.decide(self.observation(owner, i))
            self.assertEqual(d.unit_targets[0], 0.)
        ready = owner.decide(self.observation(owner, 45))
        self.assertGreater(ready.unit_targets[0], 0.)

    def test_cap_cut_does_not_spend_expected_sales_and_retries_partial_inventory(self):
        owner = self.owner(sample_market(2, 130))
        self.cap(owner, .25)
        owner.risk.cap = 1.
        o = self.observation(owner, 40, [75001.5, 0.], 20000.)
        first = owner.decide(o)
        self.assertLess(first.unit_targets[0], o.units[0])
        self.assertEqual(first.unit_targets[1], 0.)
        outstanding = owner.decide(self.observation(owner, 41, [70000., 0.], 140000.))
        self.assertLessEqual(outstanding.unit_targets[0], first.unit_targets[0]+1e-9)
        self.assertEqual(outstanding.unit_targets[1], 0.)
        self.assertLessEqual(outstanding.weights.sum(), outstanding.cap+1e-12)

    def test_unfresh_holding_exits_and_unheld_stale_quote_never_enters(self):
        market = sample_market(2, 130)
        frames = {s:f.copy() for s,f in market.frames.items()}
        frames[market.symbols[0]].loc[market.calendar[40:43], 'volume'] = 0.
        market = Market.from_frames(frames, market.calendar, quality='synthetic')
        owner = self.owner(market)
        self.cap(owner, 1.)
        d = owner.decide(self.observation(owner, 40, [1000., 0.]))
        self.assertEqual(d.unit_targets[0], 0.)
        self.assertEqual(d.unit_targets[1], 0.)  # do not fund entry from assumed exit
        flat = self.owner(market)
        self.cap(flat, 1.)
        self.assertEqual(flat.decide(self.observation(flat, 40)).unit_targets[0], 0.)

    def test_prefix_exclusion_and_delayed_execution_preserve_causal_cash_ledger(self):
        module = self.module()
        market = sample_market(4, 145)
        frames = {s:f.copy() for s,f in market.frames.items()}
        for f in frames.values():
            f.loc[market.calendar[65:80], ['open','high','low','close','raw_open','raw_close']] *= .82
        market = Market.from_frames(frames, market.calendar, quality='synthetic')
        def execute(m, delay=1):
            return run(m, delay=delay, policy_factory=lambda current,cfg:
                       module.Owner(current, module.Parameters()))
        full = execute(market)
        cut = market.calendar[110]
        short = execute(market.prefix(cut))
        pd.testing.assert_frame_equal(full.equity.loc[:cut], short.equity)
        pd.testing.assert_frame_equal(full.targets.loc[:cut], short.targets)
        excluded = market.symbols[-1]
        frames[excluded] = frames[excluded].copy()
        frames[excluded].loc[:, ['open','high','low','close','raw_open','raw_close']] *= 100.
        changed = Market.from_frames(frames, market.calendar, quality='synthetic')
        a,b = execute(market.subset(market.symbols[:-1])), execute(changed.subset(market.symbols[:-1]))
        pd.testing.assert_frame_equal(a.equity, b.equity)
        self.assertEqual(a.orders, b.orders)
        for result in (full, execute(market, 2)):
            self.assertTrue((result.equity.cash >= -1e-8).all())
            self.assertTrue((result.equity.exposure <= 1+1e-10).all())
            self.assertTrue(all(o['signal_date'] < o['date'] for o in result.orders if o['status']=='FILLED'))
            from research.ledger_attribution import attribute
            _,_,summary = attribute(market, result)
            self.assertLess(summary['max_reconciliation_error'], 1e-6)


if __name__ == '__main__':
    unittest.main()

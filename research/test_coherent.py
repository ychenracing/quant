"""Focused admission and actual-inventory coordination regressions."""
from dataclasses import replace
import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from test_core import sample_market
from techquant.policy import CloseObservation
from research.nonlinear import Prediction


class CoherentOwnershipTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module('research.coherent')
        except ImportError as exc:
            self.fail(f'Coherent admission/filled-account owner missing: {exc}')

    def owner(self, n=6):
        market = sample_market(n, 90)
        shape = (90, n)
        price = market.panel('close').to_numpy()
        f = Prediction(market.symbols, market.fingerprint(), 20,
                       np.tile(np.linspace(.1, .02, n), (90, 1)),
                       np.full(shape, .01), np.ones(shape, dtype=bool),
                       price, price * .99, price * .98, price * .97,
                       np.full(shape, .01), np.full(shape, .001), [])
        owner = self.module.Owner(market, self.module.Parameters(), prediction=f)
        owner.s = SimpleNamespace(entry=np.ones(shape, dtype=bool),
                                  exit=np.zeros(shape, dtype=bool), market=np.ones(90, dtype=bool))
        owner.risk.cap = 1.
        return owner, market, f

    def observation(self, owner, i, units, cash=1000.):
        units = np.asarray(units, dtype=float)
        values = units * owner.f.price[i]
        nav = float(cash + values.sum())
        return CloseObservation.from_inventory(i, str(owner.market.calendar[i].date()),
                                               nav, cash, units, values / nav)

    def decide(self, owner, i, units, cash=1000., cap=1., reason='TREND_OPEN'):
        def risk_update(*args):
            owner.risk.cap = cap
            return cap, reason
        with patch.object(owner.risk, 'update', side_effect=risk_update):
            return owner.decide(self.observation(owner, i, units, cash))

    def test_admission_precedes_slots_and_cash_allocation(self):
        owner, _, _ = self.owner()
        owner.s.entry[30, :2] = False
        decision = self.decide(owner, 30, np.zeros(6))
        np.testing.assert_array_equal(decision.weights[:2], [0., 0.])
        self.assertEqual(np.count_nonzero(decision.weights), 4)
        self.assertAlmostEqual(float(decision.weights.sum()), 1.)

    def test_low_utility_does_not_shadow_price_admitted_names(self):
        owner, _, f = self.owner(2)
        f.expected[:] = -.001
        decision = self.decide(owner, 30, [0., 0.])
        self.assertGreater(float(decision.weights.sum()), .99)

    def test_stale_exit_blocks_new_risk_until_actual_liquidation(self):
        owner, _, f = self.owner(2)
        f.ready[30, 0] = False
        first = self.decide(owner, 30, [20., 0.])
        second = self.decide(owner, 31, [20., 0.])
        np.testing.assert_array_equal(first.unit_targets, [0., 0.])
        np.testing.assert_array_equal(second.unit_targets, [0., 0.])
        self.assertIn('FULL_EXIT_RETRY', second.reason)

    def test_risk_cut_is_fixed_units_not_repeated_residual_halving(self):
        owner, _, _ = self.owner(2)
        first = self.decide(owner, 30, [20., 20.], cash=0., cap=.5)
        # Neither queued sale filled. Recovery must not cancel the instruction.
        second = self.decide(owner, 31, [20., 20.], cash=0., cap=1.)
        np.testing.assert_allclose(first.unit_targets, [10., 10.])
        self.assertAlmostEqual(first.cap, .5)
        np.testing.assert_allclose(second.unit_targets, first.unit_targets)
        # A partial fill is still above the exact original target, not halved again.
        third = self.decide(owner, 32, [15., 15.], cash=200., cap=1.)
        np.testing.assert_allclose(third.unit_targets, first.unit_targets)

    def test_recovery_funds_intact_holdings_only_from_observed_cash(self):
        owner, _, _ = self.owner(2)
        self.decide(owner, 30, [20., 20.], cash=0., cap=.5)
        settled = self.decide(owner, 31, [10., 10.], cash=500., cap=.5)
        np.testing.assert_allclose(settled.unit_targets, [10., 10.])
        restored = self.decide(owner, 32, [10., 10.], cash=500., cap=1., reason='CONFIRMED_RECOVERY')
        self.assertTrue((restored.unit_targets > 10.).all())
        self.assertLessEqual(float(restored.weights.sum()), 1. + 1e-12)
        extra = np.dot(restored.unit_targets - 10., owner.f.price[32])
        self.assertLessEqual(extra, 500. + 1e-7)

    def test_hold_intact_units_when_no_new_funding_event(self):
        owner, _, _ = self.owner(2)
        seen = self.observation(owner, 30, [10., 10.], cash=200.)
        request = self.decide(owner, 30, [10., 10.], cash=200.)
        np.testing.assert_array_equal(request.unit_targets, seen.units)
        request.validated_unit_targets(owner.f.price[30], seen.nav)

    def test_rejects_prediction_from_a_different_universe(self):
        owner, market, forecast = self.owner(2)
        with self.assertRaises(ValueError):
            self.module.Owner(market, self.module.Parameters(),
                              prediction=replace(forecast, symbols=('foreign',)))

    def test_real_execution_is_prefix_causal_and_reconciles(self):
        from techquant.engine import run
        from research.pathwise import forecast, Parameters as ForecastParameters
        from research.ledger_attribution import attribute
        import pandas as pd
        market = sample_market(3, 115)
        cutoff = market.calendar[89]
        def execute(m):
            issued = forecast(m, ForecastParameters(20, .12))
            return run(m, policy_factory=lambda current, cfg: self.module.Owner(
                current, self.module.Parameters('price'), prediction=issued))
        full = execute(market)
        short = execute(market.prefix(cutoff))
        pd.testing.assert_frame_equal(full.equity.loc[:cutoff], short.equity)
        pd.testing.assert_frame_equal(full.targets.loc[:cutoff], short.targets)
        self.assertTrue(any(o['status'] == 'FILLED' for o in full.orders))
        self.assertTrue((full.equity.cash >= 0).all())
        _, _, checks = attribute(market, full)
        self.assertLess(checks['max_reconciliation_error'], 1e-6)

    def test_duplicate_session_is_rejected(self):
        owner, _, _ = self.owner(2)
        self.decide(owner, 30, [0., 0.])
        with self.assertRaises(ValueError):
            self.decide(owner, 30, [0., 0.])


if __name__ == '__main__':
    unittest.main()

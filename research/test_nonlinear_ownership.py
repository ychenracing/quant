"""Contract regressions for the separately registered nonlinear owner."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from techquant.policy import CloseObservation
from research import nonlinear


class OwnershipTests(unittest.TestCase):
    def owner(self, size=2, positions=1):
        shape = (50, size)
        forecast = SimpleNamespace(
            expected=np.full(shape, .02), tail=np.full(shape, .05),
            ready=np.ones(shape, dtype=bool), price=np.full(shape, 10.),
            ema10=np.full(shape, 9.), ema20=np.full(shape, 9.),
            ema60=np.full(shape, 9.), momentum5=np.full(shape, .02),
            ret1=np.full(shape, .01),
        )
        market = SimpleNamespace(symbols=tuple(f'name{i}' for i in range(size)))
        with patch.object(nonlinear, 'prepared', return_value=forecast):
            owner = nonlinear.Owner(market, nonlinear.Parameters(positions=positions))
        return owner, forecast

    def observation(self, session, units, cash=1000.):
        units = np.asarray(units, dtype=float)
        nav = cash + units.sum() * 10.
        return CloseObservation.from_inventory(
            session, str(session), nav, cash, units, units * 10. / nav)

    def test_retention_emits_fixed_actual_units_not_opening_nav_weights(self):
        owner, _ = self.owner()
        observed = self.observation(1, [100., 0.])
        decision = owner.decide(observed)
        self.assertIsNotNone(decision.unit_targets,
                             'funded-unit policy must opt into inventory intent')
        np.testing.assert_allclose(decision.unit_targets, observed.units)
        np.testing.assert_allclose(
            decision.validated_unit_targets(np.array([10., 10.]), observed.nav),
            observed.units)

    def test_protective_exit_retries_after_signal_recovers(self):
        owner, forecast = self.owner()
        forecast.tail[1, 0] = .8
        forecast.ret1[1, 0] = -.02
        self.assertEqual(owner.decide(self.observation(1, [100., 0.])).weights[0], 0.)
        # The first sell was blocked; recovery is not evidence of liquidation.
        decision = owner.decide(self.observation(2, [100., 0.]))
        np.testing.assert_array_equal(decision.weights, [0., 0.])
        self.assertIn('EXIT_RETRY', decision.reason)

    def test_leader_replacement_retries_until_actual_liquidation(self):
        owner, forecast = self.owner(size=5, positions=2)
        forecast.expected[:] = [.01, .02, .08, .07, .06]
        held = [100., 100., 0., 0., 0.]
        first = owner.decide(self.observation(20, held, cash=0.))
        self.assertEqual(first.weights[0], 0.)
        self.assertIn('LEADER_REPLACEMENT', first.reason)
        second = owner.decide(self.observation(21, held, cash=0.))
        self.assertEqual(second.weights[0], 0.,
                         'unfilled full exit must survive the rotation day')
        self.assertEqual(second.weights[2:].sum(), 0.)

    def test_readmission_counts_only_post_liquidation_healthy_closes(self):
        owner, forecast = self.owner(size=1)
        forecast.tail[10, 0] = .8
        forecast.ret1[10, 0] = -.02
        # Healthy history before the exit must not unlock a repurchase.
        owner.healthy[:] = 20
        owner.decide(self.observation(10, [100.], cash=0.))
        first = owner.decide(self.observation(11, [0.]))
        second = owner.decide(self.observation(12, [0.]))
        third = owner.decide(self.observation(13, [0.]))
        self.assertEqual(first.weights[0], 0.)
        self.assertEqual(second.weights[0], 0.)
        self.assertGreater(third.weights[0], 0.)

    def test_stale_held_quote_cannot_open_another_position(self):
        owner, forecast = self.owner(positions=2)
        forecast.ready[1, 0] = False
        decision = owner.decide(self.observation(1, [100., 0.]))
        np.testing.assert_array_equal(decision.weights, [0., 0.])


if __name__ == '__main__':
    unittest.main()

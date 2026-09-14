"""Executable protective intent and honest partial-buy completion boundaries."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from test_core import sample_market
from techquant.data import Market
from research import coherent, test_coherent


class ResidualIntentTests(unittest.TestCase):
    def owner(self, symbols=('sz300100', 'sz300101'), caps=None):
        helper = test_coherent.CoherentTests()
        original, prediction, observed = helper.owner(len(symbols))
        market = Market.from_frames(dict(zip(symbols, original.market.frames.values())),
                                   original.market.calendar, quality='synthetic')
        prediction = replace(prediction, symbols=market.symbols, data_sha256=market.fingerprint())
        with patch.object(coherent, 'signals', return_value=observed), \
             patch.object(coherent, 'build_features', return_value=original.features):
            owner = coherent.Owner(market, coherent.Parameters(), prediction=prediction)
        owner.risk.cap = 1.
        sequence = iter(caps or [1.] * len(market.calendar))
        def update(i, features, navs, config):
            owner.risk.cap = next(sequence)
            return owner.risk.cap, 'CONTROLLED_RISK_OBSERVATION'
        owner.risk.update = update
        return helper, owner, observed

    def test_protective_sub_lot_is_tightened_and_retried_not_forgiven(self):
        for symbol, expected in [('sz300100', 900.), ('sh688001', 800.), ('bj920001', 900.)]:
            with self.subTest(symbol=symbol):
                helper, owner, _ = self.owner((symbol,))
                owner.reduction_ceiling[0] = 999.9
                first = helper.decision(owner, 1, [1000.], 10000.)
                self.assertEqual(first.unit_targets[0], expected)
                self.assertIn('EXECUTABLE_PROTECTIVE_REDUCTION', first.reason)
                self.assertLessEqual(owner.reduction_ceiling[0], 999.9)
                blocked = helper.decision(owner, 2, [1000.], 10000.)
                np.testing.assert_array_equal(blocked.unit_targets, first.unit_targets)
                filled = helper.decision(owner, 3, [expected], 20000. - expected * 10.)
                self.assertNotIn('PROTECTIVE_INVENTORY_RETRY', filled.reason)

    def test_nonintegral_protection_uses_smallest_permitted_sale(self):
        helper, owner, _ = self.owner(('sh688001',))
        owner.reduction_ceiling[0] = 799.9
        decision = helper.decision(owner, 1, [1000.], 10000.)
        self.assertEqual(decision.unit_targets[0], 799.)

    def test_inventory_smaller_than_minimum_uses_full_odd_lot_exit(self):
        helper, owner, _ = self.owner(('sh688001',))
        owner.reduction_ceiling[0] = 99.9
        decision = helper.decision(owner, 1, [100.], 10000.)
        self.assertEqual(decision.unit_targets[0], 0.)
        self.assertIn('PROTECTIVE_INVENTORY_RETRY', decision.reason)

    def test_actual_partial_buy_retires_only_immaterial_remaining_intent(self):
        helper, owner, _ = self.owner(caps=[1., 1., 1.])
        owner.risk.cap = .5
        helper.decision(owner, 1, [50000., 50000.], 1000000.)
        actual = [99500., 100000.]
        decision = helper.decision(owner, 2, actual, 5000.)
        np.testing.assert_array_equal(decision.unit_targets, actual)
        self.assertIsNone(owner.restoration_targets)
        self.assertFalse(owner.restoration_pending)
        self.assertIn('RESTORATION_RESIDUAL_CANCELLED', decision.reason)
        self.assertIn('MINIMUM_NOTIONAL', decision.reason)
        self.assertIn('5000', decision.reason)
        again = helper.decision(owner, 3, actual, 5000.)
        np.testing.assert_array_equal(again.unit_targets, actual)

    def test_no_fill_does_not_cancel_even_an_unexecutable_buy(self):
        helper, owner, _ = self.owner()
        owner.restoration_pending = True
        owner.restoration_targets = np.array([50010., 50010.])
        owner.restoration_origin_units = np.array([50000., 50000.])
        decision = helper.decision(owner, 1, [50000., 50000.], 1000000.)
        self.assertIsNotNone(owner.restoration_targets)
        np.testing.assert_array_equal(owner.restoration_targets, [50010., 50010.])
        self.assertNotIn('RESTORATION_RESIDUAL_CANCELLED', decision.reason)

    def test_progress_is_required_for_each_symbol_not_the_whole_account(self):
        helper, owner, _ = self.owner()
        owner.restoration_pending = True
        owner.restoration_targets = np.array([50010., 50010.])
        owner.restoration_origin_units = np.array([50000., 50000.])
        decision = helper.decision(owner, 1, [50005., 50000.], 999950.)
        np.testing.assert_array_equal(owner.restoration_targets, [50005., 50010.])
        self.assertIn('RESTORATION_RESIDUAL_CANCELLED:sz300100', decision.reason)
        self.assertNotIn('RESTORATION_RESIDUAL_CANCELLED:sz300101', decision.reason)

    def test_partial_fill_with_material_executable_remainder_stays_pending(self):
        helper, owner, _ = self.owner(caps=[1., 1.])
        owner.risk.cap = .5
        helper.decision(owner, 1, [50000., 50000.], 1000000.)
        decision = helper.decision(owner, 2, [60000., 50000.], 900000.)
        np.testing.assert_array_equal(decision.unit_targets, [100000., 100000.])
        self.assertNotIn('RESTORATION_RESIDUAL_CANCELLED', decision.reason)

    def test_lot_remainder_can_be_material_but_unexecutable_after_a_fill(self):
        helper, owner, _ = self.owner(('sh688001',))
        owner.restoration_pending = True
        owner.restoration_targets = np.array([399.])
        owner.restoration_origin_units = np.array([200.])
        decision = helper.decision(owner, 1, [250.], 10000.)
        self.assertEqual(decision.unit_targets[0], 250.)
        self.assertIn('MINIMUM_LOT', decision.reason)
        self.assertIsNone(owner.restoration_targets)


if __name__ == '__main__':
    unittest.main()

"""Causal learning and executable ownership, not historical profit assertions."""
import unittest
import numpy as np
import pandas as pd
from test_core import sample_market


class ExpectationTests(unittest.TestCase):
    def module(self):
        try:
            from research import expectation
            return expectation
        except ImportError as exc:
            self.fail(f'causal expectation implementation missing: {exc}')

    def test_future_price_changes_cannot_change_past_forecasts_or_fits(self):
        e = self.module()
        m = sample_market(3, 110)
        params = e.Parameters(horizon=10, shrinkage=1., positions=2)
        a = e.forecast(m, params)
        cut = m.calendar[79]
        b = e.forecast(m.prefix(cut), params)
        np.testing.assert_allclose(a.utility[:80], b.utility, rtol=0, atol=1e-12)
        self.assertEqual([f for f in a.fits if f['signal_session'] < 80], b.fits)
        self.assertTrue(a.fits)
        self.assertTrue(all(f['maximum_label_end_session'] <= f['signal_session'] for f in a.fits))
        # Changing securities which were excluded must not alter the supplied pool.
        one = e.forecast(m.subset(m.symbols[:1]), params)
        self.assertEqual(one.utility.shape, (len(m.calendar), 1))

    def test_missing_future_label_path_is_excluded_not_forward_filled(self):
        e = self.module()
        close = np.full((25, 2), 20.)
        open_ = close.copy()
        close[8, 1] = np.nan
        labels = e.matured_labels(close, open_, horizon=10, cutoff=10)
        self.assertEqual(labels.shape, (1, 2, 2))
        self.assertTrue(np.isfinite(labels[0, 0]).all())
        self.assertTrue(np.isnan(labels[0, 1]).all())
        self.assertEqual(e.matured_labels(close, open_, horizon=10, cutoff=9).shape[0], 0)

    def test_risk_sell_stays_latched_until_actual_inventory_is_zero(self):
        e = self.module()
        from techquant.policy import CloseObservation
        m = sample_market(1, 50)
        p = e.forecast(m, e.Parameters())
        p.utility[:] = 1.
        p.price[11, 0] = p.price[10, 0] * .85
        owner = e.Owner(m.symbols, p, e.Parameters())
        def observed(i, held):
            return CloseObservation.from_inventory(i, str(m.calendar[i].date()), 100.,
                                                   50. if held else 100.,
                                                   np.array([2. if held else 0.]),
                                                   np.array([.5 if held else 0.]))
        owner.decide(observed(10, True))
        first = owner.decide(observed(11, True))
        # Rebounding quotes cannot undo an unfilled protective exit.
        retry = owner.decide(observed(12, True))
        self.assertEqual(first.weights[0], 0.)
        self.assertEqual(retry.weights[0], 0.)
        self.assertIn('PROTECTIVE_EXIT_RETRY', retry.reason)
        self.assertEqual(owner.decide(observed(13, False)).weights[0], 0.)

    def test_entry_day_loss_uses_observed_fill_open_not_only_first_close(self):
        e = self.module()
        from techquant.policy import CloseObservation
        m = sample_market(1, 50)
        p = e.forecast(m, e.Parameters())
        p.utility[:] = 1.
        p.price[10, 0] *= .8
        owner = e.Owner(m.symbols, p, e.Parameters())
        decision = owner.decide(CloseObservation.from_inventory(
            10, str(m.calendar[10].date()), 100., 50., np.array([2.]), np.array([.5])))
        self.assertEqual(decision.weights[0], 0., 'first-close peak hides an entry-day loss')

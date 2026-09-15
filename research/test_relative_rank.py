"""Causality and data-integrity tests for the fixed relative-ranking primitive."""
import unittest
import numpy as np
from research.relative_rank import cohort_features, mature_cohort, learn_relative_rank


class RelativeRankTests(unittest.TestCase):
    @staticmethod
    def sample(length=110, names=3):
        t = np.arange(length, dtype=float)[:, None]
        slopes = np.linspace(.001, .006, names)[None, :]
        close = 10 * np.exp(t * slopes + .003*np.sin(t/4 + np.arange(names)))
        opened = close * 1.001
        active = np.ones(close.shape, dtype=bool)
        ready = active.copy(); ready[:5] = False
        return close, opened, active, ready

    def compute(self, data):
        return learn_relative_rank(*data, fast=5, slow=10, horizon=3, refit=3)

    def test_maturity_is_strict_next_open_endpoint(self):
        close, opened, active, ready = self.sample()
        f = cohort_features(close, ready, fast=5, slow=10)
        self.assertIsNone(mature_cohort(f, opened, active, 10, 3, 13))
        self.assertIsNotNone(mature_cohort(f, opened, active, 10, 3, 14))

    def test_unseen_future_cannot_change_any_issued_rank(self):
        data = self.sample(); ranks, _ = self.compute(data)
        future = tuple(x.copy() for x in data)
        future[0][70:] *= np.array([.25, 4., 1.7])
        future[1][70:] *= np.array([2., .4, 1.3])
        changed, _ = self.compute(future)
        np.testing.assert_array_equal(ranks[:70], changed[:70])

    def test_prefix_is_bitwise_identical(self):
        data = self.sample(); all_ranks, _ = self.compute(data)
        prefix, _ = self.compute(tuple(x[:70] for x in data))
        np.testing.assert_array_equal(all_ranks[:70], prefix)

    def test_missing_endpoint_is_omitted_not_zero_return(self):
        close, opened, active, ready = self.sample()
        f = cohort_features(close, ready, fast=5, slow=10)
        opened[14, 0] = np.nan; active[14, 0] = False
        x, y = mature_cohort(f, opened, active, 10, 3, 14)
        self.assertEqual(len(y), 2)
        self.assertAlmostEqual(float(y.sum()), 0., places=14)
        self.assertTrue(np.isfinite(x).all())

    def test_inactive_endpoint_cannot_enter_training(self):
        close, opened, active, ready = self.sample()
        f = cohort_features(close, ready, fast=5, slow=10)
        active[11, :2] = False
        self.assertIsNone(mature_cohort(f, opened, active, 10, 3, 14))

    def test_common_cohort_return_is_removed(self):
        close, opened, active, ready = self.sample()
        f = cohort_features(close, ready, fast=5, slow=10)
        x, y = mature_cohort(f, opened, active, 10, 3, 14)
        changed = opened.copy(); changed[14] *= 1.25
        x2, y2 = mature_cohort(f, changed, active, 10, 3, 14)
        np.testing.assert_array_equal(x, x2)
        np.testing.assert_allclose(y, y2, atol=1e-15, rtol=0)

    def test_single_name_is_explicit_fallback(self):
        ranks, fits = self.compute(self.sample(names=1))
        self.assertTrue(np.isnan(ranks).all())
        self.assertTrue(all(f['usable_cohorts'] == 0 for f in fits))

    def test_fit_provenance_never_reaches_future(self):
        ranks, fits = self.compute(self.sample())
        self.assertTrue(np.isfinite(ranks[-1]).all())
        for fit in fits:
            self.assertLessEqual(fit['latest_label_session'], fit['session'])
            if fit['coefficients'] is not None:
                self.assertGreaterEqual(fit['usable_cohorts'], 10)

    def test_model_learns_relative_not_absolute_return_sign(self):
        ranks, _ = self.compute(self.sample())
        self.assertGreater(ranks[-1, 2], ranks[-1, 0])
        self.assertLess(ranks[-1, 0], 0)
        self.assertGreater(ranks[-1, 2], 0)

    def test_column_permutation_does_not_change_economic_order(self):
        data = self.sample(); rank, _ = self.compute(data)
        order = [2, 0, 1]
        shuffled, _ = self.compute(tuple(x[:, order] for x in data))
        np.testing.assert_allclose(rank[:, order], shuffled, atol=1e-14, rtol=1e-12)

    def test_constant_feature_column_does_not_divide_by_zero(self):
        close = np.full((100, 3), 10.); flags = np.ones(close.shape, dtype=bool)
        ranks, _ = self.compute((close, close.copy(), flags, flags.copy()))
        self.assertTrue(np.isnan(ranks).all())

    def test_incomplete_new_listing_has_no_backward_fill(self):
        data = list(self.sample()); data[0][:50, 2] = np.nan
        data[1][:50, 2] = np.nan; data[2][:50, 2] = False; data[3][:55, 2] = False
        f = cohort_features(data[0], data[3], fast=5, slow=10)
        self.assertTrue(np.isnan(f[:55, 2]).all())
        ranks, _ = self.compute(tuple(data))
        self.assertTrue(np.isnan(ranks[:55, 2]).all())

    def test_bad_shape_rejected(self):
        data = list(self.sample()); data[1] = data[1][:-1]
        with self.assertRaises(ValueError): self.compute(tuple(data))

    def test_nonpositive_observed_price_rejected(self):
        data = list(self.sample()); data[0][20, 0] = 0
        with self.assertRaises(ValueError): self.compute(tuple(data))

    def test_ready_but_inactive_rejected(self):
        data = list(self.sample()); data[2][20, 0] = False
        with self.assertRaises(ValueError): self.compute(tuple(data))

    def test_unbounded_or_zero_horizon_rejected(self):
        data = self.sample()
        for kwargs in ({'horizon': 0}, {'slow': 5}, {'refit': True}):
            args = dict(fast=5, slow=10, horizon=3, refit=3); args.update(kwargs)
            with self.assertRaises(ValueError): learn_relative_rank(*data, **args)



class RankOrderTests(unittest.TestCase):
    def test_missing_rank_falls_back_whole_order_without_mixing_scales(self):
        from research.relative_rank import rank_order
        self.assertEqual(rank_order([3.,2.,1.], [.01,np.nan,.02], ('a','b','c'), [0,1,2]), [0,1,2])

    def test_relative_sign_is_not_an_eligibility_filter(self):
        from research.relative_rank import rank_order
        self.assertEqual(rank_order([3.,2.,1.], [-.3,-.2,-.1], ('a','b','c'), [0,1,2]), [2,1,0])
        self.assertEqual(rank_order([3.,2.,1.], [-.3,-.2,-.1], ('a','b','c'), [0,2]), [2,0])

    def test_prediction_tie_preserves_original_tiebreak(self):
        from research.relative_rank import rank_order
        self.assertEqual(rank_order([1.,2.,2.], [.1,.1,.1], ('a','b','c'), [2,0,1]), [1,2,0])

if __name__ == '__main__': unittest.main()

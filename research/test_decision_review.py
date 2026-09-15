"""Accounting attribution must reconcile; it never represents a counterfactual."""
import unittest
import numpy as np
import pandas as pd
from research import decision_review as review


def daily(exposure, carry, execution=0.):
    return pd.DataFrame({'opening_exposure': np.atleast_1d(exposure),
                         'carry_rate': np.atleast_1d(carry),
                         'execution_rate': np.atleast_1d(execution)})


class GapDecompositionTests(unittest.TestCase):
    def test_same_accounts_have_zero_gap(self):
        account = daily([1., .5], [.1, -.03], [-.002, -.001])
        result = review.decompose(account, account)
        self.assertTrue(np.all(result.to_numpy() == 0.))

    def test_cash_participation_is_not_stock_selection(self):
        result = review.decompose(daily(.5, .05), daily(1., .1))
        self.assertAlmostEqual(result.participation_log.sum(), np.log(1.1/1.05))
        self.assertAlmostEqual(result.selection_log.sum(), 0.)
        self.assertAlmostEqual(result.execution_log.sum(), 0.)

    def test_same_exposure_different_holdings_is_selection(self):
        result = review.decompose(daily(1., .02), daily(1., .1))
        self.assertAlmostEqual(result.selection_log.sum(), np.log(1.1/1.02))
        self.assertAlmostEqual(result.participation_log.sum(), 0.)

    def test_actual_execution_effect_is_kept_separate(self):
        result = review.decompose(daily(1., .05, -.01), daily(1., .05, -.001))
        self.assertAlmostEqual(result.execution_log.sum(), np.log(1.049/1.04))
        self.assertAlmostEqual(result.selection_log.sum(), 0.)

    def test_offsetting_components_and_zero_total_are_finite(self):
        result = review.decompose(daily(.5, .05, -.01), daily(1., .04))
        self.assertTrue(np.isfinite(result.to_numpy()).all())
        self.assertAlmostEqual(result.total_log.sum(), 0.)
        self.assertAlmostEqual(result[['participation_log','selection_log','execution_log']].to_numpy().sum(),0.)

    def test_cash_only_reference_is_well_defined(self):
        result = review.decompose(daily(1., -.05), daily(0., 0.))
        self.assertAlmostEqual(result.selection_log.sum(), -np.log(.95))
        self.assertAlmostEqual(result.participation_log.sum(), 0.)

    def test_mismatched_dates_fail_instead_of_implicit_alignment(self):
        candidate, reference = daily(1., .1), daily(1., .1)
        reference.index = [1]
        with self.assertRaises(ValueError):
            review.decompose(candidate, reference)

    def test_nonpositive_nav_or_invalid_exposure_is_rejected(self):
        for bad in (daily(1., -1.), daily(-1., .1), daily(1., float('nan'))):
            with self.assertRaises(ValueError):
                review.decompose(bad, daily(1., .1))


class DecisionIdentityTests(unittest.TestCase):
    def test_tiny_weight_gap_does_not_excuse_different_reason(self):
        self.assertFalse(review.decision_matches(8e-17, .5, .5, 'original', 'original|extra'))

    def test_identical_reason_cap_and_weight_tolerance_match(self):
        self.assertTrue(review.decision_matches(8e-17, .5, .5, 'original', 'original'))

    def test_cap_mismatch_and_material_weight_mismatch_fail(self):
        self.assertFalse(review.decision_matches(0., .5, .50000000000001, 'x', 'x'))
        self.assertFalse(review.decision_matches(2e-13, .5, .5, 'x', 'x'))


class PairedScreenTests(unittest.TestCase):
    def rows(self):
        return [dict(scope=s, control=dict(wealth=2.,max_drawdown=.2,orders=10),
                     treatment=dict(wealth=2.1,max_drawdown=.2,orders=11))
                for s in ('union','chatgpt_5','joint_optical_leader_removal')]

    def test_all_scopes_must_pass_without_average_compensation(self):
        rows=self.rows();rows[1]['treatment']['wealth']=1.9
        self.assertFalse(review.paired_screen(rows)['advance'])
        rows=self.rows();rows[2]['treatment']['max_drawdown']=.21
        self.assertFalse(review.paired_screen(rows)['advance'])

    def test_unchanged_candidate_is_not_improvement(self):
        rows=self.rows()
        for r in rows:r['treatment']=r['control'].copy()
        self.assertFalse(review.paired_screen(rows)['advance'])

    def test_joint_improvement_is_only_research_screen(self):
        result=review.paired_screen(self.rows())
        self.assertTrue(result['advance'])
        self.assertEqual(result['economic_acceptance'],'UNVERIFIED')

    def test_nonfinite_metric_cannot_be_screened_as_success(self):
        rows=self.rows();rows[0]['treatment']['wealth']=float('nan')
        with self.assertRaises(ValueError):review.paired_screen(rows)

    def test_missing_or_duplicate_scope_is_rejected(self):
        with self.assertRaises(ValueError):review.paired_screen(self.rows()[:2])
        rows=self.rows();rows[-1]['scope']='union'
        with self.assertRaises(ValueError):review.paired_screen(rows)


if __name__ == '__main__': unittest.main()

"""Small report checks: invalid coverage cannot become a comparison victory."""
import unittest
import pandas as pd
from research import reproduce

class ReproductionTests(unittest.TestCase):
    def test_window_counts_first_session_against_previous_close(self):
        nav = pd.Series([100., 80., 90.], index=pd.date_range('2023-01-03', periods=3))
        got = reproduce.window_metrics(nav, '2023-01-04', '2023-01-05', initial=100.)
        self.assertAlmostEqual(got['wealth'], .9)
        self.assertAlmostEqual(got['max_drawdown'], .2)

    def test_empty_window_does_not_claim_zero_loss(self):
        nav = pd.Series([100.], index=pd.to_datetime(['2023-01-03']))
        self.assertEqual(reproduce.window_metrics(nav, '2024-01-01', None)['status'], 'NO_COVERAGE')

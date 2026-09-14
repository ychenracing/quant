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

    def test_unavailable_native_preserves_every_planned_case_without_metrics(self):
        import json
        from pathlib import Path
        catalog = json.loads(Path(reproduce.__file__).with_name('catalog.json').read_text())
        self.assertTrue(hasattr(reproduce, 'unavailable_native_cases'), 'missing explicit unavailable-case accounting')
        outcomes = reproduce.unavailable_native_cases(catalog)
        self.assertEqual(len(outcomes), 27)
        self.assertEqual(len({(r['reference'], r['pool']) for r in outcomes}), 27)
        self.assertEqual(sum(r['pool'] == 'chatgpt_5' for r in outcomes), 4)
        self.assertTrue(all(r['status'] == 'REFERENCE_CHECKOUT_UNAVAILABLE' for r in outcomes))
        self.assertTrue(all('wealth' not in r and 'max_drawdown' not in r for r in outcomes))

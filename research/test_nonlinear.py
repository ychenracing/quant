"""Optional research-model correctness; not a mandatory numerical CI matrix."""
import unittest
import numpy as np
from test_core import sample_market


class NonlinearTests(unittest.TestCase):
    def module(self):
        try:
            from research import nonlinear
            return nonlinear
        except ImportError as exc:self.fail(f'separate return and downside models missing: {exc}')

    def test_forecasts_and_fit_endpoints_are_prefix_causal(self):
        n=self.module();m=sample_market(3,120);p=n.Parameters(horizon=20)
        a=n.forecast(m,p);b=n.forecast(m.prefix(m.calendar[99]),p)
        np.testing.assert_allclose(a.expected[:100],b.expected,rtol=0,atol=1e-12)
        np.testing.assert_allclose(a.tail[:100],b.tail,rtol=0,atol=1e-12)
        self.assertEqual([f for f in a.fits if f['session']<100],b.fits)
        self.assertTrue(a.fits)
        self.assertTrue(all(f['last_label_session']<=f['session'] for f in a.fits))

    def test_excluded_name_changes_cannot_affect_forecast(self):
        n=self.module();m=sample_market(3,100);p=n.Parameters(horizon=20)
        only=m.subset(m.symbols[:1]);a=n.forecast(only,p)
        # Supplied universe is the whole permitted model-input boundary.
        self.assertEqual(a.symbols,only.symbols)
        self.assertEqual(a.expected.shape,(len(m.calendar),1))
        self.assertTrue(np.isfinite(a.tail).all())
        self.assertTrue(((a.tail>=0)&(a.tail<=1)).all())

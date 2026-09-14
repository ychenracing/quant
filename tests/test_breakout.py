import importlib.util
import unittest
import numpy as np

class BreakoutTests(unittest.TestCase):
    def test_research_module_exists(self):
        self.assertIsNotNone(importlib.util.find_spec('research.breakout'))

    def test_confirmed_units_set_cost_not_unfilled_targets(self):
        from research.breakout import Ownership
        state=Ownership(2)
        state.observe(np.array([0.,0.]),np.array([10.,20.]),np.array([11.,21.]))
        np.testing.assert_equal(state.cost,[0.,0.])
        state.observe(np.array([50.,0.]),np.array([10.,20.]),np.array([11.,21.]))
        np.testing.assert_allclose(state.cost,[10.,0.])
        state.observe(np.array([50.,0.]),np.array([15.,20.]),np.array([16.,21.]))
        np.testing.assert_allclose(state.cost,[10.,0.])
        np.testing.assert_allclose(state.peak,[16.,0.])

    def test_liquidation_intent_survives_partial_exit(self):
        from research.breakout import Ownership
        s=Ownership(1)
        s.observe(np.array([50.]),np.array([10.]),np.array([10.]))
        s.liquidating[0]=True
        s.observe(np.array([10.]),np.array([9.]),np.array([9.]))
        self.assertTrue(s.liquidating[0])
        s.observe(np.array([0.]),np.array([9.]),np.array([9.]))
        self.assertFalse(s.liquidating[0]);self.assertEqual(s.cost[0],0.)

    def test_stop_ratchets_and_never_widens_when_volatility_rises(self):
        from research.breakout import Ownership
        s=Ownership(1)
        s.observe(np.array([10.]),np.array([100.]),np.array([130.]))
        a=s.stops(np.array([3.]),2.)[0]
        b=s.stops(np.array([9.]),2.)[0]
        self.assertEqual(a,b)

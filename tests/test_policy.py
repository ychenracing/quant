"""Small integration contract for independent, fill-aware close policies."""
import unittest
import numpy as np
import pandas as pd
from test_core import sample_market
from techquant.engine import run


class PolicyInterfaceTests(unittest.TestCase):
    def test_policy_observes_fills_retains_units_and_has_fresh_run_state(self):
        try:
            from techquant.policy import CloseDecision
        except ImportError as exc:
            self.fail(f'explicit close-policy interface missing: {exc}')
        observations = []
        class Hold:
            def __init__(self, market, config):
                self.shape = len(market.symbols)
            def decide(self, close):
                observations.append(close)
                weights = np.full(self.shape, .8 / self.shape) if close.session == 0 else close.weights
                return CloseDecision(weights, 'OWN_ACTUAL_UNITS')
            def identity(self):
                return {'name': 'synthetic_hold'}
        m = sample_market(1, 50)
        a, b = run(m, policy_factory=Hold), run(m, policy_factory=Hold)
        self.assertEqual(len([o for o in a.orders if o['status'] == 'FILLED']), 1)
        self.assertEqual(a.orders, b.orders)
        pd.testing.assert_frame_equal(a.equity, b.equity)
        self.assertEqual(observations[0].units[0], 0)
        self.assertGreater(observations[1].units[0], 0)
        self.assertFalse(observations[1].weights.flags.writeable)
        self.assertEqual(a.metadata['policy'], {'name': 'synthetic_hold'})

    def test_policy_cannot_bypass_funding_shape_or_conflicting_inputs(self):
        try:
            from techquant.policy import CloseDecision
        except ImportError as exc:
            self.fail(f'explicit close-policy interface missing: {exc}')
        m = sample_market(1, 20)
        class Bad:
            def __init__(self, market, config): pass
            def decide(self, close): return CloseDecision(np.array([1.1]), 'BAD')
            def identity(self): return {'name': 'bad'}
        with self.assertRaises(ValueError):
            run(m, policy_factory=Bad)
        with self.assertRaises(ValueError):
            run(m, policy_factory=Bad, benchmark='buy_hold')
        with self.assertRaises(ValueError):
            run(m, policy_factory=Bad, targets=pd.DataFrame(0., index=m.calendar, columns=m.symbols))

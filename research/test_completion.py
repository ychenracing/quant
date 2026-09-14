"""The correction study delegates four fixed controls without adding behavior."""
import importlib
import importlib.util
import unittest
import pandas as pd
from test_core import sample_market
from techquant.engine import run


class CompletionStudyTests(unittest.TestCase):
    def test_registered_controls_match_their_direct_corrected_owners(self):
        self.assertIsNotNone(importlib.util.find_spec('research.completion'))
        m = importlib.import_module('research.completion')
        self.assertEqual([p.control for p in m.grid()], ['price','support','funded','cushion'])
        for bad in ('', None, 'forecast'):
            with self.assertRaises(ValueError): m.Parameters(bad)
        market=sample_market(3,130)
        for p in m.grid():
            owner=m.Owner(market,p)
            wrapped=run(market,policy_factory=lambda x,c:m.Owner(x,p))
            direct=run(market,policy_factory=lambda x,c:m.control_owner(x,p.control))
            pd.testing.assert_frame_equal(wrapped.equity,direct.equity)
            pd.testing.assert_frame_equal(wrapped.targets,direct.targets)
            self.assertEqual(wrapped.orders,direct.orders)
            self.assertEqual(owner.identity()['control'],p.control)
        identity=m.Study('completion').identity()
        for name in ('completion_contract.json','coherent.py','funded_risk.py','support_budget.py','recovery_memory.py'):
            self.assertIn(name,identity['dependencies'])
        with self.assertRaises(ValueError):m.Study('other')


if __name__=='__main__':unittest.main()

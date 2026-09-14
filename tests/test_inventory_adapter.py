"""Explicit inventory adapter keeps causal close allocation and real holdings."""
import unittest
import numpy as np
from test_core import sample_market
from techquant.engine import run
from research.admission import Owner, Parameters
from research.inventory_intent import UnitIntent


class InventoryAdapterTests(unittest.TestCase):
    def test_close_units_and_prefix_match_and_removed_names_are_absent(self):
        market = sample_market(3, 150)
        def factory(m, cfg): return UnitIntent(m, Owner(m, Parameters()))
        full = run(market, policy_factory=factory)
        prefix = run(market.prefix(str(market.calendar[100].date())), policy_factory=factory)
        np.testing.assert_allclose(full.equity.nav.iloc[:101], prefix.equity.nav)
        np.testing.assert_allclose(full.targets.iloc[:101], prefix.targets)
        reduced = market.subset(market.symbols[:2])
        owner = factory(reduced, None)
        self.assertEqual(owner.identity()['underlying']['data_sha256'], reduced.fingerprint())
        self.assertEqual(owner.prices.shape[1], 2)
        self.assertTrue((full.equity.cash >= 0).all())

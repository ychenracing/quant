"""Shock liquidation survives blocked execution and recovery is price-driven."""
import unittest
import numpy as np
import pandas as pd
from test_core import sample_market
from techquant.policy import CloseObservation
from techquant.engine import run


class ShockOwnershipTests(unittest.TestCase):
    def module(self):
        try:
            from research import shock_ownership
            return shock_ownership
        except ImportError as exc:self.fail(f'shock ownership missing: {exc}')

    def test_unfilled_liquidation_survives_recovery_and_uses_no_phantom_cash(self):
        s=self.module();m=sample_market(1,100);p=s.Parameters()
        owner=s.Owner(m,p);owner.healthy_market[:]=True;owner.market_shock[:]=False
        def obs(i,nav,units):
            return CloseObservation.from_inventory(i,str(m.calendar[i].date()),nav,nav*(0 if units else 1),
                                                  np.array([units]),np.array([1. if units else 0.]))
        owner.decide(obs(40,1000.,10.))
        self.assertEqual(owner.decide(obs(41,940.,10.)).weights[0],0.)
        for i in range(42,47):
            self.assertEqual(owner.decide(obs(i,1000.,10.)).weights[0],0.)
        # Healthy market recovery does not create a fill; actual exit still pending.
        self.assertTrue(owner.closed)
        owner.decide(obs(47,1000.,0.))
        self.assertFalse(owner.exit_pending.any())

    def test_entire_prefix_nav_orders_and_targets_are_causal(self):
        s=self.module();m=sample_market(3,100);p=s.Parameters()
        factory=lambda m,c:s.Owner(m,p)
        a=run(m,policy_factory=factory);b=run(m.prefix(m.calendar[69]),policy_factory=factory)
        pd.testing.assert_frame_equal(a.equity.iloc[:70],b.equity)
        pd.testing.assert_frame_equal(a.targets.iloc[:70],b.targets)
        self.assertEqual([o for o in a.orders if o['date']<=str(m.calendar[69].date())],b.orders)

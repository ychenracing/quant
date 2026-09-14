"""Permanent drawdown reference and actual-fill capital protection."""
import unittest
import numpy as np
from test_core import sample_market
from techquant.policy import CloseDecision,CloseObservation


class CushionTests(unittest.TestCase):
    def module(self):
        try:
            from research import cushion
            return cushion
        except ImportError as exc:
            self.fail(f'capital cushion policy missing: {exc}')

    def test_permanent_peak_and_blocked_reduction_survive_rebound(self):
        c=self.module();market=sample_market(1,80)
        owner=c.Owner(market,c.Parameters(.2,5.))
        owner.price[:]=100.
        class Keep:
            admit=np.ones(80,dtype=bool)
            def decide(self,o):return CloseDecision(o.weights,'KEEP')
        owner.inner=Keep()
        def obs(i,nav,units,cash):
            return CloseObservation.from_inventory(i,str(market.calendar[i].date()),nav,cash,
                    np.array([units]),np.array([units*100/nav]))
        owner.decide(obs(30,1000.,8.,200.))
        cut=owner.decide(obs(31,900.,8.,100.))
        self.assertLess(cut.weights[0],.6)
        # A bounce with no actual fill must not cancel yesterday's reduction.
        bounced=owner.decide(obs(32,1000.,8.,200.))
        self.assertLessEqual(bounced.weights[0],cut.weights[0]*.9+1e-12)
        self.assertEqual(owner.peak,1000.)
        owner.decide(obs(33,850.,4.,450.))
        self.assertEqual(owner.peak,1000.)

    def test_restoration_cannot_spend_planned_sale_proceeds(self):
        c=self.module();market=sample_market(2,80)
        owner=c.Owner(market,c.Parameters(.2,5.));owner.price[:]=100.
        class SellAndKeep:
            admit=np.ones(80,dtype=bool)
            def decide(self,o):return CloseDecision(np.array([0.,o.weights[1]]),'EXIT_ONE')
        owner.inner=SellAndKeep()
        obs=CloseObservation.from_inventory(30,str(market.calendar[30].date()),1000.,100.,
                                           np.array([5.,4.]),np.array([.5,.4]))
        want=owner.decide(obs).weights
        self.assertEqual(want[0],0.)
        self.assertLessEqual(want[1],.5+1e-12)

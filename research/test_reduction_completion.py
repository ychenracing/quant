"""Protect the economic goal without letting declaration rounding move it."""
import unittest
import numpy as np
from test_core import sample_market
from techquant.data import Market
from techquant.execution import round_quantity
from techquant.policy import CloseObservation
from research import support_budget, test_coherent_residuals


class ReductionCompletionTests(unittest.TestCase):
    def owner(self, family, symbol):
        if family == 'coherent':
            helper, owner, observed = test_coherent_residuals.ResidualIntentTests().owner((symbol,))
            observed.entry[:] = False
            return owner, lambda i,u: helper.observation(i,[u],30000.-u*10.)
        m = sample_market(1,100)
        m = Market.from_frames({symbol: next(iter(m.frames.values()))},m.calendar,quality='synthetic')
        owner = support_budget.Owner(m,support_budget.Parameters())
        owner.features.entry[:] = False
        owner.risk.cap = 1.
        owner.risk.update = lambda *args:(1.,'CONTROLLED_RISK_OBSERVATION')
        def observe(i,u):
            price = owner.features.close[i,0]
            cash = 50000.-u*price
            return CloseObservation.from_inventory(i,str(m.calendar[i].date()),50000.,cash,np.array([u]),np.array([u*price/50000.]))
        return owner, observe

    def test_rounding_does_not_redefine_the_completion_goal(self):
        for family in ('coherent','support'):
            for symbol in ('sz300100','sh688001','bj920001'):
                with self.subTest(family=family,symbol=symbol):
                    owner, observe = self.owner(family,symbol)
                    owner.reduction_ceiling[0] = 999.9
                    first=owner.decide(observe(40,1000.))
                    self.assertLess(first.unit_targets[0],999.9)
                    self.assertEqual(owner.reduction_ceiling[0],999.9)
                    # No fill and a slightly different quoted conversion cannot
                    # ratchet the economic target into an additional sale.
                    owner.raw_per_unit[41,0] *= 1.0001
                    owner.decide(observe(41,1000.))
                    self.assertEqual(owner.reduction_ceiling[0],999.9)

    def test_observed_fill_that_met_goal_retires_protection_not_the_whole_rounded_request(self):
        for family in ('coherent','support'):
            for symbol in ('sz300100','sh688001','bj920001'):
                with self.subTest(family=family,symbol=symbol):
                    owner, observe = self.owner(family,symbol)
                    owner.reduction_ceiling[0] = 999.9
                    first=owner.decide(observe(40,1000.))
                    # Exactly the unchanged engine conversion and lot rule.
                    # Rounded adjusted and raw OHLC can have different ratios
                    # at the next open even without an economic position change.
                    factor=1.0001
                    raw=round_quantity(symbol,(1000.-first.unit_targets[0])*factor)
                    actual=1000.-raw/factor
                    self.assertLessEqual(actual,999.9)
                    self.assertGreater(actual,first.unit_targets[0])
                    next_decision=owner.decide(observe(41,actual))
                    self.assertEqual(next_decision.unit_targets[0],actual)
                    self.assertTrue(np.isinf(owner.reduction_ceiling[0]))

    def test_unfilled_original_goal_is_not_forgiven(self):
        for family in ('coherent','support'):
            with self.subTest(family=family):
                owner, observe=self.owner(family,'sz300100')
                owner.reduction_ceiling[0]=999.9
                owner.decide(observe(40,1000.))
                actual=999.95
                still_pending=owner.decide(observe(41,actual))
                self.assertLessEqual(still_pending.unit_targets[0],999.9)
                self.assertLess(still_pending.unit_targets[0],actual)
                self.assertEqual(owner.reduction_ceiling[0],999.9)


if __name__=='__main__':unittest.main()

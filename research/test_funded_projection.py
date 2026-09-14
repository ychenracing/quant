"""The funded ceiling must protect the book after concentration reductions."""
import unittest
import numpy as np
from test_core import sample_market
from techquant.policy import CloseObservation
from research.funded_risk import Owner, Parameters


class FundedProjectionTests(unittest.TestCase):
    def test_strict_budget_uses_the_post_concentration_risk_mix(self):
        owner = Owner(sample_market(2, 140), Parameters(False))
        def warning(*args):
            owner.risk.warnings.cap = 1.
            return 1., 'TREND_OPEN'
        owner.risk.warnings.update = warning
        i, nav = 40, 2_000_000.
        price = owner.features.close[i]
        units = np.array([.81, .19])*nav/price
        owner.previous_units = units.copy()
        owner.stop = price*(1-np.array([.02, .80]))
        owner.peak = price.copy()
        owner.support[:] = 0.
        owner.atr[:] = owner.features.close
        owner.ready[:] = True
        owner.features.exit[:] = False
        cash = nav-float(units@price)
        observation = CloseObservation.from_inventory(
            i, str(owner.market.calendar[i].date()), nav, cash, units, units*price/nav)
        decision = owner.decide(observation)
        risk = float(decision.unit_targets@np.maximum(price-owner.stop, .02*price))
        self.assertLessEqual(risk, .1*nav+1e-7)
        self.assertTrue(np.all(decision.unit_targets <= units))


if __name__ == '__main__':
    unittest.main()

"""An unfinished restoration must not strand an independently fundable vacancy."""
import unittest
import numpy as np
from research import test_coherent


class PendingRestorationVacancyTests(unittest.TestCase):
    def fixture(self):
        helper = test_coherent.CoherentTests()
        owner, _, observed = helper.owner(3, caps=[1., 1.])
        owner.risk.cap = .5
        observed.entry[:, 2] = False
        first = helper.decision(owner, 1, [50000., 50000., 0.], 1000000.)
        np.testing.assert_allclose(first.unit_targets, [100000., 100000., 0.])
        observed.entry[2, 2] = True
        return helper, owner, observed

    def test_inadmissible_restoration_does_not_reserve_actual_cash(self):
        helper, owner, observed = self.fixture()
        observed.entry[2, :2] = False
        decision = helper.decision(owner, 2, [60000., 50000., 0.], 900000.)
        np.testing.assert_allclose(decision.unit_targets, [60000., 50000., 90000.])
        self.assertIsNotNone(owner.restoration_targets)
        self.assertEqual(owner.restoration_targets[0], 100000.)

    def test_eligible_pending_buy_is_funded_before_the_vacancy(self):
        helper, owner, observed = self.fixture()
        observed.entry[2, 1] = False
        decision = helper.decision(owner, 2, [60000., 50000., 0.], 900000.)
        np.testing.assert_allclose(decision.unit_targets, [100000., 50000., 50000.])
        requested = (decision.unit_targets - [60000., 50000., 0.]).clip(min=0)
        self.assertLessEqual(float(requested.sum()) * 10., 900000. + 1e-7)

    def test_material_pending_buys_leave_no_virtual_cash_for_a_vacancy(self):
        helper, owner, _ = self.fixture()
        decision = helper.decision(owner, 2, [60000., 50000., 0.], 900000.)
        np.testing.assert_allclose(decision.unit_targets, [100000., 100000., 0.])


if __name__ == '__main__':
    unittest.main()

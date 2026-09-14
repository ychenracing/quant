"""Admission and actual-fill invariants of the preregistered coherent owner."""
import importlib
import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from test_core import sample_market
from techquant.policy import CloseObservation
from research.pathwise import Prediction


class CoherentTests(unittest.TestCase):
    def owner(self, size=6, ranking='forecast', caps=None):
        self.assertIsNotNone(importlib.util.find_spec('research.coherent'),
                             'the preregistered owner must be implemented')
        module = importlib.import_module('research.coherent')
        m = sample_market(size, 60)
        shape = (len(m.calendar), size)
        p = Prediction(m.symbols, m.fingerprint(), 20,
            np.tile(np.arange(size, 0, -1) * .01, (shape[0], 1)),
            np.full(shape, .1), np.ones(shape, dtype=bool), np.full(shape, 10.),
            np.full(shape, 9.), np.full(shape, 9.), np.full(shape, 9.),
            np.full(shape, .01), np.zeros(shape), [], np.full((*shape, 3), 1/3))
        s = SimpleNamespace(entry=np.ones(shape, dtype=bool),
            exit=np.zeros(shape, dtype=bool), market=np.ones(shape[0], dtype=bool))
        features = SimpleNamespace(score=np.tile(np.arange(1, size+1), (shape[0], 1)))
        with patch.object(module, 'signals', return_value=s), patch.object(module, 'build_features', return_value=features):
            owner = module.Owner(m, module.Parameters(ranking=ranking), prediction=p)
        owner.risk.cap = 1.
        sequence = iter(caps or [1.] * shape[0])
        def update(i, f, navs, cfg):
            owner.risk.cap = next(sequence)
            return owner.risk.cap, 'CONTROLLED_RISK_OBSERVATION'
        owner.risk.update = update
        return owner, p, s

    def observation(self, session, units, cash):
        units = np.asarray(units, dtype=float)
        nav = float(cash + units.sum() * 10.)
        return CloseObservation.from_inventory(session, str(session), nav, cash, units, units * 10. / nav)

    def decision(self, owner, session, units, cash):
        o = self.observation(session, units, cash)
        d = owner.decide(o)
        d.validated_weights(len(units))
        d.validated_unit_targets(np.full(len(units), 10.), o.nav)
        return d

    def test_admission_precedes_slot_truncation_and_funding(self):
        owner, _, s = self.owner()
        s.entry[:, :4] = False
        d = self.decision(owner, 1, [0.] * 6, 1000.)
        np.testing.assert_array_equal(d.unit_targets[:4], 0.)
        self.assertTrue((d.unit_targets[4:] > 0).all())

    def test_bounded_utility_is_ranking_not_absolute_entry_return(self):
        owner, p, _ = self.owner(2)
        p.expected[:] = .001
        self.assertGreater(self.decision(owner, 1, [0., 0.], 1000.).weights.sum(), 0.)

    def test_retained_inventory_is_not_daily_rebalanced(self):
        owner, _, s = self.owner(2)
        s.entry[:, 1] = False
        d = self.decision(owner, 1, [50., 0.], 500.)
        np.testing.assert_array_equal(d.unit_targets, [50., 0.])

    def test_reduction_survives_partial_fill_and_recovery_without_halving_again(self):
        owner, _, _ = self.owner(2, caps=[.5, 1.])
        first = self.decision(owner, 1, [100., 0.], 0.)
        second = self.decision(owner, 2, [75., 0.], 250.)
        self.assertEqual(first.unit_targets[0], 50.)
        self.assertEqual(second.unit_targets[0], 50.)
        self.assertEqual(second.unit_targets[1], 0.)

    def test_full_exit_blocks_replacement_until_actual_liquidation(self):
        owner, p, _ = self.owner(2)
        p.tail[1, 0] = .8; p.ret1[1, 0] = -.02
        first = self.decision(owner, 1, [50., 0.], 500.)
        second = self.decision(owner, 2, [50., 0.], 500.)
        np.testing.assert_array_equal(first.unit_targets, [0., 0.])
        np.testing.assert_array_equal(second.unit_targets, [0., 0.])

    def test_restoration_funds_intact_names_and_persists_after_partial_fill(self):
        owner, _, _ = self.owner(4, caps=[1., 1., .5])
        owner.risk.cap = .5
        first = self.decision(owner, 1, [10.] * 4, 400.)
        second = self.decision(owner, 2, [15., 10., 10., 10.], 350.)
        np.testing.assert_allclose(first.unit_targets, [20.] * 4)
        np.testing.assert_allclose(second.unit_targets, [20.] * 4)
        third = self.decision(owner, 3, [15., 10., 10., 10.], 350.)
        self.assertLessEqual(third.weights.sum(), .5 + 1e-12)
        self.assertTrue((third.unit_targets <= [15., 10., 10., 10.]).all())

    def test_new_cap_increase_is_not_lost_when_prior_target_fills_today(self):
        owner, _, _ = self.owner(4, caps=[.5, 1.])
        owner.risk.cap = 0.
        first = self.decision(owner, 1, [0.] * 4, 800.)
        np.testing.assert_allclose(first.unit_targets, [10.] * 4)
        second = self.decision(owner, 2, [10.] * 4, 400.)
        np.testing.assert_allclose(second.unit_targets, [20.] * 4)

    def test_new_cap_increase_can_extend_an_unfilled_restoration(self):
        owner, _, _ = self.owner(4, caps=[.5, 1.])
        owner.risk.cap = 0.
        self.decision(owner, 1, [0.] * 4, 800.)
        second = self.decision(owner, 2, [9.] * 4, 440.)
        np.testing.assert_allclose(second.unit_targets, [20.] * 4)

    def test_market_gate_blocks_only_new_risk_not_intact_ownership(self):
        owner, _, s = self.owner(2)
        s.market[:] = False
        np.testing.assert_array_equal(self.decision(owner, 1, [50., 0.], 500.).unit_targets, [50., 0.])

    def test_price_and_forecast_candidates_use_the_declared_ranking(self):
        a, _, _ = self.owner(6, ranking='forecast')
        b, _, _ = self.owner(6, ranking='price')
        da = self.decision(a, 1, [0.] * 6, 1000.)
        db = self.decision(b, 1, [0.] * 6, 1000.)
        self.assertEqual(set(np.flatnonzero(da.unit_targets)), {0, 1, 2, 3})
        self.assertEqual(set(np.flatnonzero(db.unit_targets)), {2, 3, 4, 5})

    def test_same_session_cannot_advance_risk_or_readmission_twice(self):
        owner, _, _ = self.owner(1)
        self.decision(owner, 1, [0.], 1000.)
        with self.assertRaises(ValueError):
            self.decision(owner, 1, [0.], 1000.)

    def test_removed_symbol_forecast_is_not_accepted(self):
        owner, prediction, _ = self.owner(2)
        module = importlib.import_module('research.coherent')
        with self.assertRaises(ValueError):
            module.Owner(owner.market.subset([owner.market.symbols[0]]), module.Parameters(), prediction=prediction)


if __name__ == '__main__':
    unittest.main()

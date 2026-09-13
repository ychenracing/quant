"""Small, synthetic contracts for the preregistered retention/recovery hypothesis."""
from dataclasses import replace
import unittest

import numpy as np

from test_core import sample_market
from techquant.config import Config
from techquant.features import build_features
from techquant.strategy import RiskState, target_weights


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(max_positions=2, target_vol=.8)
        self.f = build_features(sample_market(), self.cfg)
        self.f = replace(self.f, entry=np.ones_like(self.f.entry),
                         exit=np.zeros_like(self.f.exit),
                         market_return=np.full(170, .001),
                         market_vol=np.full(170, .008),
                         shock_fraction=np.zeros(170))

    def test_unchanged_membership_does_not_rebalance_on_score_noise(self):
        score = np.tile([10., 1., .01], (170, 1))
        f = replace(self.f, score=score)
        current = np.array([.35, .50, 0.])
        want, reasons = target_weights(100, f, current, self.cfg, cap=1., rebalance=True)
        np.testing.assert_allclose(want, current)
        self.assertIn('RETAIN_INTACT_MEMBERSHIP', reasons)

    def test_confirmed_recovery_is_half_then_full_not_one_jump(self):
        state = RiskState()
        caps = [state.update(i, self.f, [100.], self.cfg)[0] for i in range(90, 96)]
        self.assertEqual(caps, [0., 0., .5, .5, .5, 1.])

    def test_selective_recovery_requires_nonnegative_return_and_enough_entries(self):
        f = replace(self.f, weak=np.ones(170, dtype=bool), breadth=np.full(170, .3),
                    market_return=np.full(170, -.001))
        state = RiskState()
        for i in range(90, 96):
            state.update(i, f, [100.], self.cfg)
        self.assertEqual(state.cap, 0.)
        entry = np.zeros_like(f.entry); entry[:, 0] = True
        f = replace(f, market_return=np.full(170, .001), entry=entry)
        for i in range(100, 106):
            state.update(i, f, [100.], self.cfg)
        self.assertEqual(state.cap, 0.)

    def test_accumulated_market_loss_is_detected_without_one_day_crash(self):
        returns = self.f.market_return.copy(); returns[98:101] = -.02
        f = replace(self.f, market_return=returns, breadth=np.full(170, .3))
        cap, reason = RiskState(cap=1.).update(100, f, [100., 100.], self.cfg)
        self.assertEqual(cap, 0.)
        self.assertEqual(reason, 'ACCUMULATED_MARKET_SHOCK')

    def test_retention_cannot_disable_sector_or_volatility_limits(self):
        f = replace(self.f, sectors=('group_a', 'group_a', 'group_b'))
        current = np.array([.42, .40, 0.])
        want, _ = target_weights(100, f, current, self.cfg, cap=1., rebalance=False)
        self.assertLessEqual(want[:2].sum(), self.cfg.sector_cap + 1e-12)
        f = replace(self.f, vol=np.full_like(self.f.vol, .08))
        want, _ = target_weights(100, f, current, self.cfg, cap=1., rebalance=False)
        sigma = float(np.dot(want, f.vol[100])) * np.sqrt(252)
        self.assertLessEqual(sigma, self.cfg.target_vol + 1e-12)

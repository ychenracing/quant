"""Regression cases for execution intent and asymmetric risk-cycle recovery."""
import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

import test_core as helpers
from test_core import sample_market


class ExecutionPolicyTests(unittest.TestCase):
    setUp = helpers.CoreTests.setUp

    def test_completed_risk_episode_does_not_chatter_on_old_loss(self):
        from dataclasses import replace
        from techquant.features import build_features
        from techquant.strategy import RiskState
        cfg = self.config.Config()
        f = build_features(sample_market(), cfg)
        f = replace(f, market_return=np.full(170, .001), shock_fraction=np.zeros(170))
        state = RiskState(cap=1.)
        state.update(90, f, [100.], cfg)
        cap, _ = state.update(91, f, [100., 87.], cfg)
        self.assertEqual(cap, 0.)
        history = [100., 87.]
        for i in range(92, 92 + 2 * cfg.recovery):
            history.append(87.)
            cap, _ = state.update(i, f, history, cfg)
        self.assertEqual(cap, 1.)
        history.append(86.)
        cap, _ = state.update(98, f, history, cfg)
        self.assertEqual(cap, 1.)


    def test_progressive_warmup_uses_no_pre_2023_history(self):
        m = sample_market(3, 100)
        r = self.engine.run(m)
        first = next(o for o in r.orders if o['status'] == 'FILLED')
        i = m.calendar.get_loc(pd.Timestamp(first['signal_date']))
        self.assertGreaterEqual(i, self.config.Config().fast)
        self.assertLess(i, self.config.Config().slow)


    def test_weak_market_warning_does_not_veto_intact_leaders(self):
        from dataclasses import replace
        from techquant.features import build_features
        from techquant.strategy import RiskState
        cfg = self.config.Config()
        f = build_features(sample_market(), cfg)
        f = replace(f, breadth=np.full(170, .3), weak=np.ones(170, dtype=bool),
                    market_dd=np.full(170, .03))
        state = RiskState(cap=1.)
        cap, _ = state.update(100, f, [100.] * 101, cfg)
        self.assertEqual(cap, 1.)
        f = replace(f, market_return=np.full(170, -.10), shock_fraction=np.ones(170))
        cap, reason = state.update(101, f, [100.] * 102, cfg)
        self.assertEqual(cap, 0.)
        self.assertEqual(reason, 'CROSS_SECTION_SHOCK')


    def test_risk_budget_drift_is_not_a_new_emergency(self):
        from techquant.strategy import target_weights
        from techquant.features import build_features
        m = sample_market()
        cfg = self.config.Config()
        f = build_features(m, cfg)
        current = np.array([.26, 0., 0.])
        w, _ = target_weights(100, f, current, cfg, cap=.25, rebalance=False)
        np.testing.assert_equal(w, current)
        w, _ = target_weights(100, f, current, cfg, cap=.25, rebalance=False,
                              risk_reduction=True)
        self.assertAlmostEqual(w.sum(), .25)


    def test_buy_hold_never_rebalances_a_winner(self):
        m = sample_market(2, 90)
        f = {s: x.copy() for s, x in m.frames.items()}
        s = m.symbols[0]
        # A persistent but tradable overnight rally must not turn hold into sell.
        for col in ('open', 'high', 'low', 'close', 'raw_open', 'raw_close'):
            f[s].loc[m.calendar[20]:, col] *= 1.15
        r = self.engine.run(self.data.Market.from_frames(f, m.calendar,
                                                        quality='synthetic'), benchmark='buy_hold')
        self.assertFalse(any(o['side'] == 'SELL' for o in r.orders))


    def test_small_target_reduction_is_not_suppressed(self):
        m = sample_market(1, 90)
        w = pd.DataFrame(.5, index=m.calendar, columns=m.symbols)
        w.iloc[30:] = .495
        r = self.engine.run(m, targets=w)
        self.assertTrue(any(o['side'] == 'SELL' and o['status'] == 'FILLED'
                            and o['signal_date'] == str(m.calendar[30].date())
                            for o in r.orders))


    def test_recovery_can_reopen_a_reduced_risk_budget(self):
        from dataclasses import replace
        from techquant.features import build_features
        from techquant.strategy import RiskState
        cfg = self.config.Config()
        f = build_features(sample_market(), cfg)
        f = replace(f, breadth=np.full(170, .1), weak=np.ones(170, dtype=bool),
                    market_dd=np.full(170, .13), market_return=np.full(170, .001),
                    shock_fraction=np.zeros(170))
        state = RiskState()
        for i in range(100, 100 + cfg.recovery):
            state.update(i, f, [100.] * (i + 1), cfg)
        self.assertGreater(state.cap, 0.)
        self.assertLessEqual(state.cap, .5)

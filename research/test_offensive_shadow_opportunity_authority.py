from dataclasses import replace
import importlib
import importlib.util
import unittest

import numpy as np
import pandas as pd

from test_core import sample_market
from techquant.engine import run
from techquant.policy import CloseObservation
from research.trend_book import Owner as TrendOwner, Parameters as TrendParameters


def obs(owner, session, units, cash):
    units = np.asarray(units, dtype=float)
    marks = owner.parent.base.price_signals.price[session]
    values = units * marks
    nav = float(cash + values.sum())
    return CloseObservation.from_inventory(
        session, str(owner.market.calendar[session].date()), nav, cash, units, values / nav
    )


def force(owner, session, scores, entries):
    b = owner.parent.base
    features = b.features.score.copy()
    features[session] = np.asarray(scores, dtype=float)
    b.features = replace(b.features, score=features)
    p = b.price_signals
    ready = p.ready.copy(); ready[session] = True
    ema = p.ema20.copy(); ema[session] = np.where(np.isfinite(p.price[session]), p.price[session] * .9, 0)
    momentum = p.momentum5.copy(); momentum[session] = .1
    ret1 = p.ret1.copy(); ret1[session] = 0
    b.price_signals = replace(p, ready=ready, ema20=ema, momentum5=momentum, ret1=ret1)
    trend = b.trend
    entry = trend.entry.copy(); entry[session] = np.asarray(entries, dtype=bool)
    exit_ = trend.exit.copy(); exit_[session] = False
    market = trend.market.copy(); market[session] = True
    b.trend = replace(trend, entry=entry, exit=exit_, market=market)


class ShadowOpportunityAuthorityTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(
            importlib.util.find_spec('research.offensive_shadow_opportunity_authority'),
            'shadow opportunity authority implementation is absent',
        )
        return importlib.import_module('research.offensive_shadow_opportunity_authority')

    def prepared(self):
        m = self.module()
        owner = m.Owner(sample_market(6, 145), m.Parameters(True))
        units = np.zeros(6)
        force(owner, 100, [1.0, 3.0, 2.5, .4, .3, .2], [True, True, True, False, False, False])
        marks = owner.parent.base.price_signals.price[100]
        units[0] = 1000.0
        units[3] = 1000.0
        owner.parent.base.was_held[[0, 3]] = True
        owner.parent.base.owned_alpha_reference[0] = 2.0
        owner.parent.base.owned_alpha_reference[3] = .5
        owner.parent.base.last_session = 99
        return owner, units

    def test_control_is_exact_trend_book(self):
        m = self.module()
        market = sample_market(6, 145)
        parent = run(market, policy_factory=lambda current, cfg: TrendOwner(current, TrendParameters(2)))
        control = run(market, policy_factory=lambda current, cfg: m.Owner(current, m.Parameters(False)))
        pd.testing.assert_frame_equal(parent.equity, control.equity)
        pd.testing.assert_frame_equal(parent.targets, control.targets)
        self.assertEqual(parent.orders, control.orders)

    def test_first_full_book_pair_only_nominates_shadow_and_does_not_sell(self):
        owner, units = self.prepared()
        decision = owner.decide(obs(owner, 100, units, 0.0))
        self.assertGreater(decision.unit_targets[0], 0.0)
        self.assertGreater(decision.unit_targets[3], 0.0)
        self.assertEqual(owner.shadow_incumbent, 0)
        self.assertEqual(owner.shadow_challenger, 1)
        self.assertAlmostEqual(owner.shadow_reference, 3.0)
        self.assertTrue(any(r.get('action') == 'SHADOW_OPPORTUNITY_NOMINATION' for r in owner.trace))
        self.assertFalse(any(r.get('action') == 'SHADOW_AUTHORIZED_DISPLACEMENT' for r in owner.trace))

    def test_same_pair_needs_score_appreciation_before_authority(self):
        owner, units = self.prepared()
        owner.decide(obs(owner, 100, units, 0.0))
        force(owner, 101, [0.9, 3.0, 2.4, .4, .3, .2], [True, True, True, False, False, False])
        held = owner.decide(obs(owner, 101, units, 0.0))
        self.assertGreater(held.unit_targets[0], 0.0)
        force(owner, 102, [0.8, 3.2, 2.4, .4, .3, .2], [True, True, True, False, False, False])
        sold = owner.decide(obs(owner, 102, units, 0.0))
        self.assertEqual(sold.unit_targets[0], 0.0)
        self.assertTrue(any(r.get('action') == 'SHADOW_AUTHORIZED_DISPLACEMENT' for r in owner.trace))

    def test_changed_pair_resets_shadow_instead_of_inheriting_authority(self):
        owner, units = self.prepared()
        owner.decide(obs(owner, 100, units, 0.0))
        force(owner, 101, [0.9, 2.8, 3.4, .4, .3, .2], [True, True, True, False, False, False])
        decision = owner.decide(obs(owner, 101, units, 0.0))
        self.assertGreater(decision.unit_targets[0], 0.0)
        self.assertEqual(owner.shadow_challenger, 2)
        self.assertAlmostEqual(owner.shadow_reference, 3.4)
        self.assertTrue(any(r.get('action') == 'SHADOW_OPPORTUNITY_RESET' for r in owner.trace))


if __name__ == '__main__':
    unittest.main()

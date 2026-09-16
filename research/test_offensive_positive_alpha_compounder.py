from dataclasses import replace
import importlib
import importlib.util
import unittest

import numpy as np
import pandas as pd

from test_core import sample_market
from techquant.engine import run
from techquant.policy import CloseObservation
from research.offensive_campaign_peak_authority import Owner as ChampionOwner, Parameters as ChampionParameters


def base(owner):
    return owner.parent.parent.base


def observe(owner, session, units, cash=0.0):
    units = np.asarray(units, dtype=float)
    b = base(owner)
    marks = b.price_signals.price[session]
    values = units * marks
    nav = float(cash + values.sum())
    return CloseObservation.from_inventory(
        session,
        str(owner.market.calendar[session].date()),
        nav,
        cash,
        units,
        np.divide(values, nav, out=np.zeros_like(values), where=nav > 0),
    )


def force(owner, session, scores, *, exits=None, ret1=None, ready=None, entries=None):
    b = base(owner)
    feature_score = b.features.score.copy()
    feature_score[session] = np.asarray(scores, dtype=float)
    b.features = replace(b.features, score=feature_score)

    p = b.price_signals
    ready_values = p.ready.copy()
    ready_values[session] = True if ready is None else np.asarray(ready, dtype=bool)
    ema20 = p.ema20.copy()
    ema20[session] = np.where(np.isfinite(p.price[session]), p.price[session] * .9, 0)
    momentum5 = p.momentum5.copy(); momentum5[session] = .1
    one_day = p.ret1.copy(); one_day[session] = 0 if ret1 is None else np.asarray(ret1, dtype=float)
    b.price_signals = replace(p, ready=ready_values, ema20=ema20, momentum5=momentum5, ret1=one_day)

    trend = b.trend
    entry = trend.entry.copy()
    entry[session] = True if entries is None else np.asarray(entries, dtype=bool)
    exit_ = trend.exit.copy()
    exit_[session] = False if exits is None else np.asarray(exits, dtype=bool)
    market = trend.market.copy(); market[session] = True
    b.trend = replace(trend, entry=entry, exit=exit_, market=market)


class PositiveAlphaCompounderTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(
            importlib.util.find_spec('research.offensive_positive_alpha_compounder'),
            'positive alpha compounder implementation is absent',
        )
        return importlib.import_module('research.offensive_positive_alpha_compounder')

    def prepared(self):
        module = self.module()
        owner = module.Owner(sample_market(6, 145), module.Parameters(True))
        units = np.zeros(6)
        marks = base(owner).price_signals.price[100]
        units[0] = 1000.0
        units[3] = 1000.0
        base(owner).was_held[[0, 3]] = True
        base(owner).owned_alpha_reference[0] = 1.5
        base(owner).owned_alpha_reference[3] = .5
        base(owner).last_session = 99
        owner.parent.actual_was_held[[0, 3]] = True
        owner.parent.campaign_peak_close[[0, 3]] = marks[[0, 3]]
        return owner, units

    def test_disabled_is_exact_current_campaign_peak_champion(self):
        module = self.module()
        market = sample_market(6, 145)
        champion = run(market, policy_factory=lambda current, cfg: ChampionOwner(current, ChampionParameters(True)))
        control = run(market, policy_factory=lambda current, cfg: module.Owner(current, module.Parameters(False)))
        pd.testing.assert_frame_equal(champion.equity, control.equity)
        pd.testing.assert_frame_equal(champion.targets, control.targets)
        self.assertEqual(champion.orders, control.orders)

    def test_positive_alpha_forgives_slow_trend_exit_for_funded_campaign(self):
        owner, units = self.prepared()
        force(owner, 100, [1.2, .1, .1, .8, .1, .1], exits=[True, False, False, False, False, False])
        decision = owner.decide(observe(owner, 100, units))
        self.assertGreater(decision.unit_targets[0], 0.0)
        self.assertTrue(any(row.get('action') == 'POSITIVE_ALPHA_TREND_EXIT_FORGIVEN' for row in owner.trace))

    def test_nonpositive_alpha_is_self_failure_even_without_slow_trend_exit(self):
        owner, units = self.prepared()
        force(owner, 100, [0.0, .1, .1, .8, .1, .1], exits=[False, False, False, False, False, False])
        decision = owner.decide(observe(owner, 100, units))
        self.assertEqual(decision.unit_targets[0], 0.0)
        self.assertTrue(any(row.get('action') == 'NONPOSITIVE_ALPHA_SELF_FAILURE' for row in owner.trace))

    def test_acute_loss_remains_authoritative_with_positive_alpha(self):
        owner, units = self.prepared()
        force(
            owner,
            100,
            [1.2, .1, .1, .8, .1, .1],
            exits=[False, False, False, False, False, False],
            ret1=[-.09, 0, 0, 0, 0, 0],
        )
        decision = owner.decide(observe(owner, 100, units))
        self.assertEqual(decision.unit_targets[0], 0.0)
        self.assertFalse(any(row.get('action') == 'POSITIVE_ALPHA_TREND_EXIT_FORGIVEN' for row in owner.trace))


if __name__ == '__main__':
    unittest.main()

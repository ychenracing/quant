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


def observe(owner, session, units, cash=0.0):
    units = np.asarray(units, dtype=float)
    marks = owner.price_signals.price[session]
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


def force(owner, session, scores, entries, *, exits=None, ret1=None, ready=None):
    scores_ = owner.features.score.copy(); scores_[session] = np.asarray(scores, dtype=float)
    owner.features = replace(owner.features, score=scores_)
    p = owner.price_signals
    ready_ = p.ready.copy(); ready_[session] = True if ready is None else np.asarray(ready, dtype=bool)
    ema20 = p.ema20.copy(); ema20[session] = np.where(np.isfinite(p.price[session]), p.price[session] * .9, 0)
    momentum5 = p.momentum5.copy(); momentum5[session] = .1
    ret1_ = p.ret1.copy(); ret1_[session] = 0 if ret1 is None else np.asarray(ret1, dtype=float)
    owner.price_signals = replace(p, ready=ready_, ema20=ema20, momentum5=momentum5, ret1=ret1_)
    t = owner.trend
    entry = t.entry.copy(); entry[session] = np.asarray(entries, dtype=bool)
    exit_ = t.exit.copy(); exit_[session] = False if exits is None else np.asarray(exits, dtype=bool)
    market = t.market.copy(); market[session] = True
    owner.trend = replace(t, entry=entry, exit=exit_, market=market)


class UnifiedCompounderCoreTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(
            importlib.util.find_spec('research.offensive_unified_compounder_core'),
            'unified offensive compounder core implementation is absent',
        )
        return importlib.import_module('research.offensive_unified_compounder_core')

    def prepared(self):
        module = self.module()
        owner = module.Owner(sample_market(6, 145), module.Parameters(True))
        units = np.zeros(6)
        units[0] = 1000.0
        units[3] = 1000.0
        owner.observed_units = units.copy()
        owner.was_held[[0, 3]] = True
        owner.admission_alpha[0] = 2.0
        owner.admission_alpha[3] = .5
        close = owner.price_signals.price[100]
        owner.acquisition_basis[0] = close[0] * .9
        owner.acquisition_basis[3] = close[3] * .9
        owner.peak_close[0] = close[0] * 1.1
        owner.peak_close[3] = close[3] * 1.1
        owner.peak_alpha[0] = 2.1
        owner.peak_alpha[3] = .8
        owner.last_session = 99
        return owner, units

    def test_disabled_is_exact_campaign_peak_champion(self):
        module = self.module()
        market = sample_market(6, 145)
        champion = run(market, policy_factory=lambda current, cfg: ChampionOwner(current, ChampionParameters(True)))
        control = run(market, policy_factory=lambda current, cfg: module.Owner(current, module.Parameters(False)))
        pd.testing.assert_frame_equal(champion.equity, control.equity)
        pd.testing.assert_frame_equal(champion.targets, control.targets)
        self.assertEqual(champion.orders, control.orders)

    def test_actual_price_and_alpha_improvement_latches_proven_compounder(self):
        owner, units = self.prepared()
        owner.proven[0] = False
        force(owner, 100, [2.2, .1, .1, .6, .1, .1], [True, False, False, True, False, False])
        owner.decide(observe(owner, 100, units))
        self.assertTrue(owner.proven[0])
        self.assertTrue(any(row.get('action') == 'COMPOUNDER_PROVEN' and owner.market.symbols[0] in row.get('symbols', []) for row in owner.trace))

    def test_proven_compounder_ignores_generic_slow_trend_exit(self):
        owner, units = self.prepared()
        owner.proven[0] = True
        force(owner, 100, [1.0, .1, .1, .6, .1, .1], [True, False, False, True, False, False], exits=[True, False, False, False, False, False])
        decision = owner.decide(observe(owner, 100, units))
        self.assertGreater(decision.unit_targets[0], 0.0)
        self.assertTrue(any(row.get('action') == 'PROVEN_SLOW_EXIT_IGNORED' for row in owner.trace))

    def test_unproven_discovery_campaign_obeys_slow_trend_exit(self):
        owner, units = self.prepared()
        owner.proven[0] = False
        force(owner, 100, [1.0, .1, .1, .6, .1, .1], [True, False, False, True, False, False], exits=[True, False, False, False, False, False])
        decision = owner.decide(observe(owner, 100, units))
        self.assertEqual(decision.unit_targets[0], 0.0)

    def test_acute_loss_exits_even_proven_compounder(self):
        owner, units = self.prepared()
        owner.proven[0] = True
        force(owner, 100, [2.3, .1, .1, .6, .1, .1], [True, False, False, True, False, False], ret1=[-.09, 0, 0, 0, 0, 0])
        decision = owner.decide(observe(owner, 100, units))
        self.assertEqual(decision.unit_targets[0], 0.0)

    def test_fresh_stronger_challenger_can_displace_decayed_incumbent(self):
        owner, units = self.prepared()
        owner.proven[0] = True
        force(owner, 100, [1.0, 3.0, .2, .6, .1, .1], [True, True, False, True, False, False])
        decision = owner.decide(observe(owner, 100, units))
        self.assertEqual(decision.unit_targets[0], 0.0)
        self.assertTrue(owner.retired[0])
        self.assertTrue(any(row.get('action') == 'ACTIVE_DISPLACEMENT' and row.get('challenger') == owner.market.symbols[1] for row in owner.trace))

    def test_incumbent_at_funded_peak_blocks_same_close_displacement(self):
        owner, units = self.prepared()
        owner.proven[0] = True
        owner.peak_close[0] = owner.price_signals.price[100, 0]
        force(owner, 100, [1.0, 3.0, .2, .6, .1, .1], [True, True, False, True, False, False])
        decision = owner.decide(observe(owner, 100, units))
        self.assertGreater(decision.unit_targets[0], 0.0)
        self.assertFalse(owner.retired[0])
        self.assertTrue(any(row.get('action') == 'FUNDED_PEAK_DISPLACEMENT_BLOCK' for row in owner.trace))

    def test_rearmed_not_fresh_challenger_cannot_force_displacement_but_can_fill_vacancy(self):
        owner, units = self.prepared()
        owner.proven[0] = True
        owner.recovered_without_fresh_epoch[1] = True
        force(owner, 100, [1.0, 3.0, .2, .6, .1, .1], [True, True, False, True, False, False])
        held = owner.decide(observe(owner, 100, units))
        self.assertGreater(held.unit_targets[0], 0.0)
        self.assertFalse(owner.retired[0])
        units[3] = 0.0
        owner.observed_units[3] = 0.0
        owner.was_held[3] = False
        owner.last_session = 100
        force(owner, 101, [1.0, 3.0, .2, .6, .1, .1], [True, True, False, True, False, False])
        cash = float(units[0] * owner.price_signals.price[101, 0])
        filled = owner.decide(observe(owner, 101, units, cash))
        self.assertGreater(filled.unit_targets[1], 0.0)


if __name__ == '__main__':
    unittest.main()

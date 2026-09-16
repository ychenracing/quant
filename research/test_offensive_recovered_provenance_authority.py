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


def observe(owner, session, units, cash):
    units = np.asarray(units, dtype=float)
    marks = owner.parent.base.price_signals.price[session]
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


def force(owner, session, scores, entries):
    base = owner.parent.base
    features = base.features.score.copy()
    features[session] = np.asarray(scores, dtype=float)
    base.features = replace(base.features, score=features)
    signals = base.price_signals
    ready = signals.ready.copy(); ready[session] = True
    ema20 = signals.ema20.copy(); ema20[session] = np.where(np.isfinite(signals.price[session]), signals.price[session] * .9, 0)
    momentum5 = signals.momentum5.copy(); momentum5[session] = .1
    ret1 = signals.ret1.copy(); ret1[session] = 0
    base.price_signals = replace(signals, ready=ready, ema20=ema20, momentum5=momentum5, ret1=ret1)
    trend = base.trend
    entry = trend.entry.copy(); entry[session] = np.asarray(entries, dtype=bool)
    exit_ = trend.exit.copy(); exit_[session] = False
    market = trend.market.copy(); market[session] = True
    base.trend = replace(trend, entry=entry, exit=exit_, market=market)


class RecoveredProvenanceAuthorityTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(
            importlib.util.find_spec('research.offensive_recovered_provenance_authority'),
            'recovered provenance authority implementation is absent',
        )
        return importlib.import_module('research.offensive_recovered_provenance_authority')

    def prepared(self):
        module = self.module()
        owner = module.Owner(sample_market(6, 145), module.Parameters(True))
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
        module = self.module()
        market = sample_market(6, 145)
        parent = run(market, policy_factory=lambda current, cfg: TrendOwner(current, TrendParameters(2)))
        control = run(market, policy_factory=lambda current, cfg: module.Owner(current, module.Parameters(False)))
        pd.testing.assert_frame_equal(parent.equity, control.equity)
        pd.testing.assert_frame_equal(parent.targets, control.targets)
        self.assertEqual(parent.orders, control.orders)

    def test_recovered_challenger_cannot_force_ordinary_full_book_displacement(self):
        owner, units = self.prepared()
        force(owner, 100, [1.0, 3.0, 2.5, .4, .3, .2], [True, True, False, False, False, False])
        owner.parent.reference_rearmed_without_fresh_epoch[1] = True
        decision = owner.decide(observe(owner, 100, units, 0.0))
        self.assertGreater(decision.unit_targets[0], 0.0)
        self.assertFalse(owner.parent.base.retired[0])
        self.assertTrue(any(row.get('action') == 'RECOVERED_PROVENANCE_AUTHORITY_BLOCK' for row in owner.trace))

    def test_fresh_alternative_keeps_full_book_displacement_authority(self):
        owner, units = self.prepared()
        owner.parent.reference_rearmed_without_fresh_epoch[1] = True
        decision = owner.decide(observe(owner, 100, units, 0.0))
        self.assertEqual(decision.unit_targets[0], 0.0)
        self.assertTrue(owner.parent.base.retired[0])
        displacement = [row for row in owner.trace if row.get('action') == 'ALPHA_DECAY_DISPLACEMENT']
        self.assertTrue(displacement)
        self.assertEqual(displacement[-1].get('challenger'), owner.market.symbols[2])

    def test_recovered_challenger_remains_cash_admissible_when_vacancy_is_real(self):
        owner, units = self.prepared()
        units[3] = 0.0
        owner.parent.base.was_held[3] = False
        owner.parent.reference_rearmed_without_fresh_epoch[1] = True
        marks = owner.parent.base.price_signals.price[100]
        cash = float(units[0] * marks[0])
        decision = owner.decide(observe(owner, 100, units, cash))
        self.assertGreater(decision.unit_targets[1], 0.0)
        self.assertFalse(any(row.get('action') == 'RECOVERED_PROVENANCE_AUTHORITY_BLOCK' for row in owner.trace))


if __name__ == '__main__':
    unittest.main()

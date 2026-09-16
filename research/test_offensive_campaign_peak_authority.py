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
    return CloseObservation.from_inventory(session, str(owner.market.calendar[session].date()), nav, cash,
                                           units, np.divide(values, nav, out=np.zeros_like(values), where=nav > 0))


def force(owner, session, scores, entries):
    base = owner.parent.base
    features = base.features.score.copy(); features[session] = np.asarray(scores, dtype=float)
    base.features = replace(base.features, score=features)
    p = base.price_signals
    ready = p.ready.copy(); ready[session] = True
    ema = p.ema20.copy(); ema[session] = np.where(np.isfinite(p.price[session]), p.price[session] * .9, 0)
    mom = p.momentum5.copy(); mom[session] = .1
    ret = p.ret1.copy(); ret[session] = 0
    base.price_signals = replace(p, ready=ready, ema20=ema, momentum5=mom, ret1=ret)
    trend = base.trend
    entry = trend.entry.copy(); entry[session] = np.asarray(entries, dtype=bool)
    exit_ = trend.exit.copy(); exit_[session] = False
    market = trend.market.copy(); market[session] = True
    base.trend = replace(trend, entry=entry, exit=exit_, market=market)


class CampaignPeakAuthorityTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research.offensive_campaign_peak_authority'),
                             'campaign peak authority implementation is absent')
        return importlib.import_module('research.offensive_campaign_peak_authority')

    def prepared(self):
        module = self.module()
        owner = module.Owner(sample_market(6, 145), module.Parameters(True))
        units = np.zeros(6)
        units[0] = 1000.0; units[3] = 1000.0
        owner.parent.base.was_held[[0, 3]] = True
        owner.parent.base.owned_alpha_reference[0] = 2.0
        owner.parent.base.owned_alpha_reference[3] = .5
        owner.parent.base.last_session = 99
        force(owner, 99, [1.0, 2.8, 2.0, .4, .3, .2], [True, False, False, False, False, False])
        force(owner, 100, [1.0, 3.0, 2.0, .4, .3, .2], [True, True, False, False, False, False])
        return owner, units

    def test_control_is_exact_trend_book(self):
        module = self.module(); market = sample_market(6, 145)
        parent = run(market, policy_factory=lambda current, cfg: TrendOwner(current, TrendParameters(2)))
        control = run(market, policy_factory=lambda current, cfg: module.Owner(current, module.Parameters(False)))
        pd.testing.assert_frame_equal(parent.equity, control.equity)
        pd.testing.assert_frame_equal(parent.targets, control.targets)
        self.assertEqual(parent.orders, control.orders)

    def test_fresh_edge_cannot_liquidate_incumbent_at_campaign_closing_high(self):
        owner, units = self.prepared()
        price = float(owner.parent.base.price_signals.price[100, 0])
        owner.actual_was_held[[0, 3]] = True
        owner.campaign_peak_close[0] = price
        owner.campaign_peak_close[3] = float(owner.parent.base.price_signals.price[100, 3])
        decision = owner.decide(observe(owner, 100, units, 0.0))
        self.assertGreater(decision.unit_targets[0], 0.0)
        self.assertFalse(owner.parent.base.retired[0])
        self.assertTrue(any(row.get('action') == 'CAMPAIGN_PEAK_AUTHORITY_BLOCK' for row in owner.trace))

    def test_incumbent_below_campaign_peak_keeps_champion_displacement_authority(self):
        owner, units = self.prepared()
        price = float(owner.parent.base.price_signals.price[100, 0])
        owner.actual_was_held[[0, 3]] = True
        owner.campaign_peak_close[0] = price * 1.05
        owner.campaign_peak_close[3] = float(owner.parent.base.price_signals.price[100, 3])
        decision = owner.decide(observe(owner, 100, units, 0.0))
        self.assertEqual(decision.unit_targets[0], 0.0)
        self.assertTrue(owner.parent.base.retired[0])
        self.assertFalse(any(row.get('action') == 'CAMPAIGN_PEAK_AUTHORITY_BLOCK' for row in owner.trace))

    def test_fresh_edge_remains_cash_admissible_with_real_vacancy(self):
        owner, units = self.prepared(); units[3] = 0.0
        owner.parent.base.was_held[3] = False
        owner.actual_was_held[0] = True
        owner.campaign_peak_close[0] = float(owner.parent.base.price_signals.price[100, 0])
        cash = float(units[0] * owner.parent.base.price_signals.price[100, 0])
        decision = owner.decide(observe(owner, 100, units, cash))
        self.assertGreater(decision.unit_targets[1], 0.0)
        self.assertFalse(any(row.get('action') == 'CAMPAIGN_PEAK_AUTHORITY_BLOCK' for row in owner.trace))

    def test_actual_flat_resets_campaign_peak_before_new_inventory(self):
        owner, units = self.prepared()
        owner.actual_was_held[0] = True; owner.campaign_peak_close[0] = 999.0
        flat = units.copy(); flat[0] = 0.0; flat[3] = 0.0
        force(owner, 100, [.1, .1, .1, .1, .1, .1], [False, False, False, False, False, False])
        owner.decide(observe(owner, 100, flat, 1_000_000.0))
        self.assertFalse(owner.actual_was_held[0])
        self.assertTrue(np.isnan(owner.campaign_peak_close[0]))


if __name__ == '__main__':
    unittest.main()

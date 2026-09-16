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


def force(owner, session, *, exits=None, ret1=None, ready=None):
    b = base(owner)
    p = b.price_signals
    ready_values = p.ready.copy()
    ready_values[session] = True if ready is None else np.asarray(ready, dtype=bool)
    one_day = p.ret1.copy(); one_day[session] = 0 if ret1 is None else np.asarray(ret1, dtype=float)
    b.price_signals = replace(p, ready=ready_values, ret1=one_day)
    trend = b.trend
    exit_ = trend.exit.copy(); exit_[session] = False if exits is None else np.asarray(exits, dtype=bool)
    market = trend.market.copy(); market[session] = True
    b.trend = replace(trend, exit=exit_, market=market)


class ProfitableCampaignCompounderTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(
            importlib.util.find_spec('research.offensive_profitable_campaign_compounder'),
            'profitable campaign compounder implementation is absent',
        )
        return importlib.import_module('research.offensive_profitable_campaign_compounder')

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
        owner.observed_units = units.copy()
        return owner, units

    def test_disabled_is_exact_campaign_peak_champion(self):
        module = self.module()
        market = sample_market(6, 145)
        champion = run(market, policy_factory=lambda current, cfg: ChampionOwner(current, ChampionParameters(True)))
        control = run(market, policy_factory=lambda current, cfg: module.Owner(current, module.Parameters(False)))
        pd.testing.assert_frame_equal(champion.equity, control.equity)
        pd.testing.assert_frame_equal(champion.targets, control.targets)
        self.assertEqual(champion.orders, control.orders)

    def test_profitable_funded_campaign_forgives_slow_trend_exit(self):
        owner, units = self.prepared()
        close = base(owner).price_signals.price[100, 0]
        owner.acquisition_basis[0] = close * .8
        owner.acquisition_basis[3] = base(owner).price_signals.price[100, 3] * .8
        force(owner, 100, exits=[True, False, False, False, False, False])
        decision = owner.decide(observe(owner, 100, units))
        self.assertGreater(decision.unit_targets[0], 0.0)
        self.assertTrue(any(row.get('action') == 'PROFITABLE_CAMPAIGN_TREND_EXIT_FORGIVEN' for row in owner.trace))

    def test_unprofitable_campaign_keeps_parent_slow_trend_exit(self):
        owner, units = self.prepared()
        close = base(owner).price_signals.price[100, 0]
        owner.acquisition_basis[0] = close * 1.01
        owner.acquisition_basis[3] = base(owner).price_signals.price[100, 3] * .8
        force(owner, 100, exits=[True, False, False, False, False, False])
        decision = owner.decide(observe(owner, 100, units))
        self.assertEqual(decision.unit_targets[0], 0.0)
        self.assertFalse(any(row.get('action') == 'PROFITABLE_CAMPAIGN_TREND_EXIT_FORGIVEN' and owner.market.symbols[0] in row.get('symbols', []) for row in owner.trace))

    def test_acute_loss_remains_authoritative_for_profitable_campaign(self):
        owner, units = self.prepared()
        close = base(owner).price_signals.price[100, 0]
        owner.acquisition_basis[0] = close * .8
        owner.acquisition_basis[3] = base(owner).price_signals.price[100, 3] * .8
        force(owner, 100, exits=[False, False, False, False, False, False], ret1=[-.09, 0, 0, 0, 0, 0])
        decision = owner.decide(observe(owner, 100, units))
        self.assertEqual(decision.unit_targets[0], 0.0)

    def test_actual_inventory_first_fill_and_addition_update_weighted_basis(self):
        owner, _ = self.prepared()
        owner.observed_units[:] = 0.0
        owner.acquisition_basis[:] = np.nan
        units = np.zeros(6); units[0] = 1000.0
        force(owner, 100)
        owner.decide(observe(owner, 100, units))
        first_open = owner.execution_open[100, 0]
        self.assertAlmostEqual(owner.acquisition_basis[0], first_open)
        prior_basis = owner.acquisition_basis[0]
        units[0] = 1500.0
        base(owner).last_session = 100
        force(owner, 101)
        owner.decide(observe(owner, 101, units))
        expected = (1000.0 * prior_basis + 500.0 * owner.execution_open[101, 0]) / 1500.0
        self.assertAlmostEqual(owner.acquisition_basis[0], expected)

    def test_reduction_does_not_rewrite_basis_and_flat_clears_it(self):
        owner, units = self.prepared()
        owner.acquisition_basis[0] = 123.45
        units[0] = 500.0
        force(owner, 100)
        owner.decide(observe(owner, 100, units))
        self.assertAlmostEqual(owner.acquisition_basis[0], 123.45)
        units[0] = 0.0
        base(owner).last_session = 100
        force(owner, 101)
        owner.decide(observe(owner, 101, units))
        self.assertTrue(np.isnan(owner.acquisition_basis[0]))


if __name__ == '__main__':
    unittest.main()

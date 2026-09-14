"""Causal observed-price authority on the existing actual-inventory owner."""
from dataclasses import asdict
import importlib
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from test_core import sample_market
from techquant.data import Market
from techquant.engine import run
from techquant.policy import CloseObservation
from research.ledger_attribution import attribute


class TrendBookTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research.trend_book'),
                             'the preregistered price authority is missing')
        return importlib.import_module('research.trend_book')

    def execute(self, market, positions=2):
        module = self.module()
        with patch('research.coherent.ForecastValidator',
                   side_effect=AssertionError('price book must not depend on forecasts')):
            return run(market, policy_factory=lambda current, cfg:
                       module.Owner(current, module.Parameters(positions)))

    def observation(self, owner, i, units, cash):
        marks = owner.features.close[i]
        units = np.asarray(units, dtype=float)
        value = units * marks
        nav = float(cash + value.sum())
        return CloseObservation.from_inventory(i, str(owner.market.calendar[i].date()),
                                               nav, cash, units, value / nav)

    def test_grid_is_exact_and_capacity_follows_admission(self):
        module = self.module()
        self.assertEqual([asdict(p) for p in module.grid()], [{'positions': 2}, {'positions': 4}])
        market = sample_market(6, 80)
        for p in module.grid():
            owner = module.Owner(market, p)
            owner.risk.cap = 1.
            owner.risk.update = lambda *args: (1., 'CONTROLLED_RISK_OBSERVATION')
            d = owner.decide(self.observation(owner, 40, np.zeros(6), 2_000_000.))
            self.assertEqual(np.count_nonzero(d.unit_targets), p.positions)
            expected = set(np.argsort(owner.features.score[40])[-p.positions:])
            self.assertEqual(set(np.flatnonzero(d.unit_targets)), expected)
            self.assertLessEqual(d.weights.sum(), 1.)
            self.assertTrue((d.weights <= .6 + 1e-12).all())
        for invalid in (True, 0, 3, 2.0):
            with self.assertRaises(ValueError):
                module.Parameters(invalid)

    def test_actual_execution_is_prefix_causal_and_uses_no_forecasts(self):
        market = sample_market(6, 145)
        full = self.execute(market)
        cutoff = market.calendar[109]
        short = self.execute(market.prefix(cutoff))
        pd.testing.assert_frame_equal(full.equity.loc[:cutoff], short.equity)
        pd.testing.assert_frame_equal(full.targets.loc[:cutoff], short.targets)
        self.assertTrue((full.equity.cash >= 0).all())
        fills = [o for o in full.orders if o['status'] == 'FILLED']
        self.assertTrue(fills)
        self.assertTrue(all(o['signal_date'] < o['date'] for o in fills))
        _, _, checks = attribute(market, full)
        self.assertLess(checks['max_reconciliation_error'], 1e-6)

    def test_removed_symbol_and_sector_do_not_affect_remaining_decisions(self):
        self.module()
        market = sample_market(5, 115)
        names = market.symbols[:3]
        frames = {s: f.copy() for s, f in market.frames.items()}
        for symbol in market.symbols[3:]:
            frames[symbol].loc[:, ['open', 'high', 'low', 'close', 'raw_open', 'raw_close']] *= 97.
        changed = Market.from_frames(frames, market.calendar, quality=market.quality,
                                     sectors={s: 'excluded_sector' for s in market.symbols[3:]})
        first = self.execute(market.subset(names))
        second = self.execute(changed.subset(names))
        pd.testing.assert_frame_equal(first.equity, second.equity)
        pd.testing.assert_frame_equal(first.targets, second.targets)
        self.assertEqual(first.orders, second.orders)

    def test_readiness_counts_observations_not_forward_filled_days(self):
        module = self.module()
        market = sample_market(1, 90)
        symbol = market.symbols[0]
        frames = {symbol: market.frames[symbol].iloc[-25:].copy()}
        sparse = Market.from_frames(frames, market.calendar, quality=market.quality)
        owner = module.Owner(sparse, module.Parameters())
        ready = owner.price_signals.ready[:, 0]
        self.assertFalse(ready[:84].any())
        self.assertTrue(ready[84:].all())
        stale_frames = {symbol: frames[symbol].copy()}
        stale_frames[symbol].loc[market.calendar[-1], 'volume'] = 0.
        stale = module.Owner(Market.from_frames(stale_frames, market.calendar,
                             quality=market.quality), module.Parameters())
        self.assertFalse(stale.price_signals.ready[-1, 0])

    def test_protective_sale_stays_latched_and_cannot_spend_intended_proceeds(self):
        module = self.module()
        market = sample_market(2, 80)
        frames = {s: f.copy() for s, f in market.frames.items()}
        first = market.symbols[0]
        frames[first].iloc[40:, frames[first].columns.get_indexer(
            ['open', 'high', 'low', 'close', 'raw_open', 'raw_close'])] *= .85
        changed = Market.from_frames(frames, market.calendar, quality=market.quality)
        owner = module.Owner(changed, module.Parameters())
        owner.risk.cap = 1.
        owner.risk.update = lambda *args: (1., 'CONTROLLED_RISK_OBSERVATION')
        units = [50000., 0.]
        owner.decide(self.observation(owner, 39, units, 500000.))
        first_request = owner.decide(self.observation(owner, 40, units, 500000.))
        blocked_request = owner.decide(self.observation(owner, 41, units, 500000.))
        np.testing.assert_array_equal(first_request.unit_targets, [0., 0.])
        np.testing.assert_array_equal(blocked_request.unit_targets, [0., 0.])
        self.assertIn('PROTECTIVE_INVENTORY_RETRY', blocked_request.reason)
        self.assertFalse(owner.readmit[0], 'an unfilled request is not an actual exit')


if __name__ == '__main__':
    unittest.main()

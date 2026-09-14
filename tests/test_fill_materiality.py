"""The ordinary 1% NAV floor applies to executable size, not just intent."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from techquant.config import Config
from techquant.data import Market
from techquant.engine import run


def constant_market(*, price: float = 20., volume: float = 20_000_000.,
                    days: int = 30, names: int = 1) -> Market:
    dates = pd.bdate_range('2023-01-03', periods=days)
    frame = pd.DataFrame({
        'open': price, 'high': price, 'low': price, 'close': price,
        'raw_open': price, 'raw_close': price, 'volume': volume,
    }, index=dates)
    return Market.from_frames({f'sz300{100+j:03}': frame.copy() for j in range(names)},
                              dates, quality='synthetic')


class FillMaterialityTests(unittest.TestCase):
    def test_capacity_and_lot_sizing_cannot_create_submaterial_buy(self):
        cases = ((constant_market(volume=100_000.), .5),
                 (constant_market(price=30.), .01001))
        for market, weight in cases:
            with self.subTest(price=market.frames[market.symbols[0]].close.iloc[0],
                              weight=weight):
                targets = pd.DataFrame(weight, index=market.calendar, columns=market.symbols)
                result = run(market, targets=targets, cost_multiplier=0.)
                self.assertFalse(any(o['status'] == 'FILLED' for o in result.orders))
                self.assertTrue(result.orders)
                self.assertEqual({o['reason'] for o in result.orders},
                                 {'BELOW_MINIMUM_NOTIONAL'})
                np.testing.assert_array_equal(result.equity.cash, Config().initial_cash)

    def test_cash_clipping_cannot_create_submaterial_buy(self):
        market = constant_market(names=2)
        # The earlier symbol consumes all but 0.75% of NAV at the opening gap.
        frames = {s: f.copy() for s, f in market.frames.items()}
        first = market.symbols[0]
        # Use inventory intents so the signal-close quantity stays fixed at the gap.
        from techquant.policy import CloseDecision

        class FixedInventory:
            def decide(self, observation):
                units = np.array([90_000., 2_000.]) if observation.session == 0 else observation.units
                return CloseDecision(np.array([.9, .02]) if observation.session == 0
                                     else observation.weights, reason='FIXED', unit_targets=units)

            def identity(self):
                return {'fixture': 'cash-clipped inventory intent'}

        frames[first].iloc[1:, frames[first].columns.get_indexer(
            ['open', 'high', 'low', 'close', 'raw_open', 'raw_close'])] = 22.055
        market = Market.from_frames(frames, market.calendar, quality='synthetic')
        result = run(market, policy_factory=lambda _m, _c: FixedInventory(), cost_multiplier=0.)
        second = [o for o in result.orders if o['symbol'] == market.symbols[1]]
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0]['status'], 'BLOCKED')
        self.assertEqual(second[0]['reason'], 'BELOW_MINIMUM_NOTIONAL')

    def test_exact_floor_fills_and_zero_quantity_keeps_existing_reason(self):
        for volume, expected in ((200_000., 'NEXT_OPEN'), (1., 'MINIMUM_LOT_OR_CASH')):
            market = constant_market(volume=volume)
            targets = pd.DataFrame(.5, index=market.calendar, columns=market.symbols)
            result = run(market, targets=targets, cost_multiplier=0.)
            self.assertEqual(result.orders[0]['reason'], expected)
            if expected == 'NEXT_OPEN':
                self.assertEqual(result.orders[0]['notional'], Config().initial_cash * .01)

    def test_capacity_block_retries_without_using_today_volume(self):
        market = constant_market(volume=100_000.)
        symbol = market.symbols[0]
        frames = {symbol: market.frames[symbol].copy()}
        frames[symbol].iloc[4:, frames[symbol].columns.get_loc('volume')] = 1_000_000.
        market = Market.from_frames(frames, market.calendar, quality='synthetic')
        targets = pd.DataFrame(.5, index=market.calendar, columns=market.symbols)
        result = run(market, targets=targets, cost_multiplier=0.)
        first_fill = next(o for o in result.orders if o['status'] == 'FILLED')
        self.assertEqual(first_fill['date'], str(market.calendar[5].date()))
        self.assertTrue(all(o['reason'] == 'BELOW_MINIMUM_NOTIONAL'
                            for o in result.orders if o['date'] < first_fill['date']))
        cut = market.calendar[12]
        short = run(market.prefix(cut), targets=targets.loc[:cut], cost_multiplier=0.)
        pd.testing.assert_frame_equal(result.equity.loc[:cut], short.equity)
        self.assertEqual([o for o in result.orders if o['date'] <= str(cut.date())], short.orders)

    def test_protective_partial_exit_remains_exempt(self):
        market = constant_market(volume=100_000., days=28)
        symbol = market.symbols[0]
        frames = {symbol: market.frames[symbol].copy()}
        frames[symbol].iloc[0, frames[symbol].columns.get_loc('volume')] = 1_000_000.
        market = Market.from_frames(frames, market.calendar, quality='synthetic')
        targets = pd.DataFrame(.01, index=market.calendar, columns=market.symbols)
        targets.iloc[20:] = 0.
        result = run(market, targets=targets, cost_multiplier=0.)
        sells = [o for o in result.orders if o['side'] == 'SELL' and o['status'] == 'FILLED']
        self.assertTrue(sells)
        self.assertTrue(all(o['notional'] < Config().initial_cash * .01 for o in sells))
        self.assertEqual(result.equity.holdings.iloc[-1], 0.)


if __name__ == '__main__':
    unittest.main()

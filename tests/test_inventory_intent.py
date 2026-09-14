"""A close-time inventory intention must not become an opposite opening order."""
import unittest
import numpy as np
from test_core import sample_market
from techquant.data import Market
from techquant.engine import run
from techquant.policy import CloseDecision


class InventoryIntentTests(unittest.TestCase):
    def market(self, gap):
        market = sample_market(1, 35)
        symbol = market.symbols[0]
        frame = market.frames[symbol].copy()
        frame.loc[:, ['open', 'high', 'low', 'close', 'raw_open', 'raw_close']] = 20.
        frame.iloc[11, frame.columns.get_loc('open')] = 20. * (1. + gap)
        frame.iloc[11, frame.columns.get_loc('raw_open')] = 20. * (1. + gap)
        frame.iloc[11, frame.columns.get_loc('high')] = max(20., 20. * (1. + gap))
        frame.iloc[11, frame.columns.get_loc('low')] = min(20., 20. * (1. + gap))
        return Market.from_frames({symbol: frame}, market.calendar, quality='synthetic')

    def simulate(self, gap, change, explicit=True, delay=1):
        market = self.market(gap)
        seen = []
        class Intent:
            def __init__(self, m, cfg): self.prices = m.panel('close').to_numpy()
            def decide(self, o):
                weights = np.array([.5]) if o.session == 0 else o.weights.copy()
                if o.session == 11-delay:
                    weights[0] += change
                seen.append(o)
                kwargs = {'unit_targets': weights * o.nav / self.prices[o.session]} if explicit else {}
                return CloseDecision(weights, 'EXPLICIT_INVENTORY_INTENT', **kwargs)
            def identity(self): return {'name': 'synthetic_inventory_intent', 'explicit': explicit}
        return run(market, policy_factory=Intent, delay=delay), seen

    def test_legacy_weight_target_behavior_remains_explicitly_distinct(self):
        result, _ = self.simulate(-.15, -.01, explicit=False)
        on_gap = [o for o in result.orders if o['date'] == '2023-01-18' and o['status'] == 'FILLED']
        self.assertEqual([o['side'] for o in on_gap], ['BUY'])

    def test_unit_reduction_cannot_turn_into_buy_after_gap_down(self):
        result, seen = self.simulate(-.15, -.01)
        on_gap = [o for o in result.orders if o['date'] == '2023-01-18' and o['status'] == 'FILLED']
        self.assertEqual([o['side'] for o in on_gap], ['SELL'])
        self.assertLess(seen[11].units[0], seen[10].units[0])
        np.testing.assert_allclose(result.equity.nav, result.equity.cash + result.equity.holdings)

    def test_unit_increase_cannot_turn_into_sell_after_gap_up(self):
        result, seen = self.simulate(.15, .03)
        on_gap = [o for o in result.orders if o['date'] == '2023-01-18' and o['status'] == 'FILLED']
        self.assertEqual([o['side'] for o in on_gap], ['BUY'])
        self.assertGreater(seen[11].units[0], seen[10].units[0])
        self.assertTrue((result.equity.cash >= -1e-6).all())

    def test_delayed_unit_intent_uses_original_signal_not_future_price(self):
        result, seen = self.simulate(-.15, -.01, delay=2)
        on_gap = [o for o in result.orders if o['date'] == '2023-01-18' and o['status'] == 'FILLED']
        self.assertEqual([o['side'] for o in on_gap], ['SELL'])
        self.assertEqual(on_gap[0]['signal_date'], '2023-01-16')

    def test_unit_targets_must_match_validated_close_allocation(self):
        market = self.market(0.)
        for units in (np.array([-1.]), np.array([np.nan]), np.array([1., 2.]), np.array([1e10])):
            class Bad:
                def __init__(self, m, cfg): pass
                def decide(self, o): return CloseDecision(np.array([.5]), 'BAD', unit_targets=units)
                def identity(self): return {'name': 'bad'}
            with self.subTest(units=units), self.assertRaises(ValueError):
                run(market, policy_factory=Bad)

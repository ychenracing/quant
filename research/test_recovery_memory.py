"""Risk-event memory must follow causal signals and actual fills, not requests."""
from dataclasses import asdict
import importlib
import importlib.util
import unittest
import numpy as np
import pandas as pd
from test_core import sample_market
from techquant.data import Market
from techquant.engine import run
from techquant.policy import CloseObservation
from research.trend_book import Owner as PriceOwner, Parameters as PriceParameters


class RecoveryMemoryTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research.recovery_memory'),
                             'registered risk-event memory has not been implemented')
        return importlib.import_module('research.recovery_memory')

    def owner(self, market=None, remember=True):
        module = self.module()
        return module.Owner(market or sample_market(2, 100), module.Parameters(remember))

    def observation(self, owner, i, units=None, cash=2_000_000.):
        if units is None:
            units = np.zeros(len(owner.market.symbols))
        units = np.asarray(units, dtype=float)
        values = units * owner.features.close[i]
        nav = float(cash + values.sum())
        return CloseObservation.from_inventory(i, str(owner.market.calendar[i].date()),
                                               nav, cash, units, values / nav)

    def cap(self, owner, value):
        def update(*args):
            owner.risk.cap = value
            return value, 'CONTROLLED_CAP_EVENT'
        owner.risk.update = update

    def test_declared_control_rejects_an_entry_with_simultaneous_exit_warning(self):
        module = self.module()
        self.assertEqual([asdict(p) for p in module.grid()],
                         [{'remember_risk': False}, {'remember_risk': True}])
        for bad in (0, 1, None, 'yes'):
            with self.assertRaises(ValueError):
                module.Parameters(bad)
        market = sample_market(2, 100)
        frames = {s: f.copy() for s, f in market.frames.items()}
        symbol = market.symbols[-1]
        cols = ['open','high','low','close','raw_open','raw_close']
        frames[symbol].loc[market.calendar[45], cols] *= 1.12
        market = Market.from_frames(frames, market.calendar, quality='synthetic')
        old = PriceOwner(market, PriceParameters(2))._signal_inputs(46)
        self.assertTrue(old.allowed[-1] and old.broken[-1])
        owner = self.owner(market, False)
        owner.risk.cap = 1.; self.cap(owner, 1.)
        d = owner.decide(self.observation(owner, 46))
        self.assertEqual(d.unit_targets[-1], 0.)
        self.assertGreater(d.unit_targets[0], 0.)

    def test_cap_event_blocks_new_units_and_records_unheld_fresh_names(self):
        owner = self.owner()
        owner.risk.cap = 1.; self.cap(owner, .5)
        d = owner.decide(self.observation(owner, 40))
        np.testing.assert_array_equal(d.unit_targets, [0., 0.])
        np.testing.assert_array_equal(owner.event_price, owner.features.close[40])
        self.assertIn('RISK_EVENT_PRICE_RECORDED', d.reason)
        np.testing.assert_array_equal(owner.recovery_closes, [0, 0])

    def test_only_actual_funded_recovery_clears_the_price_memory(self):
        owner = self.owner()
        owner.risk.cap = 1.; self.cap(owner, .5)
        owner.decide(self.observation(owner, 40))
        anchors = owner.event_price.copy()
        for i in (41, 42):
            d = owner.decide(self.observation(owner, i))
            self.assertEqual(float(d.unit_targets.sum()), 0.)
        d = owner.decide(self.observation(owner, 43))
        self.assertGreater(d.unit_targets.sum(), 0.)
        # The opening order did not fill. Quote recovery alone cannot erase risk.
        blocked = owner.decide(self.observation(owner, 44))
        np.testing.assert_array_equal(owner.event_price, anchors)
        self.assertGreater(blocked.unit_targets.sum(), 0.)
        after_fill = owner.decide(self.observation(owner, 45, [1000., 0.], 1_950_000.))
        self.assertEqual(owner.event_price[0], 0.)
        self.assertEqual(owner.event_price[1], anchors[1])
        self.assertIn('FUNDED_PRICE_RECOVERY', after_fill.reason)

    def test_warning_retry_keeps_original_anchor_and_new_warning_cannot_lower_it(self):
        market = sample_market(1, 100)
        symbol = market.symbols[0]
        frame = market.frames[symbol].copy()
        cols = ['open','high','low','close','raw_open','raw_close']
        frame.loc[market.calendar[40]:, cols] *= .85
        market = Market.from_frames({symbol: frame}, market.calendar, quality='synthetic')
        owner = self.owner(market)
        owner.risk.cap = 1.; self.cap(owner, 1.)
        owner.decide(self.observation(owner, 39, [50000.], 100000.))
        first = owner.decide(self.observation(owner, 40, [50000.], 100000.))
        anchor = owner.event_price[0]
        retry = owner.decide(self.observation(owner, 41, [50000.], 100000.))
        self.assertEqual(first.unit_targets[0], 0.)
        self.assertEqual(retry.unit_targets[0], 0.)
        self.assertEqual(owner.event_price[0], anchor)
        self.assertNotIn('RISK_EVENT_PRICE_RECORDED', retry.reason)
        # An unresolved earlier high warning is never replaced by a lower one.
        owner.event_price[0] = owner.features.close[39, 0]
        earlier = owner.event_price[0]
        self.cap(owner, .5)
        owner.decide(self.observation(owner, 42, [50000.], 100000.))
        self.assertEqual(owner.event_price[0], earlier)

    def test_stale_close_resets_confirmation_and_does_not_clear_on_inventory_change(self):
        market = sample_market(2, 100)
        frames = {s: f.copy() for s, f in market.frames.items()}
        frames[market.symbols[0]].loc[market.calendar[43], 'volume'] = 0.
        market = Market.from_frames(frames, market.calendar, quality='synthetic')
        owner = self.owner(market)
        owner.risk.cap = 1.; self.cap(owner, .5)
        owner.decide(self.observation(owner, 40))
        owner.decide(self.observation(owner, 41))
        owner.decide(self.observation(owner, 42))
        anchor = owner.event_price[0]
        owner.decide(self.observation(owner, 43, [1000., 0.]))
        self.assertEqual(owner.event_price[0], anchor)
        self.assertEqual(owner.recovery_closes[0], 0)

    def test_prefix_and_exclusion_leave_all_actual_decisions_identical(self):
        module = self.module()
        market = sample_market(4, 135)
        shocked = {s: f.copy() for s, f in market.frames.items()}
        for frame in shocked.values():
            frame.loc[market.calendar[55]:market.calendar[69],
                      ['open','high','low','close','raw_open','raw_close']] *= .82
        market = Market.from_frames(shocked, market.calendar, quality='synthetic')
        def execute(m):
            return run(m, policy_factory=lambda current,cfg: module.Owner(current, module.Parameters()))
        full = execute(market)
        self.assertTrue(full.equity.reason.str.contains('RISK_EVENT_PRICE_RECORDED').any())
        cut = market.calendar[99]
        short = execute(market.prefix(cut))
        pd.testing.assert_frame_equal(full.equity.loc[:cut], short.equity)
        pd.testing.assert_frame_equal(full.targets.loc[:cut], short.targets)
        removed = market.symbols[-1]
        frames = {s: f.copy() for s, f in market.frames.items()}
        frames[removed].loc[:, ['open','high','low','close','raw_open','raw_close']] *= 100.
        changed = Market.from_frames(frames, market.calendar, quality='synthetic',
                                     sectors={removed: 'excluded'})
        names = market.symbols[:-1]
        a,b = execute(market.subset(names)),execute(changed.subset(names))
        pd.testing.assert_frame_equal(a.equity, b.equity)
        self.assertEqual(a.orders, b.orders)
        self.assertTrue((full.equity.cash >= 0).all())
        self.assertTrue(all(o['signal_date'] < o['date'] for o in full.orders if o['status']=='FILLED'))


if __name__ == '__main__':
    unittest.main()

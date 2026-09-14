"""Real-engine causality, cash and T+1 checks for the integrated coherent owner."""
import unittest
import pandas as pd
from test_core import sample_market
from techquant.engine import run
from research.coherent import Owner, Parameters
from research.pathwise import forecast, Parameters as ForecastParameters
from research.ledger_attribution import attribute


class CoherentExecutionTests(unittest.TestCase):
    def test_real_execution_is_prefix_causal_and_reconciles(self):
        market = sample_market(3, 115)
        cutoff = market.calendar[89]
        def execute(m):
            issued = forecast(m, ForecastParameters(20, .12))
            return run(m, policy_factory=lambda current, cfg: Owner(
                current, Parameters('price'), prediction=issued))
        full = execute(market)
        short = execute(market.prefix(cutoff))
        pd.testing.assert_frame_equal(full.equity.loc[:cutoff], short.equity)
        pd.testing.assert_frame_equal(full.targets.loc[:cutoff], short.targets)
        fills = [o for o in full.orders if o['status'] == 'FILLED']
        self.assertTrue(fills)
        self.assertTrue(all(o['signal_date'] < o['date'] for o in fills))
        self.assertTrue((full.equity.cash >= 0).all())
        _, _, checks = attribute(market, full)
        self.assertLess(checks['max_reconciliation_error'], 1e-6)

    def test_removed_quotes_do_not_enter_owner_or_features(self):
        from techquant.data import Market
        market = sample_market(3, 105)
        names = market.symbols[:2]
        frames = {symbol: frame.copy() for symbol, frame in market.frames.items()}
        frames[market.symbols[2]].loc[:, ['open','high','low','close','raw_open','raw_close']] *= 97.
        changed = Market.from_frames(frames, market.calendar, quality=market.quality)
        def execute(m):
            supplied = forecast(m, ForecastParameters(20, .12))
            return run(m, policy_factory=lambda current, cfg: Owner(
                current, Parameters('price'), prediction=supplied))
        first, second = execute(market.subset(names)), execute(changed.subset(names))
        pd.testing.assert_frame_equal(first.equity, second.equity)
        pd.testing.assert_frame_equal(first.targets, second.targets)
        self.assertEqual(first.orders, second.orders)

"""Production passive-ownership contracts; economics are measured separately."""
from __future__ import annotations

import unittest

import pandas as pd

from test_core import sample_market
from techquant.config import Config
from techquant.data import Market
from techquant.engine import run


class PassiveOwnershipTests(unittest.TestCase):
    def passive(self):
        try:
            from techquant.passive import run_passive_ownership
        except ImportError as exc:
            self.fail(f"passive ownership production adapter is absent: {exc}")
        return run_passive_ownership

    def test_production_adapter_exactly_matches_buy_hold_research_account(self):
        market = sample_market(4, 140)
        passive = self.passive()
        for delay, cost in ((1, 1.0), (2, 3.0)):
            benchmark = run(market, Config(), benchmark="buy_hold", delay=delay, cost_multiplier=cost)
            production = passive(market, Config(), delay=delay, cost_multiplier=cost)
            pd.testing.assert_frame_equal(benchmark.equity, production.equity)
            pd.testing.assert_frame_equal(benchmark.targets, production.targets)
            self.assertEqual(benchmark.orders, production.orders)
            self.assertEqual(production.metadata["strategy"], "passive_ownership")
            self.assertEqual(production.metadata["economic_semantics"], "same_engine_buy_hold")
            self.assertIsNone(production.metadata["benchmark"])

    def test_prefix_replay_is_exactly_causal(self):
        market = sample_market(3, 150)
        passive = self.passive()
        full = passive(market)
        cut = market.calendar[95]
        prefix = passive(market.prefix(cut))
        pd.testing.assert_frame_equal(full.equity.loc[:cut], prefix.equity)
        pd.testing.assert_frame_equal(full.targets.loc[:cut], prefix.targets)
        self.assertEqual(
            [order for order in full.orders if order["date"] <= str(cut.date())],
            prefix.orders,
        )

    def test_late_listing_keeps_its_original_sleeve_until_executable(self):
        base = sample_market(2, 120)
        frames = dict(base.frames)
        late_symbol = "sh688999"
        frames[late_symbol] = next(iter(base.frames.values())).iloc[70:].copy()
        market = Market.from_frames(frames, base.calendar, quality="synthetic")
        passive = self.passive()

        benchmark = run(market, benchmark="buy_hold")
        production = passive(market)
        pd.testing.assert_frame_equal(benchmark.equity, production.equity)
        pd.testing.assert_frame_equal(benchmark.targets, production.targets)
        self.assertEqual(benchmark.orders, production.orders)

        late_fills = [
            order for order in production.orders
            if order["symbol"] == late_symbol and order["side"] == "BUY" and order["status"] == "FILLED"
        ]
        self.assertTrue(late_fills)
        self.assertGreaterEqual(late_fills[0]["date"], str(base.calendar[71].date()))
        earlier_late_orders = [
            order for order in production.orders
            if order["symbol"] == late_symbol and order["date"] < str(base.calendar[70].date())
        ]
        self.assertFalse(earlier_late_orders)

    def test_adapter_signature_has_no_conflicting_decision_source_inputs(self):
        import inspect

        parameters = inspect.signature(self.passive()).parameters
        self.assertNotIn("benchmark", parameters)
        self.assertNotIn("targets", parameters)
        self.assertNotIn("policy_factory", parameters)


if __name__ == "__main__":
    unittest.main()

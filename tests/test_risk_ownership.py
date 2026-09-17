"""Focused contracts for causal risk-aware ownership."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from test_core import sample_market
from techquant.config import Config
from techquant.data import Market
from techquant.passive import run_passive_ownership
from techquant.policy import CloseObservation, OwnershipIntent
from techquant.risk_ownership import (
    RiskAwareOwnershipPolicy,
    RiskOwnershipParameters,
    run_risk_aware_ownership,
)


def shock_market(days: int = 190) -> Market:
    dates = pd.bdate_range("2023-01-03", periods=days)
    frames = {}
    for k in range(3):
        close = 20.0 * np.exp(np.arange(days) * (0.0015 + k * 0.0001))
        pre = close[79]
        close[80:94] = pre * 0.78 * np.exp(np.arange(14) * 0.001)
        close[94:] = close[93] * np.exp(np.arange(days - 94) * 0.018)
        opening = close * 0.999
        frames[f"sz300{300 + k:03}"] = pd.DataFrame(
            {
                "open": opening,
                "high": np.maximum(opening, close) * 1.01,
                "low": np.minimum(opening, close) * 0.99,
                "close": close,
                "volume": 20_000_000.0,
                "raw_open": opening,
                "raw_close": close,
            },
            index=dates,
        )
    return Market.from_frames(frames, dates, quality="synthetic")


class RiskAwareOwnershipTests(unittest.TestCase):
    def test_open_state_exactly_matches_passive_ownership(self):
        market = sample_market(4, 140)
        passive = run_passive_ownership(market)
        candidate = run_risk_aware_ownership(market)
        pd.testing.assert_frame_equal(passive.targets, candidate.targets)
        pd.testing.assert_frame_equal(
            passive.equity.drop(columns=["reason"]),
            candidate.equity.drop(columns=["reason"]),
        )
        self.assertEqual(passive.orders, candidate.orders)
        self.assertTrue((candidate.equity.target_cap == 1.0).all())

    def test_blocked_protection_reuses_one_absolute_goal(self):
        market = shock_market()
        policy = RiskAwareOwnershipPolicy(market, Config())
        units = np.full(3, 100.0)
        budget = np.zeros(3)
        first_goal = None
        for i in range(82):
            marks = policy.price[i]
            nav = float(units @ marks)
            ownership = OwnershipIntent.from_state(nav, units, marks, budget)
            observation = CloseObservation.from_inventory(
                i,
                str(market.calendar[i].date()),
                nav,
                0.0,
                units,
                units * marks / nav,
                ownership,
            )
            decision = policy.decide(observation)
            if i == 80:
                first_goal = decision.unit_targets.copy()
                self.assertEqual(policy.state, "CRISIS")
                self.assertTrue((first_goal < units).any())
            elif i == 81:
                np.testing.assert_allclose(decision.unit_targets, first_goal)
        self.assertIsNotNone(first_goal)

    def test_recovery_is_delayed_uses_new_buys_and_returns_toward_full_exposure(self):
        market = shock_market()
        result = run_risk_aware_ownership(market, cost_multiplier=0.0)
        filled = [order for order in result.orders if order["status"] == "FILLED"]
        first_sell = next(order for order in filled if order["side"] == "SELL")
        recovery_buys = [
            order
            for order in filled
            if order["side"] == "BUY" and order["date"] > first_sell["date"]
        ]
        self.assertTrue(recovery_buys)
        signal_index = {
            str(date.date()): i for i, date in enumerate(market.calendar)
        }
        self.assertGreaterEqual(
            signal_index[recovery_buys[0]["signal_date"]]
            - signal_index[first_sell["signal_date"]],
            RiskOwnershipParameters().recovery_confirmation,
        )
        self.assertGreater(float(result.equity.exposure.iloc[-1]), 0.85)
        self.assertTrue((result.targets.sum(axis=1) <= 1 + 1e-10).all())

    def test_prefix_replay_is_causal(self):
        market = shock_market()
        full = run_risk_aware_ownership(market)
        cut = market.calendar[145]
        prefix = run_risk_aware_ownership(market.prefix(cut))
        pd.testing.assert_frame_equal(full.targets.loc[:cut], prefix.targets)
        pd.testing.assert_frame_equal(full.equity.loc[:cut], prefix.equity)
        self.assertEqual(
            [order for order in full.orders if order["date"] <= str(cut.date())],
            prefix.orders,
        )

    def test_parameters_are_limited_to_frozen_coarse_structures(self):
        for value in (0.49, 0.55, 0.80, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                RiskOwnershipParameters(core_fraction=value)


if __name__ == "__main__":
    unittest.main()

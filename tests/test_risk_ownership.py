"""Focused contracts for event-scoped risk-aware ownership."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from test_core import sample_market
from techquant.config import Config
from techquant.data import Market
from techquant.evidence import metrics
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


def direct_observation(
    policy: RiskAwareOwnershipPolicy,
    session: int,
    units: np.ndarray,
    owned_units: np.ndarray,
    *,
    nav: float | None = None,
) -> CloseObservation:
    marks = policy.price[session]
    holdings = units * marks
    actual_nav = (
        float((owned_units * marks).sum()) if nav is None else float(nav)
    )
    if actual_nav <= 0:
        cash = actual_nav
        weights = np.zeros_like(units)
    else:
        cash = actual_nav - float(holdings.sum())
        weights = holdings / actual_nav
    ownership = OwnershipIntent.from_state(
        actual_nav,
        owned_units,
        marks,
        np.zeros_like(units),
    )
    return CloseObservation.from_inventory(
        session,
        str(policy.market.calendar[session].date()),
        actual_nav,
        cash,
        units,
        weights,
        ownership,
    )


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

    def test_old_account_peak_cannot_create_or_hold_crisis_by_itself(self):
        market = sample_market(2, 40)
        policy = RiskAwareOwnershipPolicy(market, Config())
        policy.trend_damage[:] = False
        policy.breadth_damage[:] = False
        policy.volatility_damage[:] = False
        policy.market_shock[:] = False
        units = np.zeros(2)
        states = []
        for session, nav in enumerate(
            (100.0, 70.0, 69.5, 71.0, 73.0, 75.0, 76.0)
        ):
            decision = policy.decide(
                direct_observation(
                    policy,
                    session,
                    units,
                    units,
                    nav=nav,
                )
            )
            states.append(policy.state)
            self.assertNotIn("SYSTEMIC_PROTECTION", decision.reason)
        self.assertNotIn("CRISIS", states)
        self.assertNotIn("DEFENSIVE", states)
        self.assertEqual(states[-1], "OPEN")

    def test_persistent_risk_reuses_one_absolute_protection_goal(self):
        market = sample_market(2, 40)
        policy = RiskAwareOwnershipPolicy(market, Config())
        policy.trend_damage[:] = False
        policy.breadth_damage[:] = False
        policy.volatility_damage[:] = False
        policy.market_shock[:] = False
        policy.trend_damage[10:14] = True
        policy.breadth_damage[10:14] = True
        units = np.full(2, 2_000.0)
        first_goal = None
        for session in range(13):
            decision = policy.decide(
                direct_observation(policy, session, units, units)
            )
            if session == 11:
                first_goal = decision.unit_targets.copy()
                self.assertEqual(policy.state, "DEFENSIVE")
                self.assertTrue((first_goal < units).any())
            elif session == 12:
                np.testing.assert_allclose(decision.unit_targets, first_goal)
        self.assertIsNotNone(first_goal)

    def test_same_cap_retains_stronger_holdings_before_weak_ones(self):
        market = sample_market(3, 40)
        policy = RiskAwareOwnershipPolicy(
            market,
            Config(),
            RiskOwnershipParameters(core_fraction=0.70),
        )
        policy.trend_damage[:] = False
        policy.breadth_damage[:] = False
        policy.volatility_damage[:] = False
        policy.market_shock[:] = False
        policy.market_shock[10] = True
        policy.trend_damage[10] = True
        policy.strength[10] = np.array([3.0, 2.0, -3.0])
        units = np.full(3, 10_000.0)

        decision = policy.decide(
            direct_observation(policy, 10, units, units)
        )

        np.testing.assert_allclose(decision.unit_targets[:2], units[:2])
        self.assertLess(decision.unit_targets[2], units[2] * 0.20)
        exposure = float(decision.weights.sum())
        account_value = float((units * policy.price[10]).sum())
        one_lot_residual = float(policy.raw_close[10].max() * 100.0 / account_value)
        self.assertGreaterEqual(exposure, 0.70)
        self.assertLess(exposure - 0.70, one_lot_residual + 1e-10)
        self.assertAlmostEqual(decision.cap, exposure)
        self.assertIn("SELECTIVE_SYSTEMIC_PROTECTION", decision.reason)

    def test_recovery_requires_new_evidence_and_completes_in_two_stages(self):
        market = sample_market(2, 40)
        policy = RiskAwareOwnershipPolicy(market, Config())
        policy.trend_damage[:] = False
        policy.breadth_damage[:] = False
        policy.volatility_damage[:] = False
        policy.market_shock[:] = False
        policy.recovery_early[:] = False
        policy.recovery_full[:] = False
        policy.trend_damage[10:12] = True
        policy.breadth_damage[10:12] = True
        policy.recovery_early[12:14] = True
        policy.recovery_full[14:16] = True

        owned = np.full(2, 2_000.0)
        units = owned.copy()
        goal = None
        for session in range(12):
            decision = policy.decide(
                direct_observation(policy, session, units, owned)
            )
            if session == 11:
                goal = decision.unit_targets.copy()
        self.assertIsNotNone(goal)
        reduced = goal < owned - 1e-10
        self.assertTrue(reduced.any())
        units = goal.copy()

        first = policy.decide(direct_observation(policy, 12, units, owned))
        np.testing.assert_allclose(first.unit_targets, goal)
        half = policy.decide(direct_observation(policy, 13, units, owned))
        self.assertEqual(policy.state, "RECOVERY")
        np.testing.assert_allclose(half.unit_targets[~reduced], owned[~reduced])
        self.assertTrue((half.unit_targets[reduced] > goal[reduced]).all())
        self.assertTrue((half.unit_targets[reduced] < owned[reduced]).all())

        units = half.unit_targets.copy()
        policy.decide(direct_observation(policy, 14, units, owned))
        full = policy.decide(direct_observation(policy, 15, units, owned))
        np.testing.assert_allclose(full.unit_targets[~reduced], owned[~reduced])
        self.assertTrue((full.unit_targets[reduced] > units[reduced]).all())
        self.assertTrue(
            (full.unit_targets[reduced] <= owned[reduced] + 1e-10).all()
        )

        units = full.unit_targets.copy()
        completed = policy.decide(
            direct_observation(policy, 16, units, owned)
        )
        self.assertEqual(policy.state, "OPEN")
        self.assertIn("FUNDED_RECOVERY_COMPLETE", completed.reason)

    def test_below_material_recovery_residual_completes_episode(self):
        market = sample_market(1, 20)
        policy = RiskAwareOwnershipPolicy(market, Config())
        policy.trend_damage[:] = False
        policy.breadth_damage[:] = False
        policy.volatility_damage[:] = False
        policy.market_shock[:] = False
        owned = np.array([2_000.0])
        units = np.array([1_900.0])
        policy._episode_level = 2
        policy._recovery_stage = 2
        policy._episode_base_units = owned.copy()
        policy._protection_goal = np.array([1_200.0])

        completed = policy.decide(
            direct_observation(
                policy,
                10,
                units,
                owned,
                nav=1_000_000.0,
            )
        )

        self.assertEqual(policy.state, "OPEN")
        self.assertIn("FUNDED_RECOVERY_COMPLETE", completed.reason)
        np.testing.assert_allclose(completed.unit_targets, owned)

    def test_sub_lot_partial_reduction_is_not_retried_forever(self):
        market = sample_market(1, 20)
        policy = RiskAwareOwnershipPolicy(market, Config())
        policy.trend_damage[:] = False
        policy.breadth_damage[:] = False
        policy.volatility_damage[:] = False
        policy.market_shock[:] = False
        policy.market_shock[0] = True
        policy.trend_damage[0] = True
        units = np.array([150.0])
        decision = policy.decide(
            direct_observation(policy, 0, units, units)
        )
        np.testing.assert_allclose(decision.unit_targets, units)
        self.assertIn("NO_EXECUTABLE_LOT", decision.reason)

    def test_shock_protects_then_recovers_without_a_low_exposure_trap(self):
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
        self.assertGreater(float(result.equity.exposure.iloc[-1]), 0.85)
        self.assertLess(metrics(result)["blocked_attempts"], 100)
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

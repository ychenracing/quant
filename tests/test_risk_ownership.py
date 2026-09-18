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

    def test_account_loss_cannot_confirm_a_market_shock(self):
        market = sample_market(2, 20)
        policy = RiskAwareOwnershipPolicy(market, Config())
        policy.trend_damage[:] = False
        policy.breadth_damage[:] = False
        policy.volatility_damage[:] = False
        policy.market_shock[:] = False
        units = np.zeros(2)
        policy.decide(direct_observation(policy, 0, units, units, nav=100.0))
        policy.market_shock[1] = True

        decision = policy.decide(
            direct_observation(policy, 1, units, units, nav=80.0)
        )

        self.assertEqual(policy.state, "CAUTION")
        self.assertIn("SHOCK,ACCOUNT_ACCELERATION", decision.reason)
        self.assertNotIn("SYSTEMIC_PROTECTION", decision.reason)

    def test_defensive_without_crisis_freezes_new_ownership_without_selling(self):
        market = sample_market(2, 20)
        policy = RiskAwareOwnershipPolicy(market, Config())
        policy.trend_damage[:] = False
        policy.breadth_damage[:] = False
        policy.volatility_damage[:] = False
        policy.market_shock[:] = False
        policy.trend_damage[10] = True
        policy.breadth_damage[10] = True
        units = np.full(2, 2_000.0)

        decision = policy.decide(
            direct_observation(policy, 10, units, units)
        )

        self.assertEqual(policy.state, "CAUTION")
        np.testing.assert_allclose(decision.unit_targets, units)
        self.assertFalse(decision.allow_new_ownership)
        self.assertNotIn("SYSTEMIC_PROTECTION", decision.reason)

    def test_persistent_risk_reuses_one_absolute_protection_goal(self):
        market = sample_market(2, 40)
        policy = RiskAwareOwnershipPolicy(market, Config())
        policy.trend_damage[:] = False
        policy.breadth_damage[:] = False
        policy.volatility_damage[:] = False
        policy.market_shock[:] = False
        policy.trend_damage[10:14] = True
        policy.breadth_damage[10:14] = True
        policy.market_shock[10:14] = True
        units = np.full(2, 2_000.0)
        first_goal = None
        for session in range(13):
            decision = policy.decide(
                direct_observation(policy, session, units, units)
            )
            if session == 10:
                first_goal = decision.unit_targets.copy()
                self.assertEqual(policy.state, "CRISIS")
                self.assertTrue((first_goal < units).any())
            elif session in (11, 12):
                np.testing.assert_allclose(decision.unit_targets, first_goal)
        self.assertIsNotNone(first_goal)

    def test_confirmed_crisis_can_raise_full_cash(self):
        market = sample_market(3, 40)
        policy = RiskAwareOwnershipPolicy(market, Config())
        policy.trend_damage[:] = False
        policy.breadth_damage[:] = False
        policy.volatility_damage[:] = False
        policy.market_shock[:] = False
        policy.market_shock[10] = True
        policy.trend_damage[10] = True
        policy.breadth_damage[10] = True
        policy.strength[10] = np.array([3.0, 2.0, -3.0])
        units = np.full(3, 10_000.0)

        decision = policy.decide(
            direct_observation(policy, 10, units, units)
        )

        self.assertLess(float(decision.weights.sum()), 0.05)
        self.assertTrue((decision.unit_targets <= 1e-10).all())
        self.assertIn("SELECTIVE_SYSTEMIC_PROTECTION", decision.reason)

    def test_confirmed_risk_clear_restores_full_ownership(self):
        market = sample_market(2, 40)
        policy = RiskAwareOwnershipPolicy(market, Config())
        policy.trend_damage[:] = False
        policy.breadth_damage[:] = False
        policy.volatility_damage[:] = False
        policy.market_shock[:] = False
        policy.trend_damage[10:12] = True
        policy.breadth_damage[10:12] = True
        policy.market_shock[10:12] = True

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
        restore = policy.decide(
            direct_observation(policy, 12, units, owned)
        )
        self.assertEqual(policy.state, "RECOVERY")
        self.assertIn("RECOVERY_FULL_ON_RISK_CLEAR", restore.reason)
        np.testing.assert_allclose(restore.unit_targets[~reduced], owned[~reduced])
        self.assertTrue((restore.unit_targets[reduced] > goal[reduced]).all())
        self.assertTrue(
            (restore.unit_targets[reduced] <= owned[reduced] + 1e-10).all()
        )

        units = restore.unit_targets.copy()
        completed = policy.decide(
            direct_observation(policy, 13, units, owned)
        )
        self.assertEqual(policy.state, "OPEN")
        self.assertIn("FUNDED_RECOVERY_COMPLETE", completed.reason)

    def test_fresh_risk_pauses_recovery_without_reselling_restored_units(self):
        market = sample_market(2, 40)
        policy = RiskAwareOwnershipPolicy(market, Config())
        policy.trend_damage[:] = False
        policy.breadth_damage[:] = False
        policy.volatility_damage[:] = False
        policy.market_shock[:] = False
        policy.trend_damage[10:12] = True
        policy.breadth_damage[10:12] = True
        policy.market_shock[10:12] = True

        owned = np.full(2, 10_000.0)
        units = owned.copy()
        goal = None
        for session in range(12):
            decision = policy.decide(
                direct_observation(policy, session, units, owned)
            )
            if session == 11:
                goal = decision.unit_targets.copy()
        self.assertIsNotNone(goal)
        units = goal.copy()
        restore = policy.decide(
            direct_observation(policy, 12, units, owned)
        )
        partially_restored = goal + 0.5 * (restore.unit_targets - goal)
        self.assertTrue((restore.unit_targets - partially_restored > 100).any())

        policy.trend_damage[13] = True
        policy.breadth_damage[13] = True
        policy.market_shock[13] = True
        paused = policy.decide(
            direct_observation(policy, 13, partially_restored, owned)
        )

        self.assertEqual(policy.state, "RECOVERY")
        np.testing.assert_allclose(paused.unit_targets, partially_restored)
        self.assertIn("RECOVERING", paused.reason)

    def test_below_material_recovery_residual_completes_episode(self):
        market = sample_market(1, 20)
        policy = RiskAwareOwnershipPolicy(market, Config())
        policy.trend_damage[:] = False
        policy.breadth_damage[:] = False
        policy.volatility_damage[:] = False
        policy.market_shock[:] = False
        owned = np.array([2_000.0])
        units = np.array([1_900.0])
        policy._episode_active = True
        policy._recovery_active = True
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

    def test_submaterial_partial_protection_is_economically_complete(self):
        market = sample_market(1, 20)
        policy = RiskAwareOwnershipPolicy(market, Config())
        current = np.array([2_000.0])
        desired = np.array([1_990.0])
        notional = float(
            (current[0] - desired[0]) * policy.price[10, 0]
        )
        self.assertLess(notional, 10_000.0)
        self.assertTrue(
            policy._executable_goal_reached(
                current, desired, 10, nav=1_000_000.0
            )
        )

    def test_sub_lot_partial_reduction_is_not_retried_forever(self):
        market = sample_market(1, 20)
        policy = RiskAwareOwnershipPolicy(market, Config())
        policy.trend_damage[:] = False
        policy.breadth_damage[:] = False
        policy.volatility_damage[:] = False
        policy.market_shock[:] = False
        policy.market_shock[0] = True
        policy.trend_damage[0] = True
        policy.breadth_damage[0] = True
        units = np.array([150.0])
        decision = policy.decide(
            direct_observation(policy, 0, units, units)
        )
        np.testing.assert_allclose(decision.unit_targets, np.zeros(1))
        self.assertIn("SELECTIVE_SYSTEMIC_PROTECTION", decision.reason)

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

    def test_finished_late_cluster_does_not_restart_on_same_lookback(self):
        market = sample_market(2, 870)
        policy = RiskAwareOwnershipPolicy(market, Config())
        policy.trend_damage[:] = False
        policy.breadth_damage[:] = False
        policy.volatility_damage[:] = False
        policy.market_shock[:] = False
        policy._first_nav = 1.0
        policy._peak_nav = 20.0
        policy.market_shock[827] = True
        policy.market_shock[828] = True
        policy.market_shock[829] = True
        owned = np.full(2, 2_000.0)
        units = owned.copy()
        started = policy.decide(
            direct_observation(policy, 829, units, owned, nav=20.0)
        )
        self.assertTrue(policy._episode_active)
        self.assertTrue(policy._extended_hold)
        self.assertIn("SELECTIVE_SYSTEMIC_PROTECTION", started.reason)
        units = started.unit_targets.copy()
        for session in range(830, 860):
            decision = policy.decide(
                direct_observation(policy, session, units, owned, nav=20.0)
            )
            units = decision.unit_targets.copy()
        self.assertFalse(policy._episode_active)
        self.assertEqual(policy.state, "OPEN")
        still_clustered = policy.decide(
            direct_observation(policy, 860, units, owned, nav=20.0)
        )
        self.assertFalse(policy._episode_active)
        self.assertNotIn("SELECTIVE_SYSTEMIC_PROTECTION", still_clustered.reason)

    def test_parameters_accept_a_global_cash_core(self):
        self.assertEqual(RiskOwnershipParameters().core_fraction, 0.0)
        self.assertEqual(
            RiskOwnershipParameters(core_fraction=0.0).core_fraction,
            0.0,
        )
        self.assertEqual(
            RiskOwnershipParameters(core_fraction=0.90).core_fraction,
            0.90,
        )
        for value in (-0.01, 1.01, float("nan")):
            with self.subTest(invalid=value), self.assertRaises(ValueError):
                RiskOwnershipParameters(core_fraction=value)



if __name__ == "__main__":
    unittest.main()

"""Engine-owned passive intent can be protected without duplicating accounting."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from test_core import sample_market
from techquant.config import Config
from techquant.data import Market
from techquant.engine import run
from techquant.policy import CloseDecision


class OpenOwnership:
    """Neutral overlay: keep acquired units and permit residual sleeve funding."""

    def __init__(self, market, config):
        self.seen = []

    def decide(self, close):
        self.seen.append(close)
        ownership = close.ownership
        if ownership is None:
            raise AssertionError("ownership observation missing")
        return CloseDecision(
            ownership.weights,
            "BUY_HOLD",
            unit_targets=ownership.units,
            allow_new_ownership=True,
        )

    def identity(self):
        return {"name": "open_ownership_test_policy"}


class OwnershipIntentTests(unittest.TestCase):
    def test_open_overlay_exactly_matches_shared_buy_hold_execution(self):
        market = sample_market(4, 140)
        for delay, cost in ((1, 0.0), (1, 1.0), (2, 3.0)):
            with self.subTest(delay=delay, cost=cost):
                benchmark = run(
                    market,
                    Config(),
                    benchmark="buy_hold",
                    delay=delay,
                    cost_multiplier=cost,
                )
                overlay = run(
                    market,
                    Config(),
                    policy_factory=OpenOwnership,
                    ownership_mode=True,
                    delay=delay,
                    cost_multiplier=cost,
                )
                pd.testing.assert_frame_equal(benchmark.equity, overlay.equity)
                pd.testing.assert_frame_equal(benchmark.targets, overlay.targets)
                self.assertEqual(benchmark.orders, overlay.orders)

    def test_open_overlay_preserves_late_listing_budget_and_retry_equivalence(self):
        base = sample_market(2, 120)
        frames = dict(base.frames)
        frames["sh688999"] = next(iter(base.frames.values())).iloc[70:].copy()
        market = Market.from_frames(frames, base.calendar, quality="synthetic")

        benchmark = run(market, benchmark="buy_hold")
        overlay = run(
            market,
            policy_factory=OpenOwnership,
            ownership_mode=True,
        )
        pd.testing.assert_frame_equal(benchmark.equity, overlay.equity)
        pd.testing.assert_frame_equal(benchmark.targets, overlay.targets)
        self.assertEqual(benchmark.orders, overlay.orders)

    def test_observation_exposes_read_only_engine_owned_intent(self):
        market = sample_market(2, 40)
        policies = []

        class Capture(OpenOwnership):
            def __init__(self, market, config):
                super().__init__(market, config)
                policies.append(self)

        run(market, policy_factory=Capture, ownership_mode=True)
        seen = policies[0].seen
        first, after_fill = seen[0], seen[1]
        self.assertIsNotNone(first.ownership)
        np.testing.assert_array_equal(first.ownership.units, np.zeros(2))
        np.testing.assert_array_equal(first.ownership.weights, np.zeros(2))
        np.testing.assert_array_equal(
            first.ownership.remaining_budget,
            np.full(2, Config().initial_cash / 2),
        )
        self.assertFalse(first.ownership.units.flags.writeable)
        self.assertFalse(first.ownership.weights.flags.writeable)
        self.assertFalse(first.ownership.remaining_budget.flags.writeable)
        self.assertTrue((after_fill.ownership.units > 0).all())
        self.assertTrue(
            (after_fill.ownership.remaining_budget < first.ownership.remaining_budget).all()
        )
        with self.assertRaises(ValueError):
            first.ownership.remaining_budget[0] = 0

    def test_policy_cannot_restore_above_engine_owned_units(self):
        market = sample_market(1, 20)
        close = market.panel("close").to_numpy()

        class InventUnits:
            def __init__(self, market, config):
                pass

            def decide(self, observation):
                target = np.array([1.0])
                weights = target * close[observation.session] / observation.nav
                return CloseDecision(
                    weights,
                    "INVENTED_OWNERSHIP",
                    unit_targets=target,
                    allow_new_ownership=False,
                )

            def identity(self):
                return {"name": "invented_ownership"}

        with self.assertRaisesRegex(ValueError, "engine-owned ownership"):
            run(market, policy_factory=InventUnits, ownership_mode=True)

    def test_protection_keeps_ownership_memory_for_causal_recovery(self):
        market = sample_market(1, 20)
        prices = market.panel("close").to_numpy()
        observations = []

        class ProtectRecover:
            def __init__(self, market, config):
                pass

            def decide(self, observation):
                observations.append(observation)
                ownership = observation.ownership
                if observation.session == 0:
                    units = ownership.units
                    allow_new = True
                elif observation.session == 2:
                    units = ownership.units * 0.5
                    allow_new = False
                elif observation.session >= 5:
                    affordable = 0.99 * observation.nav / prices[observation.session]
                    units = np.minimum(ownership.units, affordable)
                    allow_new = False
                else:
                    units = observation.units
                    allow_new = False
                weights = units * prices[observation.session] / observation.nav
                return CloseDecision(
                    weights,
                    "PROTECT_RECOVER",
                    unit_targets=units,
                    allow_new_ownership=allow_new,
                )

            def identity(self):
                return {"name": "protect_recover"}

        result = run(
            market,
            policy_factory=ProtectRecover,
            ownership_mode=True,
            cost_multiplier=0.0,
        )
        filled = [order for order in result.orders if order["status"] == "FILLED"]
        self.assertEqual([order["side"] for order in filled[:3]], ["BUY", "SELL", "BUY"])
        protected = observations[3]
        self.assertLess(protected.units[0], protected.ownership.units[0])
        self.assertEqual(
            protected.ownership.units[0], observations[2].ownership.units[0]
        )
        recovered = observations[7]
        self.assertGreater(recovered.units[0], protected.units[0])
        self.assertLessEqual(recovered.units[0], recovered.ownership.units[0])

    def test_ownership_mode_requires_explicit_fixed_unit_intent(self):
        market = sample_market(1, 20)

        class WeightOnly:
            def __init__(self, market, config):
                pass

            def decide(self, observation):
                return CloseDecision(observation.weights, "WEIGHT_ONLY")

            def identity(self):
                return {"name": "weight_only"}

        with self.assertRaisesRegex(ValueError, "fixed unit targets"):
            run(market, policy_factory=WeightOnly, ownership_mode=True)
        with self.assertRaisesRegex(ValueError, "requires a policy"):
            run(market, ownership_mode=True)

    def test_new_sleeve_funding_requires_full_open_cap(self):
        market = sample_market(1, 20)

        class PartialCap(OpenOwnership):
            def decide(self, close):
                ownership = close.ownership
                return CloseDecision(
                    ownership.weights,
                    "PARTIAL_CAP_ACQUISITION",
                    cap=0.8,
                    unit_targets=ownership.units,
                    allow_new_ownership=True,
                )

        with self.assertRaisesRegex(ValueError, "full OPEN cap"):
            run(market, policy_factory=PartialCap, ownership_mode=True)

    def test_protection_cannot_fund_new_ownership_in_the_same_signal(self):
        market = sample_market(1, 20)
        prices = market.panel("close").to_numpy()

        class SellAndAcquire:
            def __init__(self, market, config):
                pass

            def decide(self, observation):
                ownership = observation.ownership
                if observation.session == 0:
                    units = ownership.units
                    allow_new = True
                elif observation.session == 2:
                    units = ownership.units * 0.5
                    allow_new = True
                else:
                    units = observation.units
                    allow_new = False
                weights = units * prices[observation.session] / observation.nav
                return CloseDecision(
                    weights,
                    "SELL_AND_ACQUIRE",
                    unit_targets=units,
                    allow_new_ownership=allow_new,
                )

            def identity(self):
                return {"name": "sell_and_acquire"}

        with self.assertRaisesRegex(ValueError, "while reducing"):
            run(market, policy_factory=SellAndAcquire, ownership_mode=True)


if __name__ == "__main__":
    unittest.main()

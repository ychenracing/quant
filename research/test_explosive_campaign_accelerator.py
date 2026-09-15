"""Correctness contract for the preregistered explosive campaign accelerator."""
from __future__ import annotations

from dataclasses import asdict
import importlib
import importlib.util
import unittest

import numpy as np
import pandas as pd

from test_core import sample_market
from techquant.data import Market
from techquant.engine import run
from research.trend_book import SignalInputs


class ExplosiveCampaignAcceleratorTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(
            importlib.util.find_spec("research.explosive_campaign_accelerator"),
            "preregistered explosive campaign implementation is absent",
        )
        return importlib.import_module("research.explosive_campaign_accelerator")

    @staticmethod
    def inputs(allowed, score=None):
        allowed = np.asarray(allowed, dtype=bool)
        n = len(allowed)
        return SignalInputs(
            price=np.full(n, 10.0),
            ready=np.ones(n, dtype=bool),
            healthy=np.ones(n, dtype=bool),
            broken=np.zeros(n, dtype=bool),
            score=np.arange(n, 0, -1, dtype=float) if score is None else np.asarray(score, float),
            allowed=allowed,
            capacity=2,
        )

    def test_registered_pair_and_identity_are_source_bound(self):
        module = self.module()
        self.assertEqual(
            [asdict(parameters) for parameters in module.grid()],
            [{"enabled": False}, {"enabled": True}],
        )
        for invalid in (None, 0, 1, "yes"):
            with self.assertRaises(ValueError):
                module.Parameters(invalid)
        owner = module.Owner(sample_market(6, 100), module.Parameters(True))
        identity = owner.identity()
        self.assertEqual(identity["name"], "explosive_campaign_accelerator")
        self.assertFalse(identity["reference_runtime_inputs"])
        self.assertEqual(len(identity["implementation_sha256"]), 64)
        self.assertEqual(len(identity["contract_sha256"]), 64)

    def test_mask_requires_top_decile_breakout_volume_and_sector_peer(self):
        module = self.module()
        mask = module.select_explosive(
            ready=np.ones(10, dtype=bool),
            return20=np.array([.50, .08, .07, .06, .05, .04, .03, .02, .01, 0.0]),
            fresh_high=np.ones(10, dtype=bool),
            volume_confirmed=np.ones(10, dtype=bool),
            sectors=("a", "a", "b", "c", "d", "e", "f", "g", "h", "i"),
            symbols=tuple(f"s{i}" for i in range(10)),
        )
        np.testing.assert_array_equal(mask, [True, False, False, False, False,
                                             False, False, False, False, False])

    def test_mask_rejects_without_positive_same_sector_peer(self):
        module = self.module()
        mask = module.select_explosive(
            ready=np.ones(10, dtype=bool),
            return20=np.array([.50, -.01, .07, .06, .05, .04, .03, .02, .01, 0.0]),
            fresh_high=np.ones(10, dtype=bool),
            volume_confirmed=np.ones(10, dtype=bool),
            sectors=("a", "a", "b", "c", "d", "e", "f", "g", "h", "i"),
            symbols=tuple(f"s{i}" for i in range(10)),
        )
        self.assertFalse(mask.any())

    def test_treatment_only_adds_explosive_permission_and_temporary_capacity(self):
        module = self.module()
        original = self.inputs([False, True, False, False], [4.0, 3.0, 2.0, 1.0])
        treated = module.apply_accelerator(original, np.array([True, False, False, False]))
        np.testing.assert_array_equal(treated.allowed, [True, True, False, False])
        np.testing.assert_array_equal(treated.score, original.score)
        self.assertEqual(treated.capacity, 4)
        unchanged = module.apply_accelerator(original, np.zeros(4, dtype=bool))
        self.assertIs(unchanged, original)

    def test_control_matches_parent_and_treatment_is_prefix_causal(self):
        module = self.module()
        market = sample_market(12, 150)
        sectors = {symbol: ("a" if index < 3 else f"s{index}")
                   for index, symbol in enumerate(market.symbols)}
        market = Market.from_frames(
            {symbol: frame.copy() for symbol, frame in market.frames.items()},
            market.calendar,
            sectors=sectors,
            quality=market.quality,
        )
        parent = run(market, policy_factory=lambda current, cfg:
                     module.ParentOwner(current, module.ParentParameters(2)))
        control = run(market, policy_factory=lambda current, cfg:
                      module.Owner(current, module.Parameters(False)))
        pd.testing.assert_frame_equal(parent.equity, control.equity)
        pd.testing.assert_frame_equal(parent.targets, control.targets)
        self.assertEqual(parent.orders, control.orders)

        full = run(market, policy_factory=lambda current, cfg:
                   module.Owner(current, module.Parameters(True)))
        cutoff = market.calendar[115]
        short_market = market.prefix(cutoff)
        short = run(short_market, policy_factory=lambda current, cfg:
                    module.Owner(current, module.Parameters(True)))
        pd.testing.assert_frame_equal(full.equity.loc[:cutoff], short.equity)
        pd.testing.assert_frame_equal(full.targets.loc[:cutoff], short.targets)


if __name__ == "__main__":
    unittest.main()

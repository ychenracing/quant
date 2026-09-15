"""Causal leader-anchor ownership of the marginal second slot."""
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


class LeaderAnchorSlotTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(
            importlib.util.find_spec("research.leader_anchor_slots"),
            "preregistered leader-anchor implementation is absent",
        )
        return importlib.import_module("research.leader_anchor_slots")

    @staticmethod
    def inputs(allowed, score=None):
        allowed = np.asarray(allowed, dtype=bool)
        n = len(allowed)
        if score is None:
            score = np.arange(n, 0, -1, dtype=float)
        return SignalInputs(
            price=np.full(n, 10.), ready=np.ones(n, dtype=bool),
            healthy=np.ones(n, dtype=bool), broken=np.zeros(n, dtype=bool),
            score=np.asarray(score, dtype=float), allowed=allowed, capacity=2,
        )

    def test_registered_pair_and_dependency_identity(self):
        module = self.module()
        self.assertEqual(
            [asdict(parameters) for parameters in module.grid()],
            [{"enabled": False}, {"enabled": True}],
        )
        for invalid in (None, 0, 1, "yes"):
            with self.assertRaises(ValueError):
                module.Parameters(invalid)
        from research.finite_study import Study
        dependencies = Study("leader_anchor_slots").identity()["dependencies"]
        for name in ("leader_anchor_slots.py", "leader_anchor_slots_contract.json",
                     "trend_book.py", "coherent.py"):
            self.assertIn(name, dependencies)

    def test_anchor_is_preserved_and_second_slot_prefers_best_sector_peer(self):
        module = self.module()
        state = module.AnchorSlots(("a", "b", "a", "a"),
                                   ("s0", "s1", "s2", "s3"))
        original = self.inputs([True, True, True, True], [4., 3., 2., 1.])

        filtered, event = state.apply(original)

        np.testing.assert_array_equal(filtered.allowed, [True, False, True, False])
        np.testing.assert_array_equal(filtered.price, original.price)
        np.testing.assert_array_equal(filtered.score, original.score)
        self.assertEqual(event["anchor"], "s0")
        self.assertEqual(event["companion"], "s2")
        self.assertEqual(event["mode"], "SECTOR_COMPANION")
        self.assertEqual(event["suppressed"], ["s1", "s3"])

    def test_no_sector_peer_falls_back_to_unchanged_second_candidate(self):
        module = self.module()
        state = module.AnchorSlots(("a", "b", "c"), ("s0", "s1", "s2"))
        filtered, event = state.apply(self.inputs([True, True, True]))
        np.testing.assert_array_equal(filtered.allowed, [True, True, False])
        self.assertEqual(event["anchor"], "s0")
        self.assertEqual(event["companion"], "s1")
        self.assertEqual(event["mode"], "PARENT_FALLBACK")

    def test_less_than_two_candidates_are_unchanged(self):
        module = self.module()
        state = module.AnchorSlots(("a", "a", "b"), ("s0", "s1", "s2"))
        original = self.inputs([False, True, False])
        filtered, event = state.apply(original)
        np.testing.assert_array_equal(filtered.allowed, original.allowed)
        self.assertEqual(event["anchor"], "s1")
        self.assertIsNone(event["companion"])
        self.assertEqual(event["mode"], "SINGLE")

    def test_control_is_exact_parent_and_treatment_is_prefix_causal(self):
        module = self.module()
        market = sample_market(6, 145)
        parent = run(market, policy_factory=lambda current, cfg:
                     module.ParentOwner(current, module.ParentParameters(2)))
        control = run(market, policy_factory=lambda current, cfg:
                      module.Owner(current, module.Parameters(False)))
        pd.testing.assert_frame_equal(parent.equity, control.equity)
        pd.testing.assert_frame_equal(parent.targets, control.targets)
        self.assertEqual(parent.orders, control.orders)

        sectors = {symbol: ("a" if index in (0, 2, 4) else "b")
                   for index, symbol in enumerate(market.symbols)}
        themed = Market.from_frames(
            {symbol: frame.copy() for symbol, frame in market.frames.items()},
            market.calendar, sectors=sectors, quality=market.quality,
        )
        full = run(themed, policy_factory=lambda current, cfg:
                   module.Owner(current, module.Parameters(True)))
        cutoff = themed.calendar[109]
        short = run(themed.prefix(cutoff), policy_factory=lambda current, cfg:
                    module.Owner(current, module.Parameters(True)))
        pd.testing.assert_frame_equal(full.equity.loc[:cutoff], short.equity)
        pd.testing.assert_frame_equal(full.targets.loc[:cutoff], short.targets)


if __name__ == "__main__":
    unittest.main()

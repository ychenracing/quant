"""Causal same-sector campaign state on the independent price-led owner."""
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


class ThemeCampaignTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(
            importlib.util.find_spec("research.theme_campaign"),
            "preregistered causal theme campaign implementation is absent",
        )
        return importlib.import_module("research.theme_campaign")

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

    def test_registered_pair_is_one_control_and_one_treatment(self):
        module = self.module()
        self.assertEqual(
            [asdict(parameters) for parameters in module.grid()],
            [{"enabled": False}, {"enabled": True}],
        )
        for invalid in (None, 0, 1, "yes"):
            with self.assertRaises(ValueError):
                module.Parameters(invalid)
        from research.finite_study import Study
        dependencies = Study("theme_campaign").identity()["dependencies"]
        for name in ("theme_campaign.py", "theme_campaign_contract.json",
                     "trend_book.py", "coherent.py"):
            self.assertIn(name, dependencies)

    def test_top_two_same_sector_form_theme_and_only_filter_admission(self):
        module = self.module()
        state = module.ThemeState(("a", "a", "b", "b"), ("s0", "s1", "s2", "s3"), 10)
        original = self.inputs([True, True, True, True], [4., 3., 2., 1.])

        filtered, event = state.apply(21, original)

        self.assertEqual(state.active, "a")
        np.testing.assert_array_equal(filtered.allowed, [True, True, False, False])
        np.testing.assert_array_equal(filtered.broken, original.broken)
        np.testing.assert_array_equal(filtered.price, original.price)
        np.testing.assert_array_equal(filtered.score, original.score)
        self.assertEqual(event["candidate"], "a")
        self.assertEqual(event["suppressed"], ["s2", "s3"])

    def test_theme_persists_between_reviews_then_switches_on_existing_clock(self):
        module = self.module()
        state = module.ThemeState(("a", "a", "b", "b"), ("s0", "s1", "s2", "s3"), 10)
        state.apply(21, self.inputs([True, True, True, True], [4., 3., 2., 1.]))

        between, _ = state.apply(22, self.inputs([True, False, True, True], [1., .5, 4., 3.]))
        self.assertEqual(state.active, "a")
        np.testing.assert_array_equal(between.allowed, [True, False, False, False])

        reviewed, event = state.apply(30, self.inputs([True, False, True, True], [1., .5, 4., 3.]))
        self.assertEqual(state.active, "b")
        np.testing.assert_array_equal(reviewed.allowed, [False, False, True, True])
        self.assertEqual(event["previous"], "a")
        self.assertEqual(event["active"], "b")

    def test_review_without_a_cohort_deactivates_and_control_is_exact_parent(self):
        module = self.module()
        state = module.ThemeState(("a", "a", "b"), ("s0", "s1", "s2"), 10)
        state.apply(21, self.inputs([True, True, False]))
        filtered, _ = state.apply(30, self.inputs([True, False, True]))
        self.assertIsNone(state.active)
        self.assertFalse(filtered.allowed.any())

        market = sample_market(6, 145)
        parent = run(market, policy_factory=lambda current, cfg:
                     module.ParentOwner(current, module.ParentParameters(2)))
        control = run(market, policy_factory=lambda current, cfg:
                      module.Owner(current, module.Parameters(False)))
        pd.testing.assert_frame_equal(parent.equity, control.equity)
        pd.testing.assert_frame_equal(parent.targets, control.targets)
        self.assertEqual(parent.orders, control.orders)

    def test_execution_is_prefix_causal_and_excluded_names_cannot_leak(self):
        module = self.module()
        market = sample_market(6, 145)
        sectors = {symbol: ("a" if index < 3 else "b")
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
        from research.quantity_obligation import trace_digest
        owner = module.Owner(themed, module.Parameters(True))
        run(themed, policy_factory=lambda current, cfg: owner)
        self.assertEqual(len(trace_digest(owner.trace)), 64)

        names = themed.symbols[:3]
        changed_frames = {symbol: frame.copy() for symbol, frame in themed.frames.items()}
        for symbol in themed.symbols[3:]:
            columns = changed_frames[symbol].columns.get_indexer(
                ["open", "high", "low", "close", "raw_open", "raw_close"]
            )
            changed_frames[symbol].iloc[:, columns] *= 97.
        changed = Market.from_frames(changed_frames, themed.calendar,
                                     sectors=sectors, quality=themed.quality)
        first = run(themed.subset(names), policy_factory=lambda current, cfg:
                    module.Owner(current, module.Parameters(True)))
        second = run(changed.subset(names), policy_factory=lambda current, cfg:
                     module.Owner(current, module.Parameters(True)))
        pd.testing.assert_frame_equal(first.equity, second.equity)
        pd.testing.assert_frame_equal(first.targets, second.targets)
        self.assertEqual(first.orders, second.orders)


if __name__ == "__main__":
    unittest.main()

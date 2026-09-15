"""Correctness contract for the preregistered breadth/concentration owner."""
from __future__ import annotations

import importlib
import importlib.util
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from test_core import sample_market
from techquant.engine import run
from techquant.policy import CloseObservation
from research.ledger_attribution import attribute


class BreadthConcentrationTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(
            importlib.util.find_spec("research.breadth_concentration"),
            "preregistered breadth concentration implementation is absent",
        )
        return importlib.import_module("research.breadth_concentration")

    @staticmethod
    def observation(owner, session, units, cash):
        marks = owner.signals.price[session]
        units = np.asarray(units, dtype=float)
        value = units * marks
        nav = float(cash + value.sum())
        return CloseObservation.from_inventory(
            session,
            str(owner.market.calendar[session].date()),
            nav,
            cash,
            units,
            value / nav,
        )

    @staticmethod
    def controlled(owner, breadth, qualified, broken=None, score=None):
        shape = owner.signals.qualified.shape
        qualified = np.broadcast_to(np.asarray(qualified, bool), shape).copy()
        broken = np.zeros(shape, bool) if broken is None else np.broadcast_to(
            np.asarray(broken, bool), shape
        ).copy()
        score = np.broadcast_to(
            np.arange(shape[1], dtype=float) if score is None else np.asarray(score, float),
            shape,
        ).copy()
        owner.signals = SimpleNamespace(
            price=np.full(shape, 10.0),
            ready=np.ones(shape, bool),
            breadth=np.full(shape[0], breadth, dtype=float),
            qualified=qualified,
            broken=broken,
            score=score,
        )

    def test_singleton_parameters_and_identity_are_source_bound(self):
        module = self.module()
        self.assertEqual(module.Parameters(), module.Parameters())
        with self.assertRaises(TypeError):
            module.Parameters(2)
        owner = module.Owner(sample_market(4, 90), module.Parameters())
        identity = owner.identity()
        self.assertEqual(identity["name"], "breadth_concentration_ownership")
        self.assertEqual(identity["status"], "RESEARCH_NOT_ACCEPTED")
        self.assertEqual(len(identity["implementation_sha256"]), 64)
        self.assertEqual(len(identity["contract_sha256"]), 64)

    def test_majority_selects_two_and_minority_selects_every_qualified_name(self):
        module = self.module()
        qualified = np.array([True, True, True, False])
        score = np.array([1.0, 4.0, 2.0, 99.0])
        self.assertEqual(
            module.select_members(0.5, qualified, score, ("a", "b", "c", "d")),
            (1, 2),
        )
        self.assertEqual(
            module.select_members(0.49, qualified, score, ("a", "b", "c", "d")),
            (0, 1, 2),
        )

    def test_mode_change_reallocates_the_book_but_not_by_pool_name(self):
        module = self.module()
        owner = module.Owner(sample_market(4, 90), module.Parameters())
        self.controlled(owner, 0.75, [True, True, True, True], score=[1, 4, 3, 2])
        first = owner.decide(self.observation(owner, 60, [0, 0, 0, 0], 2_000_000))
        self.assertEqual(set(np.flatnonzero(first.weights)), {1, 2})
        owner.signals.breadth[61] = 0.25
        second = owner.decide(self.observation(owner, 61, [0, 100000, 100000, 0], 0))
        self.assertEqual(set(np.flatnonzero(second.weights)), {0, 1, 2, 3})
        self.assertAlmostEqual(float(second.weights.sum()), 1.0)

    def test_between_reviews_retains_actual_economic_units(self):
        module = self.module()
        owner = module.Owner(sample_market(3, 90), module.Parameters())
        self.controlled(owner, 0.8, [True, True, True], score=[3, 2, 1])
        owner.decide(self.observation(owner, 60, [0, 0, 0], 2_000_000))
        funded = self.observation(owner, 61, [90000, 90000, 0], 200000)
        owner.decide(funded)  # observe completed membership
        retained = owner.decide(self.observation(owner, 62, [90000, 90000, 0], 200000))
        np.testing.assert_allclose(retained.weights, funded.weights)
        np.testing.assert_allclose(retained.unit_targets, funded.units)

    def test_broken_holding_exits_before_the_next_review(self):
        module = self.module()
        owner = module.Owner(sample_market(3, 90), module.Parameters())
        self.controlled(owner, 0.8, [True, True, True], score=[3, 2, 1])
        owner.decide(self.observation(owner, 60, [0, 0, 0], 2_000_000))
        owner.decide(self.observation(owner, 61, [90000, 90000, 0], 200000))
        owner.signals.broken[62, 0] = True
        decision = owner.decide(self.observation(owner, 62, [90000, 90000, 0], 200000))
        self.assertEqual(decision.weights[0], 0)
        self.assertIn("BROKEN_HOLDING_EXIT", decision.reason)

    def test_replay_is_prefix_causal_and_reconciles_next_open_cash_ledger(self):
        module = self.module()
        market = sample_market(6, 145)
        full = run(market, policy_factory=lambda current, cfg: module.Owner(current, module.Parameters()))
        cutoff = market.calendar[109]
        short_market = market.prefix(cutoff)
        short = run(
            short_market,
            policy_factory=lambda current, cfg: module.Owner(current, module.Parameters()),
        )
        pd.testing.assert_frame_equal(full.equity.loc[:cutoff], short.equity)
        pd.testing.assert_frame_equal(full.targets.loc[:cutoff], short.targets)
        fills = [order for order in full.orders if order["status"] == "FILLED"]
        self.assertTrue(fills)
        self.assertTrue(all(order["signal_date"] < order["date"] for order in fills))
        self.assertTrue((full.equity.cash >= 0).all())
        _, _, checks = attribute(market, full)
        self.assertLess(checks["max_reconciliation_error"], 1e-6)


if __name__ == "__main__":
    unittest.main()

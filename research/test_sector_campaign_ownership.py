"""Correctness contract for the preregistered sector campaign owner."""
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


class SectorCampaignOwnershipTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(
            importlib.util.find_spec("research.sector_campaign_ownership"),
            "preregistered sector campaign implementation is absent",
        )
        return importlib.import_module("research.sector_campaign_ownership")

    @staticmethod
    def observation(owner, session, units, cash):
        units = np.asarray(units, dtype=float)
        values = units * owner.signals.price[session]
        nav = float(cash + values.sum())
        return CloseObservation.from_inventory(
            session,
            str(owner.market.calendar[session].date()),
            nav,
            cash,
            units,
            values / nav,
        )

    @staticmethod
    def controlled(owner, sector_eligible, security_qualified, scores, ready=None):
        shape = owner.signals.security_qualified.shape
        sector_shape = owner.signals.sector_eligible.shape
        owner.signals = SimpleNamespace(
            price=np.full(shape, 10.0),
            ready=np.ones(shape, bool) if ready is None else np.broadcast_to(
                np.asarray(ready, bool), shape
            ).copy(),
            security_qualified=np.broadcast_to(
                np.asarray(security_qualified, bool), shape
            ).copy(),
            sector_score=np.broadcast_to(
                np.asarray(scores, float), sector_shape
            ).copy(),
            sector_eligible=np.broadcast_to(
                np.asarray(sector_eligible, bool), sector_shape
            ).copy(),
        )

    def test_singleton_parameters_and_identity_are_source_bound(self):
        module = self.module()
        self.assertEqual(module.Parameters(), module.Parameters())
        with self.assertRaises(TypeError):
            module.Parameters(2)
        owner = module.Owner(sample_market(4, 100), module.Parameters())
        identity = owner.identity()
        self.assertEqual(identity["name"], "sector_campaign_ownership")
        self.assertFalse(identity["reference_runtime_inputs"])
        self.assertEqual(len(identity["implementation_sha256"]), 64)
        self.assertEqual(len(identity["contract_sha256"]), 64)

    def test_select_sector_uses_score_then_name_without_pool_branch(self):
        module = self.module()
        sectors = ("zeta", "alpha", "beta")
        self.assertEqual(
            module.select_sector(np.array([True, True, False]), np.array([2.0, 2.0, 99.0]), sectors),
            "alpha",
        )
        self.assertIsNone(
            module.select_sector(np.array([False, False, False]), np.array([2.0, 3.0, 4.0]), sectors)
        )

    def test_initial_campaign_owns_every_qualified_member_in_best_sector(self):
        module = self.module()
        market = sample_market(4, 100)
        owner = module.Owner(market, module.Parameters(), sectors=("a", "a", "b", "b"))
        self.controlled(owner, [True, True], [True, True, True, False], [1.0, 3.0])
        decision = owner.decide(self.observation(owner, 60, [0, 0, 0, 0], 2_000_000))
        self.assertEqual(owner.active_sector, "b")
        self.assertEqual(set(np.flatnonzero(decision.weights)), {2})
        self.assertAlmostEqual(float(decision.weights.sum()), 1.0)

    def test_valid_campaign_preserves_funded_units_without_routine_rebalance(self):
        module = self.module()
        market = sample_market(4, 100)
        owner = module.Owner(market, module.Parameters(), sectors=("a", "a", "b", "b"))
        self.controlled(owner, [True, True], [True, True, True, True], [3.0, 1.0])
        owner.decide(self.observation(owner, 60, [0, 0, 0, 0], 2_000_000))
        funded = self.observation(owner, 61, [150000, 50000, 0, 0], 0)
        owner.decide(funded)
        retained = owner.decide(self.observation(owner, 70, [150000, 50000, 0, 0], 0))
        np.testing.assert_allclose(retained.unit_targets, funded.units)
        self.assertEqual(owner.active_sector, "a")

    def test_campaign_switches_only_after_failure_on_review(self):
        module = self.module()
        market = sample_market(4, 100)
        owner = module.Owner(market, module.Parameters(), sectors=("a", "a", "b", "b"))
        self.controlled(owner, [True, True], [True, True, True, True], [3.0, 1.0])
        owner.decide(self.observation(owner, 60, [0, 0, 0, 0], 2_000_000))
        owner.decide(self.observation(owner, 61, [100000, 100000, 0, 0], 0))
        owner.signals.sector_eligible[:, 0] = False
        owner.signals.sector_score[:, 1] = 5.0
        before_review = owner.decide(self.observation(owner, 69, [100000, 100000, 0, 0], 0))
        np.testing.assert_allclose(before_review.unit_targets, [100000, 100000, 0, 0])
        switched = owner.decide(self.observation(owner, 70, [100000, 100000, 0, 0], 0))
        self.assertEqual(owner.active_sector, "b")
        self.assertEqual(set(np.flatnonzero(switched.weights)), {2, 3})

    def test_replay_is_prefix_causal_next_open_cash_only_and_reconciled(self):
        module = self.module()
        market = sample_market(6, 150)
        sectors = ("a", "a", "b", "b", "c", "c")
        full = run(
            market,
            policy_factory=lambda current, cfg: module.Owner(
                current, module.Parameters(), sectors=sectors
            ),
        )
        cutoff = market.calendar[115]
        short_market = market.prefix(cutoff)
        short = run(
            short_market,
            policy_factory=lambda current, cfg: module.Owner(
                current, module.Parameters(), sectors=sectors
            ),
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

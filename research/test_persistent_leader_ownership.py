"""Contracts for market-gated entry with security-owned exits."""
from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
import pandas as pd

from test_core import sample_market
from techquant.config import Config
from techquant.engine import run
from techquant.features import build_features
from techquant.strategy import RiskState
from research.persistent_leader_ownership import (
    EntryOnlyRisk,
    Owner,
    Parameters,
)
from research.trend_book import Owner as ParentOwner, Parameters as ParentParameters


class PersistentLeaderOwnershipTests(unittest.TestCase):
    def test_registered_grid_is_one_fixed_pair(self):
        from research.persistent_leader_ownership import grid

        self.assertEqual(grid(), [Parameters(False), Parameters(True)])
        with self.assertRaises(ValueError):
            Parameters(1)

    def test_entry_only_risk_observes_but_does_not_liquidate_market_shock(self):
        market = sample_market(3, 170)
        config = Config()
        features = build_features(market, config)
        features = replace(
            features,
            market_return=np.full(170, -0.10),
            shock_fraction=np.ones(170),
        )
        trace = []
        risk = EntryOnlyRisk(trace)
        cap, reason = risk.update(100, features, [100.0] * 101, config)
        self.assertEqual(cap, 1.0)
        self.assertEqual(reason, "SECURITY_OWNED_EXIT")
        self.assertEqual(trace[-1]["shadow_reason"], "CROSS_SECTION_SHOCK")
        self.assertEqual(trace[-1]["shadow_cap"], 0.0)

    def test_no_ready_security_keeps_new_risk_closed(self):
        market = sample_market(3, 170)
        config = Config()
        features = build_features(market, config)
        features = replace(features, ready=np.zeros_like(features.ready))
        risk = EntryOnlyRisk([])
        cap, reason = risk.update(0, features, [100.0], config)
        self.assertEqual((cap, reason), (0.0, "WARMUP_OR_NO_FRESH_QUOTES"))

    def test_disabled_owner_is_exact_parent_control(self):
        market = sample_market(4, 170)
        parent = run(
            market,
            Config(),
            policy_factory=lambda m, _c: ParentOwner(m, ParentParameters(2)),
        )
        control = run(
            market,
            Config(),
            policy_factory=lambda m, _c: Owner(m, Parameters(False)),
        )
        pd.testing.assert_frame_equal(parent.equity, control.equity)
        pd.testing.assert_frame_equal(parent.targets, control.targets)
        self.assertEqual(parent.orders, control.orders)

    def test_treatment_remains_cash_funded_long_only_and_causal(self):
        market = sample_market(4, 170)
        full = run(
            market,
            Config(),
            policy_factory=lambda m, _c: Owner(m, Parameters(True)),
        )
        prefix_market = market.prefix(str(market.calendar[130].date()))
        prefix = run(
            prefix_market,
            Config(),
            policy_factory=lambda m, _c: Owner(m, Parameters(True)),
        )
        pd.testing.assert_frame_equal(full.equity.iloc[: len(prefix.equity)], prefix.equity)
        pd.testing.assert_frame_equal(full.targets.iloc[: len(prefix.targets)], prefix.targets)
        self.assertTrue((full.targets >= 0).all().all())
        self.assertTrue((full.targets.sum(axis=1) <= 1 + 1e-10).all())
        self.assertTrue((full.equity.cash >= -1e-6).all())

    def test_shadow_state_never_licenses_more_than_cash_funded_one(self):
        market = sample_market(3, 170)
        config = Config()
        features = build_features(market, config)
        risk = EntryOnlyRisk([])
        for i in range(90, 110):
            cap, _ = risk.update(i, features, [100.0] * (i + 1), config)
            self.assertIn(cap, (0.0, 1.0))
            self.assertLessEqual(cap, 1.0)
        self.assertIsInstance(risk.shadow, RiskState)


if __name__ == "__main__":
    unittest.main()

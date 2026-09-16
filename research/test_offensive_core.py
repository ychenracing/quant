"""Contracts for the zero-grid offensive core candidate."""
from dataclasses import asdict, replace
import importlib
import importlib.util
import unittest

import numpy as np
import pandas as pd

from test_core import sample_market
from techquant.engine import run
from techquant.policy import CloseObservation
from research.trend_book import Owner as ParentOwner, Parameters as ParentParameters


def observation(owner, session, units, cash):
    units = np.asarray(units, dtype=float)
    marks = owner.price_signals.price[session]
    values = units * marks
    nav = float(cash + values.sum())
    return CloseObservation.from_inventory(
        session, str(owner.market.calendar[session].date()), nav, cash, units, values / nav
    )


class OffensiveCoreTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(
            importlib.util.find_spec("research.offensive_core"),
            "preregistered offensive core implementation is absent",
        )
        return importlib.import_module("research.offensive_core")

    def test_registered_pair_and_control_is_exact_parent(self):
        module = self.module()
        self.assertEqual(
            [asdict(p) for p in module.grid()],
            [{"enabled": False}, {"enabled": True}],
        )
        for bad in (None, 0, 1, "yes"):
            with self.assertRaises(ValueError):
                module.Parameters(bad)
        market = sample_market(5, 145)
        parent = run(
            market,
            policy_factory=lambda current, cfg: ParentOwner(current, ParentParameters(2)),
        )
        control = run(
            market,
            policy_factory=lambda current, cfg: module.Owner(current, module.Parameters(False)),
        )
        pd.testing.assert_frame_equal(parent.equity, control.equity)
        pd.testing.assert_frame_equal(parent.targets, control.targets)
        self.assertEqual(parent.orders, control.orders)

    def test_registered_with_existing_paired_study_overlay(self):
        self.module()
        from research.finite_study import Study
        study = Study("offensive_core")
        self.assertEqual(study.family, "offensive_core")
        self.assertEqual(len(study.module.grid()), 2)
        identity = study.identity()
        self.assertIn("offensive_core.py", identity["dependencies"])
        self.assertIn("offensive_core_contract.json", identity["dependencies"])

    def test_treatment_targets_full_exposure_without_portfolio_risk_cap(self):
        module = self.module()
        market = sample_market(5, 145)
        owner = module.Owner(market, module.Parameters(True))
        session = 100  # Config.rebalance boundary.
        d = owner.decide(observation(owner, session, np.zeros(5), 2_000_000.))
        self.assertAlmostEqual(float(d.weights.sum()), 1.0, places=12)
        self.assertEqual(np.count_nonzero(d.weights), 2)
        self.assertEqual(d.cap, 1.0)
        self.assertIn("OFFENSIVE_SELECTION", d.reason)

    def test_same_members_keep_actual_drift_between_reviews(self):
        module = self.module()
        market = sample_market(3, 145)
        owner = module.Owner(market, module.Parameters(True))
        owner.decide(observation(owner, 100, np.zeros(3), 2_000_000.))
        units = np.array([0.0, 12_000.0, 18_000.0])
        score = owner.features.score.copy()
        score[101] = np.array([-1.0, 2.0, 3.0])
        owner.features = replace(owner.features, score=score)
        current = observation(owner, 101, units, 500_000.0)
        d = owner.decide(current)
        np.testing.assert_allclose(d.weights, current.weights, rtol=0, atol=1e-12)
        self.assertIn("OFFENSIVE_RETAIN", d.reason)

    def test_broad_market_weakness_does_not_liquidate_intact_holdings(self):
        module = self.module()
        market = sample_market(2, 145)
        owner = module.Owner(market, module.Parameters(True))
        units = np.array([10_000.0, 10_000.0])
        current = observation(owner, 101, units, 500_000.0)
        market_state = owner.trend.market.copy()
        market_state[101] = False
        owner.trend = replace(owner.trend, market=market_state)
        d = owner.decide(current)
        np.testing.assert_allclose(d.weights, current.weights, rtol=0, atol=1e-12)
        self.assertNotIn("PORTFOLIO", d.reason)

    def test_security_break_cannot_reenter_on_same_close_even_if_entry_stays_true(self):
        module = self.module()
        market = sample_market(3, 145)
        owner = module.Owner(market, module.Parameters(True))
        units = np.array([10_000.0, 10_000.0, 0.0])
        session = 101
        exit_state = owner.trend.exit.copy()
        entry_state = owner.trend.entry.copy()
        exit_state[session, 0] = True
        entry_state[session, 0] = True
        owner.trend = replace(owner.trend, exit=exit_state, entry=entry_state)
        score = owner.features.score.copy()
        score[session] = np.array([100.0, 2.0, 4.0])
        owner.features = replace(owner.features, score=score)
        d = owner.decide(observation(owner, session, units, 500_000.0))
        self.assertEqual(float(d.weights[0]), 0.0)
        self.assertIn("SECURITY_EXIT", d.reason)

    def test_security_break_exits_immediately_and_can_fill_vacancy(self):
        module = self.module()
        market = sample_market(3, 145)
        owner = module.Owner(market, module.Parameters(True))
        units = np.array([10_000.0, 10_000.0, 0.0])
        exit_state = owner.trend.exit.copy()
        exit_state[101, 0] = True
        owner.trend = replace(owner.trend, exit=exit_state)
        score = owner.features.score.copy()
        score[101] = np.array([1.0, 2.0, 4.0])
        owner.features = replace(owner.features, score=score)
        d = owner.decide(observation(owner, 101, units, 500_000.0))
        self.assertEqual(float(d.weights[0]), 0.0)
        self.assertGreater(float(d.weights[2]), 0.0)
        self.assertAlmostEqual(float(d.weights.sum()), 1.0, places=12)
        self.assertIn("SECURITY_EXIT", d.reason)


class OffensiveCoreScreenTests(unittest.TestCase):
    def test_alpha_screen_requires_all_scope_nonregression_and_one_gain(self):
        self.assertIsNotNone(importlib.util.find_spec("research.offensive_core_screen"))
        screen = importlib.import_module("research.offensive_core_screen")
        passing = [
            {"scope": "union", "wealth_change": 0.10},
            {"scope": "chatgpt_5", "wealth_change": 0.0},
            {"scope": "joint_optical_leader_removal", "wealth_change": 0.02},
        ]
        failing = [*passing[:-1], {"scope": "joint_optical_leader_removal", "wealth_change": -1e-4}]
        self.assertTrue(screen.alpha_decision(passing)["advance"])
        self.assertFalse(screen.alpha_decision(failing)["advance"])
        self.assertFalse(screen.alpha_decision([
            {"scope": "union", "wealth_change": 0.0},
            {"scope": "chatgpt_5", "wealth_change": 0.0},
            {"scope": "joint_optical_leader_removal", "wealth_change": 0.0},
        ])["advance"])


if __name__ == "__main__":
    unittest.main()

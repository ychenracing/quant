"""Contracts for the preregistered confirmed market-shock lifecycle."""
from dataclasses import asdict, replace
import importlib
import importlib.util
import unittest

import numpy as np
import pandas as pd

from test_core import sample_market
from techquant.config import Config
from techquant.engine import run
from techquant.features import build_features
from research.observed_admission_completion import (
    Owner as ParentOwner,
    Parameters as ParentParameters,
)
from research.test_quantity_obligation import observe


class ConfirmedShockTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(
            importlib.util.find_spec("research.confirmed_shock"),
            "preregistered confirmed-shock implementation is absent",
        )
        return importlib.import_module("research.confirmed_shock")

    def features(self, sessions=170):
        config = Config()
        features = build_features(sample_market(2, sessions), config)
        return config, replace(
            features,
            ready=np.ones_like(features.ready, dtype=bool),
            entry=np.zeros_like(features.entry, dtype=bool),
            exit=np.zeros_like(features.exit, dtype=bool),
            breadth=np.ones(sessions),
            market_return=np.zeros(sessions),
            market_vol=np.full(sessions, .01),
            market_dd=np.zeros(sessions),
            weak=np.zeros(sessions, dtype=bool),
            shock_fraction=np.zeros(sessions),
        )

    @staticmethod
    def shock(features, *sessions):
        market_return = features.market_return.copy()
        breadth = features.breadth.copy()
        shock_fraction = features.shock_fraction.copy()
        for session in sessions:
            market_return[session] = -.10
            breadth[session] = .10
            shock_fraction[session] = 1.
        return replace(
            features,
            market_return=market_return,
            breadth=breadth,
            shock_fraction=shock_fraction,
        )

    def test_registered_single_candidate_and_source_identity(self):
        module = self.module()
        self.assertEqual(
            [asdict(parameters) for parameters in module.grid()],
            [{"confirm_market_shock": False}, {"confirm_market_shock": True}],
        )
        for bad in (None, 0, 1, "yes"):
            with self.assertRaises(ValueError):
                module.Parameters(bad)
        from research.finite_study import Study
        dependencies = Study("confirmed_shock").identity()["dependencies"]
        self.assertIn("confirmed_shock.py", dependencies)
        self.assertIn("confirmed_shock_contract.json", dependencies)

    def test_control_is_exact_parent_account(self):
        module = self.module()
        market = sample_market(3, 145)
        parent = run(
            market,
            policy_factory=lambda current, config: ParentOwner(
                current, ParentParameters(), config=config
            ),
        )
        control = run(
            market,
            policy_factory=lambda current, config: module.Owner(
                current, module.Parameters(False), config=config
            ),
        )
        pd.testing.assert_frame_equal(parent.equity, control.equity)
        pd.testing.assert_frame_equal(parent.targets, control.targets)
        self.assertEqual(parent.orders, control.orders)

    def test_first_isolated_market_shock_halves_then_next_shock_zeros(self):
        module = self.module()
        config, features = self.features()
        features = self.shock(features, 100, 101)
        state = module.ConfirmedMarketShockState(cap=1.)
        cap, reason = state.update(100, features, [100.] * 101, config)
        self.assertEqual(cap, .5)
        self.assertEqual(reason, "MARKET_SHOCK_CONFIRMATION_HALF_RISK")
        self.assertTrue(state.confirmation_pending)
        cap, reason = state.update(101, features, [100.] * 102, config)
        self.assertEqual(cap, 0.)
        self.assertEqual(reason, "CONFIRMED_CROSS_SECTION_SHOCK")

    def test_nonshock_clears_confirmation_and_half_cap_shock_still_zeros(self):
        module = self.module()
        config, features = self.features()
        features = self.shock(features, 100, 102)
        state = module.ConfirmedMarketShockState(cap=1.)
        state.update(100, features, [100.] * 101, config)
        state.update(101, features, [100.] * 102, config)
        self.assertFalse(state.confirmation_pending)
        cap, _ = state.update(102, features, [100.] * 103, config)
        self.assertEqual(cap, 0.)

    def test_account_drawdown_has_priority_over_same_day_market_shock(self):
        module = self.module()
        config, features = self.features()
        features = self.shock(features, 100)
        state = module.ConfirmedMarketShockState(cap=1., episode_peak=100.)
        cap, reason = state.update(100, features, [100., 81.9], config)
        self.assertEqual(cap, 0.)
        self.assertEqual(reason, "PORTFOLIO_DRAWDOWN_SHOCK")
        self.assertFalse(state.confirmation_pending)

    def test_cap_cut_cannot_fund_a_new_position_and_security_exit_stays_latched(self):
        module = self.module()
        market = sample_market(2, 130)
        owner = module.Owner(market, module.Parameters(True))
        inner = owner.inner
        inner.features = self.shock(inner.features, 40)
        inner.ready[:] = True
        inner.features = replace(
            inner.features,
            ready=np.ones_like(inner.features.ready, dtype=bool),
            entry=np.ones_like(inner.features.entry, dtype=bool),
            exit=np.zeros_like(inner.features.exit, dtype=bool),
            score=np.ones_like(inner.features.score),
        )
        inner.risk.cap = 1.
        decision = owner.decide(observe(inner, 40, [10_000., 0.], cash=1_000_000.))
        self.assertLessEqual(decision.unit_targets[0], 10_000.)
        self.assertEqual(decision.unit_targets[1], 0.)
        inner.stop[0] = inner.features.close[41, 0] * 1.01
        retry = owner.decide(observe(inner, 41, [10_000., 0.], cash=1_000_000.))
        self.assertEqual(retry.unit_targets[0], 0.)
        self.assertTrue(inner.exit_pending[0])


if __name__ == "__main__":
    unittest.main()

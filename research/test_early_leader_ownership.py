"""Contracts for prefix-confirmed leader ownership."""
from __future__ import annotations

from dataclasses import asdict, replace
import importlib
import importlib.util
import unittest

import numpy as np
import pandas as pd

from test_core import sample_market
from techquant.config import Config
from techquant.engine import run
from techquant.policy import CloseObservation
from research.coherent import SignalInputs
from research.trend_book import Owner as ParentOwner, Parameters as ParentParameters


def observation(owner, session, units, cash=0.0):
    units = np.asarray(units, dtype=float)
    price = np.nan_to_num(owner.features.close[session], nan=0.0)
    values = units * price
    nav = float(cash + values.sum())
    return CloseObservation.from_inventory(
        session,
        str(owner.market.calendar[session].date()),
        nav,
        cash,
        units,
        values / nav,
    )


class EarlyLeaderOwnershipTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(
            importlib.util.find_spec("research.early_leader_ownership"),
            "preregistered early-leader implementation is absent",
        )
        return importlib.import_module("research.early_leader_ownership")

    def owner(self, enabled=True):
        module = self.module()
        owner = module.Owner(sample_market(3, 170), module.Parameters(enabled))
        shape = owner.features.ready.shape
        owner.features = replace(
            owner.features,
            ready=np.ones(shape, dtype=bool),
            entry=np.zeros(shape, dtype=bool),
            exit=np.zeros(shape, dtype=bool),
            breadth=np.ones(shape[0]),
            market_return=np.zeros(shape[0]),
            market_vol=np.full(shape[0], 0.01),
            market_dd=np.zeros(shape[0]),
            weak=np.zeros(shape[0], dtype=bool),
            shock_fraction=np.zeros(shape[0]),
        )
        owner.s = replace(
            owner.s,
            exit=np.zeros(shape, dtype=bool),
            entry=np.ones(shape, dtype=bool),
            market=np.ones(shape[0], dtype=bool),
        )
        owner.risk.cap = 1.0
        return owner

    @staticmethod
    def shock(owner, session):
        f = owner.features
        market_return = f.market_return.copy()
        breadth = f.breadth.copy()
        shock_fraction = f.shock_fraction.copy()
        market_return[session] = -0.10
        breadth[session] = 0.10
        shock_fraction[session] = 1.0
        owner.features = replace(
            f,
            market_return=market_return,
            breadth=breadth,
            shock_fraction=shock_fraction,
        )

    def test_registered_pair_and_disabled_control_is_exact_parent(self):
        module = self.module()
        self.assertEqual(
            [asdict(parameters) for parameters in module.grid()],
            [{"enabled": False}, {"enabled": True}],
        )
        for bad in (None, 0, 1, "yes"):
            with self.assertRaises(ValueError):
                module.Parameters(bad)
        market = sample_market(4, 170)
        parent = run(
            market,
            Config(),
            policy_factory=lambda current, _config: ParentOwner(
                current, ParentParameters(2)
            ),
        )
        control = run(
            market,
            Config(),
            policy_factory=lambda current, _config: module.Owner(
                current, module.Parameters(False)
            ),
        )
        pd.testing.assert_frame_equal(parent.equity, control.equity)
        pd.testing.assert_frame_equal(parent.targets, control.targets)
        self.assertEqual(parent.orders, control.orders)

    def test_confirmation_uses_actual_basis_existing_rank_and_fixed_clock(self):
        owner = self.owner()
        session = 100
        units = np.array([10_000.0, 0.0, 0.0])
        current = observation(owner, session, units)
        owner._observe_inventory(current)
        owner.acquisition_open[0] = owner.features.close[session, 0] * 0.9
        observed = SignalInputs(
            owner.features.close[session],
            np.ones(3, dtype=bool),
            np.ones(3, dtype=bool),
            np.zeros(3, dtype=bool),
            np.array([3.0, 2.0, 1.0]),
            np.ones(3, dtype=bool),
            2,
        )

        owner._update_confirmation(current, observed)

        self.assertTrue(owner.confirmed[0])
        self.assertEqual(owner.episode_age[0], 0)
        self.assertEqual(owner.trace[-1]["rank"], 1)
        owner.confirmed[:] = False
        owner.episode_age[0] = Config().rebalance
        owner._update_confirmation(current, observed)
        self.assertFalse(owner.confirmed[0])

    def test_confirmation_rejects_profitless_or_out_of_capacity_holdings(self):
        owner = self.owner()
        session = 100
        units = np.array([10_000.0, 10_000.0, 10_000.0])
        current = observation(owner, session, units)
        owner._observe_inventory(current)
        price = owner.features.close[session]
        owner.acquisition_open = np.array([price[0] * 1.1, price[1] * 0.9, price[2] * 0.9])
        observed = SignalInputs(
            price,
            np.ones(3, dtype=bool),
            np.ones(3, dtype=bool),
            np.zeros(3, dtype=bool),
            np.array([3.0, 2.0, 1.0]),
            np.ones(3, dtype=bool),
            2,
        )

        owner._update_confirmation(current, observed)

        np.testing.assert_array_equal(owner.confirmed, np.array([False, True, False]))

    def test_market_shock_retains_only_confirmed_actual_inventory(self):
        owner = self.owner()
        session = 100
        units = np.array([10_000.0, 10_000.0, 0.0])
        owner.observed_units = units.copy()
        owner.acquisition_open = owner.features.close[session] * 0.9
        owner.episode_age[:] = 2
        owner.confirmed[:] = np.array([True, False, False])
        self.shock(owner, session)

        decision = owner.decide(observation(owner, session, units))

        self.assertEqual(decision.unit_targets[0], units[0])
        self.assertEqual(decision.unit_targets[1], 0.0)
        self.assertIn("EARLY_LEADER_OWNERSHIP", decision.reason)
        self.assertFalse(owner.exit_pending[0])
        self.assertTrue(owner.exit_pending[1])

    def test_account_drawdown_and_existing_obligation_remain_authoritative(self):
        owner = self.owner()
        session = 100
        units = np.array([10_000.0, 10_000.0, 0.0])
        owner.observed_units = units.copy()
        owner.acquisition_open = owner.features.close[session] * 0.9
        owner.episode_age[:] = 2
        owner.confirmed[:] = np.array([True, True, False])
        owner.exit_pending[1] = True
        self.shock(owner, session)
        current = observation(owner, session, units)
        owner.risk.episode_peak = current.nav / 0.75
        owner.history.append(current.nav / 0.95)

        decision = owner.decide(current)

        np.testing.assert_array_equal(decision.unit_targets, np.zeros(3))
        self.assertNotIn("EARLY_LEADER_OWNERSHIP", decision.reason)
        self.assertTrue(owner.exit_pending[1])

    def test_treatment_is_prefix_causal_cash_funded_and_long_only(self):
        module = self.module()
        market = sample_market(4, 170)
        full = run(
            market,
            Config(),
            policy_factory=lambda current, _config: module.Owner(
                current, module.Parameters(True)
            ),
        )
        prefix_market = market.prefix(str(market.calendar[130].date()))
        prefix = run(
            prefix_market,
            Config(),
            policy_factory=lambda current, _config: module.Owner(
                current, module.Parameters(True)
            ),
        )
        pd.testing.assert_frame_equal(full.equity.iloc[: len(prefix.equity)], prefix.equity)
        pd.testing.assert_frame_equal(full.targets.iloc[: len(prefix.targets)], prefix.targets)
        self.assertTrue((full.targets >= 0).all().all())
        self.assertTrue((full.targets.sum(axis=1) <= 1 + 1e-10).all())
        self.assertTrue((full.equity.cash >= -1e-6).all())


if __name__ == "__main__":
    unittest.main()

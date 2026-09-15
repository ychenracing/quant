"""Contracts for the preregistered profitable-trend shock shield."""
from dataclasses import asdict, replace
import importlib
import importlib.util
import unittest

import numpy as np
import pandas as pd

from test_core import sample_market
from techquant.engine import run
from techquant.policy import CloseObservation
from research.observed_admission_completion import (
    Owner as ParentOwner,
    Parameters as ParentParameters,
)


def observation(owner, session, units, cash=0.):
    units = np.asarray(units, dtype=float)
    price = np.nan_to_num(owner.inner.features.close[session], nan=0.)
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


class ProfitTrendShieldTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(
            importlib.util.find_spec("research.profit_trend_shield"),
            "preregistered profit-trend shield implementation is absent",
        )
        return importlib.import_module("research.profit_trend_shield")

    def owner(self, enabled=True):
        module = self.module()
        owner = module.Owner(sample_market(2, 170), module.Parameters(enabled))
        inner = owner.inner
        shape = inner.features.ready.shape
        inner.features = replace(
            inner.features,
            ready=np.ones(shape, dtype=bool),
            entry=np.zeros(shape, dtype=bool),
            exit=np.zeros(shape, dtype=bool),
            breadth=np.ones(shape[0]),
            market_return=np.zeros(shape[0]),
            market_vol=np.full(shape[0], .01),
            market_dd=np.zeros(shape[0]),
            weak=np.zeros(shape[0], dtype=bool),
            shock_fraction=np.zeros(shape[0]),
        )
        inner.ready[:] = True
        inner.stop[:] = 0.
        inner.risk.cap = 1.
        return owner

    @staticmethod
    def shock(owner, session):
        f = owner.inner.features
        market_return = f.market_return.copy()
        breadth = f.breadth.copy()
        shock_fraction = f.shock_fraction.copy()
        market_return[session] = -.10
        breadth[session] = .10
        shock_fraction[session] = 1.
        owner.inner.features = replace(
            f,
            market_return=market_return,
            breadth=breadth,
            shock_fraction=shock_fraction,
        )

    @staticmethod
    def mark_campaigns(owner, session, units, strong=(True, False)):
        price = owner.inner.features.close[session]
        owner.observed_units = np.asarray(units, dtype=float).copy()
        owner.entry_open = np.where(strong, price * .5, price * 1.1)
        owner.ema20[session] = np.where(strong, price * .9, price * 1.1)
        owner.ema60[session] = np.where(strong, price * .8, price * 1.1)
        owner.return20[session] = np.where(strong, .2, -.1)
        owner.return60[session] = np.where(strong, .5, -.1)

    def test_registered_pair_and_control_is_exact_parent(self):
        module = self.module()
        self.assertEqual(
            [asdict(parameters) for parameters in module.grid()],
            [{"enabled": False}, {"enabled": True}],
        )
        for bad in (None, 0, 1, "yes"):
            with self.assertRaises(ValueError):
                module.Parameters(bad)
        from research.finite_study import Study
        dependencies = Study("profit_trend_shield").identity()["dependencies"]
        self.assertIn("profit_trend_shield.py", dependencies)
        self.assertIn("profit_trend_shield_contract.json", dependencies)
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

    def test_pure_market_shock_retains_only_strong_profitable_inventory(self):
        owner = self.owner()
        session = 100
        units = np.array([10_000., 10_000.])
        self.mark_campaigns(owner, session, units)
        self.shock(owner, session)

        decision = owner.decide(observation(owner, session, units))

        self.assertEqual(decision.unit_targets[0], units[0])
        self.assertEqual(decision.unit_targets[1], 0.)
        self.assertIn("PROFIT_TREND_SHIELD", decision.reason)
        self.assertFalse(owner.inner.exit_pending[0])
        self.assertTrue(np.isinf(owner.inner.reduction_ceiling[0]))
        self.assertEqual(owner.inner.reduction_ceiling[1], 0.)
        self.assertLessEqual(decision.unit_targets[0], units[0])

    def test_account_drawdown_and_existing_security_obligation_cannot_be_shielded(self):
        owner = self.owner()
        session = 100
        units = np.array([10_000., 10_000.])
        self.mark_campaigns(owner, session, units, strong=(True, True))
        self.shock(owner, session)
        current = observation(owner, session, units)
        owner.inner.risk.episode_peak = current.nav / .75
        owner.inner.history.append(current.nav / .95)
        owner.inner.exit_pending[1] = True

        decision = owner.decide(current)

        np.testing.assert_array_equal(decision.unit_targets, np.zeros(2))
        self.assertNotIn("PROFIT_TREND_SHIELD", decision.reason)
        self.assertTrue(owner.inner.exit_pending[1])

    def test_preexisting_reduction_is_not_relaxed(self):
        owner = self.owner()
        session = 100
        units = np.array([10_000., 10_000.])
        self.mark_campaigns(owner, session, units, strong=(True, True))
        owner.inner.reduction_ceiling[0] = 6_000.
        self.shock(owner, session)

        decision = owner.decide(observation(owner, session, units))

        self.assertLessEqual(decision.unit_targets[0], 6_000.)
        self.assertEqual(decision.unit_targets[1], units[1])

    def test_shielded_inventory_is_not_mechanically_sold_on_next_healthy_close(self):
        owner = self.owner()
        units = np.array([10_000., 0.])
        self.mark_campaigns(owner, 100, units, strong=(True, False))
        self.shock(owner, 100)
        first = owner.decide(observation(owner, 100, units, cash=100_000.))
        self.assertEqual(first.unit_targets[0], units[0])
        self.assertEqual(owner.inner.risk.cap, 0.)
        self.assertEqual(owner.shield_units[0], units[0])

        price = owner.inner.features.close[101]
        owner.ema20[101] = price * .9
        owner.ema60[101] = price * .8
        owner.return20[101] = .2
        owner.return60[101] = .5
        second = owner.decide(observation(owner, 101, units, cash=100_000.))

        self.assertEqual(second.unit_targets[0], units[0])
        self.assertIn("PROFIT_TREND_SHIELD", second.reason)

    def test_latched_shield_clears_when_fixed_strong_definition_fails(self):
        owner = self.owner()
        units = np.array([10_000., 0.])
        self.mark_campaigns(owner, 100, units, strong=(True, False))
        self.shock(owner, 100)
        owner.decide(observation(owner, 100, units, cash=100_000.))

        price = owner.inner.features.close[101]
        owner.ema20[101] = price * 1.1
        owner.ema60[101] = price * .8
        owner.return20[101] = .2
        owner.return60[101] = .5
        decision = owner.decide(observation(owner, 101, units, cash=100_000.))

        self.assertEqual(decision.unit_targets[0], 0.)
        self.assertEqual(owner.shield_units[0], 0.)

    def test_latched_shield_cannot_override_a_non_market_risk_condition(self):
        owner = self.owner()
        units = np.array([10_000., 0.])
        self.mark_campaigns(owner, 100, units, strong=(True, False))
        self.shock(owner, 100)
        owner.decide(observation(owner, 100, units, cash=100_000.))

        price = owner.inner.features.close[101]
        owner.ema20[101] = price * .9
        owner.ema60[101] = price * .8
        owner.return20[101] = .2
        owner.return60[101] = .5
        f = owner.inner.features
        market_dd = f.market_dd.copy()
        weak = f.weak.copy()
        breadth = f.breadth.copy()
        market_dd[101] = .30
        weak[101] = True
        breadth[101] = .10
        owner.inner.features = replace(
            f, market_dd=market_dd, weak=weak, breadth=breadth
        )
        decision = owner.decide(observation(owner, 101, units, cash=100_000.))

        self.assertEqual(decision.unit_targets[0], 0.)
        self.assertEqual(owner.shield_units[0], 0.)


if __name__ == "__main__":
    unittest.main()

"""Contracts for the preregistered actual-basis campaign failure exit."""
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


def observation(owner, session, units, cash=100_000.):
    units = np.asarray(units, dtype=float)
    price = owner.price_signals.price[session]
    values = units * price
    nav = float(cash + values.sum())
    return CloseObservation.from_inventory(
        session, str(owner.market.calendar[session].date()), nav, cash,
        units, values / nav,
    )


class BasisFailureExitTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(
            importlib.util.find_spec("research.basis_failure_exit"),
            "preregistered basis-failure implementation is absent",
        )
        return importlib.import_module("research.basis_failure_exit")

    def owner(self, enabled=True):
        module = self.module()
        owner = module.Owner(sample_market(1, 170), module.Parameters(enabled))
        shape = owner.features.ready.shape
        owner.features = replace(
            owner.features,
            ready=np.ones(shape, dtype=bool),
            breadth=np.ones(shape[0]),
            market_return=np.zeros(shape[0]),
            market_vol=np.full(shape[0], .01),
            market_dd=np.zeros(shape[0]),
            weak=np.zeros(shape[0], dtype=bool),
            shock_fraction=np.zeros(shape[0]),
        )
        owner.s = replace(
            owner.s,
            exit=np.zeros(shape, dtype=bool),
            entry=np.zeros(shape, dtype=bool),
            market=np.ones(shape[0], dtype=bool),
        )
        owner.price_signals = replace(
            owner.price_signals,
            ready=np.ones(shape, dtype=bool),
            ema10=np.zeros(shape),
            ema20=np.zeros(shape),
            momentum5=np.ones(shape),
            ret1=np.zeros(shape),
        )
        owner.risk.cap = 1.
        return owner

    def test_registered_pair_and_disabled_control_is_exact_parent(self):
        module = self.module()
        self.assertEqual(
            [asdict(parameters) for parameters in module.grid()],
            [{"enabled": False}, {"enabled": True}],
        )
        for bad in (None, 0, 1, "yes"):
            with self.assertRaises(ValueError):
                module.Parameters(bad)
        market = sample_market(2, 145)
        parent = run(
            market,
            policy_factory=lambda current, config: ParentOwner(
                current, ParentParameters(2)
            ),
        )
        control = run(
            market,
            policy_factory=lambda current, config: module.Owner(
                current, module.Parameters(False)
            ),
        )
        pd.testing.assert_frame_equal(parent.equity, control.equity)
        pd.testing.assert_frame_equal(parent.targets, control.targets)
        self.assertEqual(parent.orders, control.orders)

    def test_two_consecutive_owned_basis_and_ema_failures_latch_full_exit(self):
        owner = self.owner()
        units = np.array([10_000.])
        owner.decide(observation(owner, 100, units))
        basis = owner.acquisition_basis[0]
        owner.price_signals.price[101:103, 0] = basis * .90
        owner.ema40[101:103, 0] = basis * .95

        first = owner.decide(observation(owner, 101, units))
        second = owner.decide(observation(owner, 102, units))

        self.assertEqual(first.unit_targets[0], units[0])
        self.assertEqual(second.unit_targets[0], 0.)
        self.assertTrue(owner.exit_pending[0])
        self.assertIn("BASIS_FAILURE_EXIT", second.reason)
        self.assertEqual(len(owner.events), 1)
        self.assertEqual(owner.events[0]["symbol"], owner.market.symbols[0])
        self.assertEqual(owner.events[0]["session"], 102)
        self.assertAlmostEqual(owner.events[0]["acquisition_basis"], basis)

    def test_nonconsecutive_failure_resets_without_selling(self):
        owner = self.owner()
        units = np.array([10_000.])
        owner.decide(observation(owner, 100, units))
        basis = owner.acquisition_basis[0]
        owner.price_signals.price[100:104, 0] = [basis * .9, basis * 1.01, basis * .9, basis * .9]
        owner.ema40[100:104, 0] = basis * .95

        decisions = [owner.decide(observation(owner, i, units)) for i in range(101, 104)]

        self.assertEqual(decisions[0].unit_targets[0], units[0])
        self.assertEqual(decisions[1].unit_targets[0], units[0])
        self.assertEqual(decisions[2].unit_targets[0], 0.)

    def test_actual_addition_updates_weighted_basis_and_full_sale_clears_state(self):
        owner = self.owner()
        first = np.array([10_000.])
        owner.decide(observation(owner, 100, first))
        first_price = owner.open[100, 0]
        second = np.array([15_000.])
        owner.decide(observation(owner, 101, second))
        expected = (10_000. * first_price + 5_000. * owner.open[101, 0]) / 15_000.
        self.assertAlmostEqual(owner.acquisition_basis[0], expected)

        owner.decide(observation(owner, 102, np.zeros(1)))

        self.assertEqual(owner.acquisition_basis[0], 0.)
        self.assertEqual(owner.failure_streak[0], 0)


if __name__ == "__main__":
    unittest.main()

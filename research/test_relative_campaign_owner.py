"""Contracts for the preregistered benchmark-relative campaign owner."""
from __future__ import annotations

from dataclasses import asdict
import importlib
import importlib.util
import unittest

import numpy as np
import pandas as pd

from test_core import sample_market
from techquant.config import Config
from techquant.engine import run
from techquant.policy import CloseObservation
from research.trend_book import Owner as ParentOwner, Parameters as ParentParameters


def benchmark(market, value=100.0):
    values = np.full(len(market.calendar), value, dtype=float)
    return values.copy(), values.copy(), "fixed-test-benchmark"


def observation(owner, session, units, cash=1_000_000.0):
    units = np.asarray(units, dtype=float)
    marks = np.nan_to_num(owner.price_signals.price[session], nan=0.0)
    values = units * marks
    nav = float(cash + values.sum())
    return CloseObservation.from_inventory(
        session,
        str(owner.market.calendar[session].date()),
        nav,
        cash,
        units,
        values / nav,
    )


class RelativeCampaignOwnerTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(
            importlib.util.find_spec("research.relative_campaign_owner"),
            "preregistered relative campaign owner is absent",
        )
        return importlib.import_module("research.relative_campaign_owner")

    def owner(self, enabled=True, symbols=3, sessions=120):
        module = self.module()
        market = sample_market(symbols, sessions)
        op, close, identity = benchmark(market)
        return module.Owner(
            market,
            module.Parameters(enabled),
            benchmark_open=op,
            benchmark_close=close,
            benchmark_identity=identity,
        )

    def mature(self, owner, symbol_index=0, start=40):
        units = np.zeros(len(owner.market.symbols))
        units[symbol_index] = 10_000.0
        entry = float(owner.open[start, symbol_index])
        owner.price_signals.price[start + 4, symbol_index] = entry * 1.10
        owner.price_signals.price[start + 9, symbol_index] = entry * 1.20
        for session in range(start, start + 10):
            if session not in (start + 4, start + 9):
                owner.price_signals.price[session, symbol_index] = entry * 1.05
            owner._observe_inventory(observation(owner, session, units))
        return units, start + 9

    def test_registered_pair_and_disabled_control_is_exact_parent(self):
        module = self.module()
        self.assertEqual(
            [asdict(parameters) for parameters in module.grid()],
            [{"enabled": False}, {"enabled": True}],
        )
        for invalid in (None, 0, 1, "yes"):
            with self.assertRaises(ValueError):
                module.Parameters(invalid)
        market = sample_market(4, 150)
        op, close, identity = benchmark(market)
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
                current,
                module.Parameters(False),
                benchmark_open=op,
                benchmark_close=close,
                benchmark_identity=identity,
            ),
        )
        pd.testing.assert_frame_equal(parent.equity, control.equity)
        pd.testing.assert_frame_equal(parent.targets, control.targets)
        self.assertEqual(parent.orders, control.orders)

    def test_actual_inventory_starts_and_resets_the_campaign_clock(self):
        owner = self.owner()
        units = np.array([10_000.0, 0.0, 0.0])
        owner._observe_inventory(observation(owner, 40, units))
        self.assertEqual(owner.campaign_age[0], 0)
        self.assertEqual(owner.campaign_entry_session[0], 40)
        self.assertEqual(owner.campaign_entry_open[0], owner.open[40, 0])
        owner._observe_inventory(observation(owner, 41, units))
        self.assertEqual(owner.campaign_age[0], 1)
        owner._observe_inventory(observation(owner, 42, np.zeros(3)))
        self.assertEqual(owner.campaign_age[0], -1)
        self.assertFalse(owner.fast_relative_positive[0])
        owner._observe_inventory(observation(owner, 43, units))
        self.assertEqual(owner.campaign_age[0], 0)
        self.assertEqual(owner.campaign_entry_session[0], 43)

    def test_quality_requires_fixed_fast_and_full_relative_strength(self):
        owner = self.owner()
        units, session = self.mature(owner)
        mature, quality = owner._campaign_quality(session)
        self.assertTrue(mature[0])
        self.assertTrue(quality[0])
        self.assertTrue(owner.fast_relative_positive[0])

        owner.price_signals.price[session, 0] = owner.campaign_entry_open[0] * 0.99
        _, quality = owner._campaign_quality(session)
        self.assertFalse(quality[0], "absolute campaign loss must revoke quality")

    def test_mature_quality_owns_slow_exit_but_not_hard_loss(self):
        owner = self.owner()
        units, session = self.mature(owner)
        owner.s.exit[session, 0] = True
        owner.price_signals.ret1[session, 0] = 0.01
        observed = owner._signal_inputs(session)
        self.assertFalse(observed.broken[0])

        owner.price_signals.ret1[session, 0] = -0.08
        observed = owner._signal_inputs(session)
        self.assertTrue(observed.broken[0])
        self.assertGreater(units[0], 0)

    def test_mature_nonquality_campaign_exits_without_changing_entry_ranking(self):
        owner = self.owner()
        _, session = self.mature(owner)
        parent = ParentOwner._signal_inputs(owner, session)
        owner.benchmark_close[session] = owner.campaign_benchmark_open[0] * 2.0
        observed = owner._signal_inputs(session)
        self.assertTrue(observed.broken[0])
        np.testing.assert_array_equal(observed.allowed, parent.allowed)
        np.testing.assert_array_equal(observed.score, parent.score)
        self.assertEqual(observed.capacity, parent.capacity)

    def test_benchmark_identity_and_alignment_are_required(self):
        module = self.module()
        market = sample_market(2, 80)
        op, close, identity = benchmark(market)
        with self.assertRaises(ValueError):
            module.Owner(
                market,
                module.Parameters(True),
                benchmark_open=op[:-1],
                benchmark_close=close,
                benchmark_identity=identity,
            )
        with self.assertRaises(ValueError):
            module.Owner(
                market,
                module.Parameters(True),
                benchmark_open=op,
                benchmark_close=close,
                benchmark_identity="",
            )


if __name__ == "__main__":
    unittest.main()

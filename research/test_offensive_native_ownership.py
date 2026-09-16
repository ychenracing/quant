"""Contracts for native ownership under permanently offensive account exposure."""
from dataclasses import asdict
import importlib
import importlib.util
import unittest
import pandas as pd
from test_core import sample_market
from techquant.engine import run
from research.trend_book import Owner as ParentOwner, Parameters as ParentParameters


class FullCap:
    def __init__(self):
        self.cap = 1.0

    def update(self, *args):
        return 1.0, 'OFFENSIVE_FULL_CAP'


def controlled_parent(market):
    owner = ParentOwner(market, ParentParameters(2))
    owner.risk = FullCap()
    return owner


class OffensiveNativeOwnershipTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(
            importlib.util.find_spec('research.offensive_native_ownership'),
            'offensive native ownership implementation is absent',
        )
        return importlib.import_module('research.offensive_native_ownership')

    def test_control_is_exact_parent(self):
        m = self.module()
        self.assertEqual([asdict(p) for p in m.grid()], [{'enabled': False}, {'enabled': True}])
        market = sample_market(6, 145)
        parent = run(market, policy_factory=lambda current, cfg: ParentOwner(current, ParentParameters(2)))
        control = run(market, policy_factory=lambda current, cfg: m.Owner(current, m.Parameters(False)))
        pd.testing.assert_frame_equal(parent.equity, control.equity)
        pd.testing.assert_frame_equal(parent.targets, control.targets)
        self.assertEqual(parent.orders, control.orders)

    def test_treatment_is_exact_native_owner_with_only_full_cap_risk_authority(self):
        m = self.module()
        market = sample_market(6, 145)
        expected = run(market, policy_factory=lambda current, cfg: controlled_parent(current))
        treatment = run(market, policy_factory=lambda current, cfg: m.Owner(current, m.Parameters(True)))
        pd.testing.assert_frame_equal(expected.equity, treatment.equity)
        pd.testing.assert_frame_equal(expected.targets, treatment.targets)
        self.assertEqual(expected.orders, treatment.orders)
        self.assertTrue((treatment.equity.exposure <= 1.0 + 1e-9).all())

    def test_treatment_never_reports_account_recovery_or_risk_cut(self):
        m = self.module()
        owner = m.Owner(sample_market(6, 100), m.Parameters(True))
        self.assertEqual(owner.parent.risk.cap, 1.0)
        cap, reason = owner.parent.risk.update(50, owner.parent.features, [2_000_000.0], owner.parent.config)
        self.assertEqual(cap, 1.0)
        self.assertEqual(reason, 'OFFENSIVE_FULL_CAP')
        self.assertEqual(owner.parent.risk.cap, 1.0)

    def test_registered_with_alpha_discovery_selector(self):
        self.module()
        from research.finite_study import Study
        study = Study('offensive_native_ownership')
        self.assertEqual(study.family, 'offensive_native_ownership')
        self.assertEqual(len(study.module.grid()), 2)
        self.assertIn('offensive_native_ownership.py', study.identity()['dependencies'])


if __name__ == '__main__':
    unittest.main()

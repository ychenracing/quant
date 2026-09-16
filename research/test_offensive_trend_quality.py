"""Contracts for deterministic causal trend-quality discovery."""
from dataclasses import asdict
import importlib
import importlib.util
import unittest
import numpy as np
import pandas as pd

from test_core import sample_market
from techquant.engine import run
from research.trend_book import Owner as ParentOwner, Parameters as ParentParameters


class OffensiveTrendQualityTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(
            importlib.util.find_spec('research.offensive_trend_quality'),
            'offensive trend quality implementation is absent',
        )
        return importlib.import_module('research.offensive_trend_quality')

    def test_control_is_exact_parent(self):
        m = self.module()
        self.assertEqual([asdict(p) for p in m.grid()], [{'enabled': False}, {'enabled': True}])
        market = sample_market(6, 145)
        parent = run(market, policy_factory=lambda current, cfg: ParentOwner(current, ParentParameters(2)))
        control = run(market, policy_factory=lambda current, cfg: m.Owner(current, m.Parameters(False)))
        pd.testing.assert_frame_equal(parent.equity, control.equity)
        pd.testing.assert_frame_equal(parent.targets, control.targets)
        self.assertEqual(parent.orders, control.orders)

    def test_quality_is_prefix_causal(self):
        m = self.module()
        market = sample_market(7, 145)
        full = m.trend_quality(market)
        prefix = m.trend_quality(market.prefix(str(market.calendar[109].date())))
        np.testing.assert_allclose(full[:110], prefix, rtol=0, atol=0, equal_nan=True)

    def test_treatment_replaces_only_alpha_score_inside_campaign_owner(self):
        m = self.module()
        owner = m.Owner(sample_market(6, 145), m.Parameters(True))
        np.testing.assert_allclose(owner.base.features.score, owner.quality, rtol=0, atol=0, equal_nan=True)
        self.assertEqual(owner.base.config.max_positions, 2)

    def test_registered_with_alpha_discovery_selector(self):
        self.module()
        from research.finite_study import Study
        study = Study('offensive_trend_quality')
        self.assertEqual(study.family, 'offensive_trend_quality')
        self.assertEqual(len(study.module.grid()), 2)
        self.assertIn('offensive_trend_quality.py', study.identity()['dependencies'])


if __name__ == '__main__':
    unittest.main()

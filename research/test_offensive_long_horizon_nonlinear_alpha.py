import importlib
import importlib.util
import json
import unittest

import numpy as np
import pandas as pd

from test_core import sample_market
from techquant.config import Config
from techquant.engine import run
from techquant.features import build_features
from research.offensive_campaign_peak_authority import Owner as ChampionOwner, Parameters as ChampionParameters


def audit_json(value):
    return json.dumps(value, sort_keys=True, allow_nan=True, separators=(',', ':'))


class LongHorizonNonlinearAlphaTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(
            importlib.util.find_spec('research.offensive_long_horizon_nonlinear_alpha'),
            'long horizon nonlinear alpha implementation is absent',
        )
        return importlib.import_module('research.offensive_long_horizon_nonlinear_alpha')

    def market(self):
        return sample_market(6, 260)

    def test_disabled_is_exact_campaign_peak_champion(self):
        module = self.module()
        market = self.market()
        champion = run(market, policy_factory=lambda current, cfg: ChampionOwner(current, ChampionParameters(True)))
        control = run(market, policy_factory=lambda current, cfg: module.Owner(current, module.Parameters(False)))
        pd.testing.assert_frame_equal(champion.equity, control.equity)
        pd.testing.assert_frame_equal(champion.targets, control.targets)
        self.assertEqual(champion.orders, control.orders)

    def test_walk_forward_scores_are_deterministic(self):
        module = self.module()
        market = self.market()
        first, first_audit = module.walk_forward_scores(market)
        second, second_audit = module.walk_forward_scores(market)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(audit_json(first_audit), audit_json(second_audit))
        self.assertGreater(first_audit['fit_count'], 0)
        self.assertGreater(first_audit['training_samples'], 0)

    def test_prefix_scores_are_identical_to_full_snapshot_history(self):
        module = self.module()
        market = self.market()
        cut_index = 220
        cut = market.calendar[cut_index]
        full_scores, full_audit = module.walk_forward_scores(market)
        prefix_scores, prefix_audit = module.walk_forward_scores(market.prefix(cut))
        np.testing.assert_array_equal(full_scores[: cut_index + 1], prefix_scores)
        expected = [r for r in full_audit['refits'] if r['refit_session'] <= cut_index]
        self.assertEqual(audit_json(expected), audit_json(prefix_audit['refits']))

    def test_every_training_label_is_fully_matured_before_refit(self):
        module = self.module()
        _, audit = module.walk_forward_scores(self.market())
        self.assertTrue(audit['refits'])
        for row in audit['refits']:
            self.assertLessEqual(row['max_label_end'], row['refit_session'])
            self.assertEqual(row['refit_session'] % Config().rebalance, 0)
            self.assertEqual(row['horizon'], 60)

    def test_before_first_fit_falls_back_to_existing_causal_score(self):
        module = self.module()
        market = self.market()
        learned, audit = module.walk_forward_scores(market)
        base = build_features(market, Config()).score
        first_fit = min(row['refit_session'] for row in audit['refits'])
        np.testing.assert_array_equal(learned[:first_fit], base[:first_fit])

    def test_nonready_predictions_remain_negative_infinity(self):
        module = self.module()
        market = self.market()
        learned, _ = module.walk_forward_scores(market)
        base = build_features(market, Config())
        self.assertTrue(np.isneginf(learned[~base.ready]).all())

    def test_enabled_owner_replaces_only_alpha_score_inputs(self):
        module = self.module()
        market = self.market()
        owner = module.Owner(market, module.Parameters(True))
        champion = ChampionOwner(market, ChampionParameters(True))
        learned, audit = module.walk_forward_scores(market)
        np.testing.assert_array_equal(owner.learned_score, learned)
        self.assertEqual(audit_json(owner.audit), audit_json(audit))
        np.testing.assert_array_equal(owner.parent.parent.base.features.score, learned)
        np.testing.assert_array_equal(owner.parent.parent.base.trend.entry, champion.parent.base.trend.entry)
        np.testing.assert_array_equal(owner.parent.parent.base.trend.exit, champion.parent.base.trend.exit)
        self.assertEqual(owner.parent.parent.base.config.max_positions, champion.parent.base.config.max_positions)


if __name__ == '__main__':
    unittest.main()

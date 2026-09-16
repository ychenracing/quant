import json
from pathlib import Path
import unittest

import pandas as pd

from test_core import sample_market
from techquant.engine import run
from research.trend_book import Owner as TrendOwner, Parameters as TrendParameters
from research.offensive_campaign_peak_authority import Owner, Parameters
from research import campaign_peak_formal_validation as campaign
from research import fresh_challenger_formal_validation as prior


class CampaignPeakFormalValidationTests(unittest.TestCase):
    def test_case_generator_and_aggregator_are_exact_reuse(self):
        self.assertIs(campaign.build_case_plan, prior.build_case_plan)
        self.assertIs(campaign.shard_plan, prior.shard_plan)
        self.assertIs(campaign.aggregate_existing, prior.aggregate)

    def test_control_policy_is_exact_trend_book(self):
        market = sample_market(6, 145)
        expected = run(market, policy_factory=lambda current, cfg: TrendOwner(current, TrendParameters(2)))
        actual = campaign.policy_result(market, 'control', 1.0, 1)
        pd.testing.assert_frame_equal(expected.equity, actual.equity)
        pd.testing.assert_frame_equal(expected.targets, actual.targets)
        self.assertEqual(expected.orders, actual.orders)

    def test_treatment_is_campaign_peak_candidate(self):
        market = sample_market(6, 145)
        expected = run(market, policy_factory=lambda current, cfg: Owner(current, Parameters(True)))
        actual = campaign.policy_result(market, 'treatment', 1.0, 1)
        pd.testing.assert_frame_equal(expected.equity, actual.equity)
        pd.testing.assert_frame_equal(expected.targets, actual.targets)
        self.assertEqual(expected.orders, actual.orders)

    def test_contract_is_bound_to_promoted_canonical_candidate(self):
        contract = json.loads((Path(__file__).parent / 'campaign_peak_formal_validation_contract.json').read_text())
        self.assertEqual(contract['candidate']['source_commit'], 'e1b0d8c0473c5f3c365d29de9a95fe8280d96765')
        self.assertEqual(contract['candidate']['canonical_run'], 35121169501)
        self.assertEqual(contract['candidate']['source_git_blob'], '13cb2c47b278a90d034b237e30cce8f740c64213')
        self.assertFalse(contract['candidate']['reselection'])


if __name__ == '__main__':
    unittest.main()

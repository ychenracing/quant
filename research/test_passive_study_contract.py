"""Regression tests for the registered comparator and evaluation windows."""
import json
from pathlib import Path
import unittest
from research import causal_passive_ownership_study as study
from research.offensive_campaign_peak_authority import Owner, Parameters
class PassiveStudyContractTests(unittest.TestCase):
    def test_control_is_the_registered_campaign_peak_candidate(self):
        self.assertIs(study.ActiveOwner, Owner)
        self.assertIs(study.ActiveParameters, Parameters)
    def test_windows_are_the_frozen_formal_windows_including_recovery(self):
        contract = json.loads(Path(study.__file__).with_name('campaign_peak_formal_validation_contract.json').read_text())
        self.assertEqual(study.evaluation_windows(), contract['time_windows'])
        self.assertEqual(study.evaluation_windows()['recovery'], ['2026-09-01', '2026-09-11'])

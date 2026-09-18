"""Frozen dynamic-relative case registry contracts."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from research.materialize_relative_cases import (
    CASE_MAP_SHA256,
    NATIVE_VERIFICATION_SHA256,
    WINDOW_METRICS_SHA256,
    WINDOWS,
    MaterializationError,
    group_aliases,
    load_native_attempts,
    load_window_metrics,
    materialize,
    require_sha256,
)

EVIDENCE = Path(os.environ.get(
    "QUANT_RELATIVE_EVIDENCE",
    "/mnt/data/passive-final-audit-outer/passive-closure",
))
CASE_MAP = Path(os.environ.get(
    "QUANT_CASE_MAP",
    str(EVIDENCE / "proof/case-map.json"),
))
WINDOW_METRICS = Path(os.environ.get(
    "QUANT_WINDOW_METRICS",
    str(EVIDENCE / "proof/passive-window-metrics.csv"),
))
NATIVE = Path(os.environ.get(
    "QUANT_NATIVE_VERIFICATION",
    "/mnt/data/references-35178033028/published/"
    "bbc002449093e458868a8748687df023958ba283/local-native/verification.json",
))


@unittest.skipUnless(
    CASE_MAP.exists() and WINDOW_METRICS.exists() and NATIVE.exists(),
    "local frozen evidence unavailable",
)
class RealEvidenceMaterializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_sha256(CASE_MAP, CASE_MAP_SHA256)
        require_sha256(WINDOW_METRICS, WINDOW_METRICS_SHA256)
        require_sha256(NATIVE, NATIVE_VERIFICATION_SHA256)
        cls.cases = json.loads(CASE_MAP.read_text())
        cls.metrics = load_window_metrics(WINDOW_METRICS)
        cls.attempts, cls.native_payload = load_native_attempts(NATIVE)
        cls.result = materialize(
            cls.cases, cls.metrics, cls.attempts, cls.native_payload
        )

    def test_aliases_collapse_to_exact_unique_base_cases(self):
        grouped = group_aliases(self.cases)
        self.assertEqual(len(self.cases), 199)
        self.assertEqual(len(grouped), 179)
        self.assertEqual(sum(len(item["aliases"]) - 1 for item in grouped), 20)
        common = next(item for item in grouped if "chatgpt_5" in item["aliases"])
        self.assertEqual(
            common["aliases"],
            ["chatgpt_5", "workbuddy_b", "common_exhaustive_5_0"],
        )

    def test_contract_windows_are_frozen_before_candidate_measurement(self):
        self.assertEqual(
            [window["name"] for window in WINDOWS],
            [
                "selection",
                "full",
                "bull",
                "final_evaluation",
                "late_june_august",
                "july_august",
                "recovery",
            ],
        )
        selection = WINDOWS[0]
        final = WINDOWS[3]
        self.assertEqual(
            (selection["start"], selection["end"]),
            ("2023-01-03", "2025-12-31"),
        )
        self.assertEqual(
            (final["start"], final["end"]),
            ("2026-01-01", "2026-09-11"),
        )
        self.assertIn("NO_PARAMETER_SELECTION", final["role"])

    def test_matrix_has_one_unique_gate_per_identity(self):
        counts = self.result["case_count"]
        self.assertEqual(counts["mandatory_exact_cases"], 179 * 7)
        hashes = [case["case_identity_hash"] for case in self.result["cases"]]
        self.assertEqual(len(hashes), len(set(hashes)))
        self.assertEqual(counts["candidate_unmeasured"], len(hashes))
        self.assertEqual(self.result["economic_acceptance"], "NOT_MET")

    def test_only_authenticated_full_window_references_are_reused(self):
        complete = [
            case for case in self.result["cases"]
            if case["reference_status"] == "REFERENCE_COMPLETE"
        ]
        self.assertEqual(len(complete), 1)
        case = complete[0]
        self.assertEqual(case["base_case"]["canonical_name"], "chatgpt_5")
        self.assertEqual(case["window"]["name"], "full")
        self.assertEqual(
            set(case["reference_results"]),
            {"chatgpt", "trae", "dumate", "workbuddy"},
        )
        self.assertEqual(case["missing_references"], [])
        self.assertEqual(
            max(
                row["terminal_wealth"]
                for row in case["reference_results"].values()
            ),
            20.221072996623576,
        )
        self.assertEqual(
            max(
                row["max_drawdown"]
                for row in case["reference_results"].values()
            ),
            0.3310775282947074,
        )

    def test_missing_or_invalid_reference_attempts_never_become_results(self):
        union = next(
            case for case in self.result["cases"]
            if case["base_case"]["canonical_name"] == "union"
            and case["window"]["name"] == "full"
        )
        self.assertEqual(set(union["reference_results"]), {"trae", "dumate"})
        self.assertEqual(
            set(union["missing_references"]),
            {"chatgpt", "workbuddy"},
        )
        self.assertEqual(union["reference_status"], "REFERENCE_INCOMPLETE")
        self.assertEqual(union["case_acceptance"], "NOT_MET")

    def test_inherited_windows_bind_state_rule_instead_of_fake_empty_cash(self):
        full = next(
            case for case in self.result["cases"]
            if case["window"]["name"] == "full"
        )
        stress = next(
            case for case in self.result["cases"]
            if case["window"]["name"] == "july_august"
        )
        self.assertEqual(full["identity"]["initial_positions"], {})
        rule = stress["identity"]["initial_positions"]["$state_rule"]
        self.assertEqual(rule["mode"], "ENDOGENOUS_STRATEGY_ACCOUNT_STATE")
        self.assertTrue(rule["result_specific_snapshot_binding_required"])
        self.assertEqual(rule["state_as_of"], "2026-06-30")

    def test_historical_seed_covers_only_the_five_preexisting_windows(self):
        counts = self.result["case_count"]
        self.assertEqual(counts["historical_passive_seed_measured"], 179 * 5)
        self.assertEqual(counts["historical_passive_seed_pending"], 179 * 2)
        for case in self.result["cases"]:
            if case["window"]["name"] in {"selection", "final_evaluation"}:
                self.assertIsNone(case["historical_passive_seed"])
            else:
                self.assertIsNotNone(case["historical_passive_seed"])

    def test_wrong_source_hash_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            path.write_text("{}")
            with self.assertRaises(MaterializationError):
                require_sha256(path, CASE_MAP_SHA256)


if __name__ == "__main__":
    unittest.main()

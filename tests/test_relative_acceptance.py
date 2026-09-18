"""Return-first acceptance must be exact-case, four-reference, and fail closed."""
from __future__ import annotations

import math
import unittest

from research.relative_acceptance import (
    AcceptanceError,
    case_identity_hash,
    evaluate_case,
    evaluate_matrix,
)


def identity(layer: str = "Native", **changes):
    value = {
        "evidence_layer": layer,
        "data_sha256": "data-a",
        "universe_and_order": ["sz300308", "sz300502"],
        "start_date": "2023-01-03",
        "end_date": "2026-09-11",
        "account_start_semantics": "INDEPENDENT_EMPTY_ACCOUNT",
        "initial_cash": 2_000_000.0,
        "initial_positions": {},
        "seed": 17,
        "costs": {
            "commission_bps": 2.5,
            "stamp_schedule": "frozen",
        },
        "slippage": {"bps": 10.0},
        "delay": 1,
        "capacity": {"max_adv": 0.005},
        "board_lot": "A_SHARE_BOARD_LOT",
        "tradability_rules": "FROZEN_OPEN_APPROXIMATION",
        "corporate_action_semantics": "ADJUSTED_ECONOMIC_UNITS",
        "runner": "techquant.engine.run",
        "reference_source_sha": "5575d1b1",
        "runtime_identity": {
            "python": "3.13",
            "numpy": "2.3.5",
            "pandas": "2.2.3",
        },
    }
    value.update(changes)
    return value


def metrics(case, wealth, drawdown, *, layer=None):
    return {
        "case_identity_hash": case_identity_hash(case),
        "evidence_layer": layer or case["evidence_layer"],
        "terminal_wealth": wealth,
        "max_drawdown": drawdown,
    }


def record(
    *,
    quant=(4.0, 0.30),
    chatgpt=(1.0, 0.20),
    trae=(3.0, 0.35),
    dumate=(2.0, 0.25),
    workbuddy=(4.0, 0.28),
    layer="Native",
    mandatory=True,
):
    case = identity(layer)
    return {
        "case_id": "case-a",
        "mandatory": mandatory,
        "identity": case,
        "case_identity_hash": case_identity_hash(case),
        "quant": metrics(case, *quant),
        "references": {
            "chatgpt": metrics(case, *chatgpt),
            "trae": metrics(case, *trae),
            "dumate": metrics(case, *dumate),
            "workbuddy": metrics(case, *workbuddy),
        },
    }


class RelativeAcceptanceTests(unittest.TestCase):
    def test_thresholds_are_derived_from_the_same_case(self):
        row = evaluate_case(record())
        self.assertEqual(row["return_floor"], 4.0)
        self.assertEqual(row["return_floor_reference"], ["workbuddy"])
        self.assertEqual(row["drawdown_target"], 0.35)
        self.assertEqual(row["drawdown_target_reference"], ["trae"])
        self.assertEqual(row["return_ratio"], 1.0)
        self.assertAlmostEqual(row["drawdown_margin"], 0.05)
        self.assertTrue(row["return_pass"])
        self.assertTrue(row["drawdown_target_met"])
        self.assertTrue(row["case_pass"])
        self.assertEqual(row["status"], "PASS")

    def test_return_is_hard_gate_and_drawdown_is_secondary_target(self):
        tied_return = evaluate_case(record(quant=(4.0, 0.349999999)))
        self.assertTrue(tied_return["return_pass"])
        self.assertTrue(tied_return["drawdown_target_met"])
        self.assertTrue(tied_return["case_pass"])

        tied_drawdown = evaluate_case(record(quant=(5.0, 0.35)))
        self.assertTrue(tied_drawdown["return_pass"])
        self.assertFalse(tied_drawdown["drawdown_target_met"])
        self.assertTrue(tied_drawdown["case_pass"])
        self.assertEqual(tied_drawdown["status"], "PASS")

        worse_drawdown = evaluate_case(record(quant=(5.0, 0.50)))
        self.assertTrue(worse_drawdown["return_pass"])
        self.assertFalse(worse_drawdown["drawdown_target_met"])
        self.assertTrue(worse_drawdown["case_pass"])

        failed_return = evaluate_case(record(quant=(3.99, 0.10)))
        self.assertFalse(failed_return["return_pass"])
        self.assertTrue(failed_return["drawdown_target_met"])
        self.assertFalse(failed_return["case_pass"])
        self.assertEqual(failed_return["status"], "FAIL_RETURN")

    def test_reference_incomplete_is_skipped_not_a_pass(self):
        value = record()
        del value["references"]["dumate"]
        row = evaluate_case(value)
        self.assertEqual(row["status"], "REFERENCE_INCOMPLETE")
        self.assertFalse(row["case_pass"])
        self.assertEqual(row["missing_references"], ["dumate"])
        result = evaluate_matrix([value])
        self.assertEqual(result["economic_acceptance"], "MET")
        self.assertEqual(
            result["reference_incomplete_cases"],
            ["case-a"],
        )
        self.assertEqual(
            result["skipped_incomplete_cases"],
            ["case-a"],
        )
        self.assertEqual(result["failed_return_cases"], [])

    def test_incomplete_does_not_hide_a_complete_return_failure(self):
        incomplete = record()
        incomplete["case_id"] = "missing-ref"
        del incomplete["references"]["dumate"]
        failing = record(quant=(3.99, 0.10))
        failing["case_id"] = "complete-miss"
        result = evaluate_matrix([incomplete, failing])
        self.assertEqual(result["economic_acceptance"], "NOT_MET")
        self.assertEqual(result["failed_return_cases"], ["complete-miss"])
        self.assertEqual(result["skipped_incomplete_cases"], ["missing-ref"])

    def test_layers_and_case_identities_cannot_be_mixed(self):
        value = record()
        value["references"]["trae"]["evidence_layer"] = "Corrected"
        with self.assertRaisesRegex(AcceptanceError, "evidence layer"):
            evaluate_case(value)

        changed = identity(delay=2)
        value = record()
        value["references"]["trae"]["case_identity_hash"] = (
            case_identity_hash(changed)
        )
        with self.assertRaisesRegex(AcceptanceError, "case identity"):
            evaluate_case(value)

    def test_every_identity_field_changes_the_hash(self):
        base = identity()
        baseline = case_identity_hash(base)
        for field in base:
            changed = dict(base)
            if field == "evidence_layer":
                changed[field] = "Corrected"
            elif field == "universe_and_order":
                changed[field] = list(reversed(base[field]))
            elif field == "initial_positions":
                changed[field] = {"sz300308": 1.0}
            elif isinstance(base[field], dict):
                changed[field] = {**base[field], "probe": 1}
            elif isinstance(base[field], (int, float)):
                changed[field] = base[field] + 1
            else:
                changed[field] = str(base[field]) + "-changed"
            with self.subTest(field=field):
                self.assertNotEqual(
                    case_identity_hash(changed),
                    baseline,
                )

    def test_matrix_uses_worst_return_case_not_average(self):
        passing = record(quant=(4.0, 0.50))
        passing["case_id"] = "strong"
        failing = record(quant=(3.99, 0.10))
        failing["case_id"] = "weak"
        result = evaluate_matrix([passing, failing])
        self.assertEqual(result["economic_acceptance"], "NOT_MET")
        self.assertEqual(result["failed_return_cases"], ["weak"])
        self.assertEqual(result["failed_cases"], ["weak"])
        self.assertEqual(result["weakest_return_case"], ["weak"])
        self.assertAlmostEqual(
            result["min_return_ratio"],
            3.99 / 4.0,
        )

    def test_drawdown_target_does_not_block_return_qualified_matrix(self):
        value = record(quant=(4.0, 0.50))
        result = evaluate_matrix([value])
        self.assertEqual(result["economic_acceptance"], "MET")
        self.assertEqual(result["failed_return_cases"], [])
        self.assertEqual(
            result["drawdown_target_not_met_cases"],
            ["case-a"],
        )
        self.assertFalse(
            result["all_mandatory_drawdown_targets_met"]
        )
        self.assertEqual(
            result["acceptance_priority"]["hard_gate"],
            "terminal_wealth_at_least_best_reference",
        )
        self.assertFalse(
            result["acceptance_priority"][
                "drawdown_target_is_merge_blocking"
            ]
        )

    def test_required_layer_must_have_a_mandatory_case(self):
        optional = record(layer="Corrected", mandatory=False)
        result = evaluate_matrix(
            [optional],
            required_layers=["Native"],
        )
        self.assertEqual(result["economic_acceptance"], "NOT_MET")
        self.assertEqual(
            result["required_layers_without_mandatory_cases"],
            ["Native"],
        )

    def test_duplicate_case_evidence_and_nonfinite_values_fail_closed(self):
        value = record()
        with self.assertRaisesRegex(
            AcceptanceError,
            "duplicate case evidence",
        ):
            evaluate_matrix([value, value])
        value = record()
        value["quant"]["terminal_wealth"] = math.inf
        with self.assertRaisesRegex(AcceptanceError, "finite"):
            evaluate_case(value)


if __name__ == "__main__":
    unittest.main()

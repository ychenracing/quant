"""Core candidate overlay must reuse exact matrix identity and fail closed."""
from __future__ import annotations

import unittest

from research.measure_relative_candidate import acceptance_record
from research.relative_acceptance import evaluate_case


def _case(*, complete: bool, wealth_refs: dict[str, tuple[float, float]] | None = None):
    identity = {
        "evidence_layer": "Native",
        "data_sha256": "data-a",
        "universe_and_order": ["sz300308", "sz300502"],
        "start_date": "2023-01-03",
        "end_date": "2026-09-11",
        "account_start_semantics": "INDEPENDENT_EMPTY_ACCOUNT_AT_WINDOW_START",
        "initial_cash": 2_000_000.0,
        "initial_positions": {},
        "seed": 34,
        "costs": {"commission_bps": 2.5},
        "slippage": {"bps": 10.0},
        "delay": 1,
        "capacity": {"max_adv": 0.005},
        "board_lot": "A_SHARE_BOARD_LOT",
        "tradability_rules": "FROZEN",
        "corporate_action_semantics": "ADJUSTED_UNITS",
        "runner": "techquant.engine.run",
        "reference_source_sha": "5575d1b1",
        "runtime_identity": {"python": "3.13.5"},
    }
    refs = {}
    if wealth_refs:
        for name, (wealth, drawdown) in wealth_refs.items():
            refs[name] = {
                "case_identity_hash": "pending",
                "evidence_layer": "Native",
                "terminal_wealth": wealth,
                "max_drawdown": drawdown,
            }
    return {
        "case_id": "chatgpt_5::full::Native",
        "mandatory": True,
        "identity": identity,
        "case_identity_hash": None,
        "evidence_layer": "Native",
        "reference_results": refs,
        "reference_status": "REFERENCE_COMPLETE" if complete else "REFERENCE_INCOMPLETE",
        "missing_references": [] if complete else ["chatgpt"],
    }


class MeasureRelativeCandidateTests(unittest.TestCase):
    def test_complete_case_passes_only_on_return_gate(self):
        from research.relative_acceptance import case_identity_hash

        refs = {
            "chatgpt": (1.778233112870138, 0.298051526553913),
            "trae": (10.674861915266794, 0.3310775282947074),
            "dumate": (1.465009570726167, 0.20686106331849607),
            "workbuddy": (20.221072996623576, 0.25331952059019447),
        }
        case = _case(complete=True, wealth_refs=refs)
        digest = case_identity_hash(case["identity"])
        case["case_identity_hash"] = digest
        for row in case["reference_results"].values():
            row["case_identity_hash"] = digest
        measured = {
            "status": "MEASURED",
            "wealth": 21.0,
            "max_drawdown": 0.50,
            "inherited_drawdown": 0.50,
            "sessions": 896,
            "orders": 20,
        }
        row = evaluate_case(acceptance_record(case, measured))
        self.assertTrue(row["return_pass"])
        self.assertTrue(row["case_pass"])
        self.assertFalse(row["drawdown_target_met"])
        self.assertEqual(row["return_floor"], 20.221072996623576)

    def test_incomplete_references_stay_incomplete_even_with_candidate(self):
        from research.relative_acceptance import case_identity_hash

        case = _case(complete=False, wealth_refs={"trae": (10.0, 0.3)})
        digest = case_identity_hash(case["identity"])
        case["case_identity_hash"] = digest
        case["reference_results"]["trae"]["case_identity_hash"] = digest
        measured = {
            "status": "MEASURED",
            "wealth": 30.0,
            "max_drawdown": 0.1,
            "inherited_drawdown": 0.1,
            "sessions": 896,
            "orders": 5,
        }
        row = evaluate_case(acceptance_record(case, measured))
        self.assertEqual(row["status"], "REFERENCE_INCOMPLETE")
        self.assertFalse(row["case_pass"])


if __name__ == "__main__":
    unittest.main()

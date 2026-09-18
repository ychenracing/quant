"""Measure the selected risk candidate on frozen core exact cases.

This is not economic acceptance.  It records the same global candidate on the
preregistered core universes, slices every frozen window from one full causal
path, and evaluates only cases that already have four Native references.
Production default stays passive.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from research.relative_acceptance import evaluate_matrix
from research.risk_ownership_screen import resolve_cases
from techquant.data import load_market
from techquant.evidence import metrics, source_identity
from techquant.risk_ownership import RiskOwnershipParameters, run_risk_aware_ownership

CORE_CANONICAL = (
    "chatgpt_5",
    "union",
    "remove_optical_leaders",
    "chatgpt_22",
    "trae_24",
    "dumate_18",
    "workbuddy_e",
)
FULL_START = "2023-01-03"
FULL_END = "2026-09-11"


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def quant_metrics(identity_hash: str, layer: str, measured: dict[str, Any]) -> dict[str, Any]:
    if measured.get("status") != "MEASURED":
        raise ValueError(f"candidate window is not measured: {measured}")
    return {
        "case_identity_hash": identity_hash,
        "evidence_layer": layer,
        "terminal_wealth": float(measured["wealth"]),
        "max_drawdown": float(measured["max_drawdown"]),
        "inherited_drawdown": float(measured["inherited_drawdown"]),
        "sessions": int(measured["sessions"]),
        "orders": int(measured["orders"]),
    }


def acceptance_record(case: dict[str, Any], measured: dict[str, Any]) -> dict[str, Any]:
    """Adapter from the frozen matrix row to the evaluator record shape."""

    return {
        "case_id": case["case_id"],
        "mandatory": bool(case["mandatory"]),
        "identity": case["identity"],
        "case_identity_hash": case["case_identity_hash"],
        "quant": quant_metrics(
            case["case_identity_hash"], case["evidence_layer"], measured
        ),
        "references": case["reference_results"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=Path("research/catalog.json"))
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("research/risk_ownership_screen_contract.json"),
    )
    parser.add_argument("--request", type=Path, default=Path("research/run-request.json"))
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("research/relative_reference_case_matrix.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    request = json.loads(args.request.read_text(encoding="utf-8"))
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    universes = resolve_cases(contract, catalog)
    market = load_market(
        args.data, supplement=args.supplement, sectors=catalog["sectors"]
    )
    if market.fingerprint() != request["data_sha256"]:
        raise ValueError("restored data fingerprint differs from the frozen request")

    selected = [
        case
        for case in matrix["cases"]
        if case["base_case"]["canonical_name"] in CORE_CANONICAL
    ]
    if len({case["base_case"]["canonical_name"] for case in selected}) != len(CORE_CANONICAL):
        raise ValueError("core exact cases missing from the frozen matrix")

    by_universe: dict[str, Any] = {}
    for name, symbols in universes.items():
        result = run_risk_aware_ownership(
            market.subset(symbols),
            parameters=RiskOwnershipParameters(core_fraction=0.9),
            start=FULL_START,
            end=FULL_END,
        )
        by_universe[name] = result

    overlay: list[dict[str, Any]] = []
    evaluable: list[dict[str, Any]] = []
    for case in selected:
        canonical = case["base_case"]["canonical_name"]
        universe_key = {
            "chatgpt_5": "common_five",
            "union": "union_34",
            "remove_optical_leaders": "remove_optical_leaders",
            "chatgpt_22": "chatgpt_original",
            "trae_24": "trae_original",
            "dumate_18": "dumate_original",
            "workbuddy_e": "workbuddy_original",
        }[canonical]
        window = case["window"]
        measured = metrics(by_universe[universe_key], window["start"], window["end"])
        record = {
            "case_id": case["case_id"],
            "case_identity_hash": case["case_identity_hash"],
            "canonical_name": canonical,
            "window": window["name"],
            "reference_status": case["reference_status"],
            "missing_references": list(case["missing_references"]),
            "candidate": quant_metrics(
                case["case_identity_hash"], case["evidence_layer"], measured
            ),
        }
        overlay.append(record)
        if case["reference_status"] == "REFERENCE_COMPLETE":
            evaluable.append(acceptance_record(case, measured))

    evaluation = (
        evaluate_matrix(evaluable)
        if evaluable
        else {
            "economic_acceptance": "NOT_MET",
            "reference_incomplete_cases": [row["case_id"] for row in overlay],
            "rows": [],
        }
    )
    # The frozen contract still has 1252 mandatory incomplete cases.
    # A complete subset cannot promote campaign acceptance.
    campaign_acceptance = "NOT_MET"
    summary = {
        "status": "CORE_CANDIDATE_MEASURED_NOT_ACCEPTED",
        "source": source_identity(),
        "source_commit": os.environ.get("QUANT_SOURCE_COMMIT", "UNBOUND_LOCAL_SNAPSHOT"),
        "data_sha256": market.fingerprint(),
        "core_universes": list(CORE_CANONICAL),
        "measured_cases": len(overlay),
        "evaluable_complete_reference_cases": len(evaluable),
        "complete_subset_return_gate": evaluation.get("economic_acceptance"),
        "economic_acceptance": campaign_acceptance,
        "production_default_changed": False,
        "limitations": [
            "Four-reference coverage remains incomplete outside the measured complete cases.",
            "Private ychenracing/trades checkout is required to fill remaining Native references.",
            "This measurement does not change the production default.",
        ],
    }
    write_json(args.output / "overlay.json", {"cases": overlay})
    write_json(args.output / "evaluation.json", evaluation)
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

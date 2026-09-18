"""Bounded selection-window screen for the risk-aware ownership structure.

This runner is not the dynamic four-reference acceptance gate.  It measures the
three preregistered core-retention structures against the same-engine passive
ownership baseline on frozen selection-period cases only.  It never reads the
2026 final-evaluation window and never changes the production default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from techquant.data import file_hash, load_market
from techquant.evidence import save_result, source_identity
from techquant.passive import run_passive_ownership
from techquant.risk_ownership import RiskOwnershipParameters, run_risk_aware_ownership


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def resolve_cases(contract: dict[str, Any], catalog: dict[str, Any]) -> dict[str, list[str]]:
    pools = catalog["pools"]
    result: dict[str, list[str]] = {}
    for record in contract["cases"]:
        case_id = record["case_id"]
        if case_id in result:
            raise ValueError(f"duplicate case_id: {case_id}")
        if "catalog_pool" in record:
            symbols = list(pools[record["catalog_pool"]])
        else:
            symbols = list(pools[record["derived_from"]])
            removed = set(record["remove"])
            if not removed <= set(symbols):
                raise ValueError(f"case {case_id} removes symbols outside its parent")
            symbols = [symbol for symbol in symbols if symbol not in removed]
        if not symbols or len(symbols) != len(set(symbols)):
            raise ValueError(f"case {case_id} must be a nonempty unique universe")
        result[case_id] = symbols
    return result


def result_row(
    *,
    case_id: str,
    core_fraction: float,
    candidate: dict[str, Any],
    passive: dict[str, Any],
) -> dict[str, Any]:
    if candidate["status"] != "MEASURED" or passive["status"] != "MEASURED":
        raise ValueError("screening requires complete measured metrics")
    return {
        "case_id": case_id,
        "core_fraction": core_fraction,
        "candidate_wealth": candidate["wealth"],
        "passive_wealth": passive["wealth"],
        "wealth_ratio_to_passive": candidate["wealth"] / passive["wealth"],
        "candidate_max_drawdown": candidate["max_drawdown"],
        "passive_max_drawdown": passive["max_drawdown"],
        "drawdown_margin_to_passive": (
            passive["max_drawdown"] - candidate["max_drawdown"]
        ),
        "candidate_orders": candidate["orders"],
        "passive_orders": passive["orders"],
        "candidate_turnover": candidate["gross_turnover"],
        "passive_turnover": passive["gross_turnover"],
        "candidate_average_exposure": candidate["average_exposure"],
        "passive_average_exposure": passive["average_exposure"],
        "candidate_blocked_attempts": candidate["blocked_attempts"],
        "passive_blocked_attempts": passive["blocked_attempts"],
    }


def aggregate(rows: list[dict[str, Any]], fractions: list[float]) -> list[dict[str, Any]]:
    summaries = []
    for fraction in fractions:
        subset = [row for row in rows if row["core_fraction"] == fraction]
        if not subset:
            raise ValueError(f"no rows for core_fraction={fraction}")
        wealth = np.array([row["wealth_ratio_to_passive"] for row in subset], dtype=float)
        drawdown = np.array([row["drawdown_margin_to_passive"] for row in subset], dtype=float)
        summaries.append(
            {
                "core_fraction": fraction,
                "case_count": len(subset),
                "minimum_wealth_ratio_to_passive": float(wealth.min()),
                "median_wealth_ratio_to_passive": float(np.median(wealth)),
                "minimum_drawdown_margin_to_passive": float(drawdown.min()),
                "median_drawdown_margin_to_passive": float(np.median(drawdown)),
                "drawdown_improved_cases": int((drawdown > 0).sum()),
                "wealth_not_below_passive_cases": int((wealth >= 1).sum()),
            }
        )
    return summaries


def choose_structure(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    # The ordering is frozen in the contract: preserve the worst return first,
    # then improve the weakest/typical drawdown, with larger core only as an
    # exact deterministic tie-break.  This is screening, not acceptance.
    return max(
        summaries,
        key=lambda row: (
            row["minimum_wealth_ratio_to_passive"],
            row["minimum_drawdown_margin_to_passive"],
            row["median_drawdown_margin_to_passive"],
            row["median_wealth_ratio_to_passive"],
            row["core_fraction"],
        ),
    )


def build_manifest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): file_hash(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "MANIFEST.json"
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument(
        "--catalog", type=Path, default=Path("research/catalog.json")
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("research/risk_ownership_screen_contract.json"),
    )
    parser.add_argument(
        "--request", type=Path, default=Path("research/run-request.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    runs = args.output / "runs"
    runs.mkdir()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    request = json.loads(args.request.read_text(encoding="utf-8"))
    if contract["selection_window"] != {
        "start": "2023-01-03",
        "end": "2025-12-31",
    }:
        raise ValueError("selection window differs from the frozen contract")
    if contract["final_evaluation_window"]["used_for_selection"]:
        raise ValueError("final evaluation data cannot be used during selection")

    fractions = contract["parameters"]["core_fraction"]
    if fractions != [0.9]:
        raise ValueError("core-retention structure differs from the frozen contract")

    market = load_market(
        args.data,
        supplement=args.supplement,
        sectors=catalog["sectors"],
    )
    if market.fingerprint() != request["data_sha256"]:
        raise ValueError("restored data fingerprint differs from the frozen request")
    cases = resolve_cases(contract, catalog)

    selection_start = contract["selection_window"]["start"]
    selection_end = contract["selection_window"]["end"]
    rows: list[dict[str, Any]] = []
    passive_by_case: dict[str, dict[str, Any]] = {}

    for case_id, symbols in cases.items():
        scoped = market.subset(symbols)
        case_root = runs / case_id
        case_root.mkdir()
        passive = run_passive_ownership(
            scoped, start=selection_start, end=selection_end
        )
        passive_path = save_result(passive, case_root / "passive")
        passive_metrics = json.loads(
            (passive_path / "metrics.json").read_text(encoding="utf-8")
        )
        passive_by_case[case_id] = passive_metrics

        for fraction in fractions:
            parameters = RiskOwnershipParameters(core_fraction=fraction)
            candidate = run_risk_aware_ownership(
                scoped,
                parameters=parameters,
                start=selection_start,
                end=selection_end,
            )
            candidate_path = save_result(
                candidate,
                case_root / f"core_{int(round(fraction * 100)):02d}",
            )
            candidate_metrics = json.loads(
                (candidate_path / "metrics.json").read_text(encoding="utf-8")
            )
            row = result_row(
                case_id=case_id,
                core_fraction=fraction,
                candidate=candidate_metrics,
                passive=passive_metrics,
            )
            reasons = candidate.equity["reason"].astype(str)
            row.update(
                {
                    "state_transitions": int(reasons.str.contains("STATE:", regex=False).sum()),
                    "systemic_protection_signals": int(
                        reasons.str.contains("SYSTEMIC_PROTECTION:", regex=False).sum()
                    ),
                    "acute_damage_signals": int(
                        reasons.str.contains("ACUTE_HOLDING_DAMAGE:", regex=False).sum()
                    ),
                    "funded_recovery_completions": int(
                        reasons.str.contains("FUNDED_RECOVERY_COMPLETE", regex=False).sum()
                    ),
                    "minimum_exposure": float(candidate.equity.exposure.min()),
                    "final_exposure": float(candidate.equity.exposure.iloc[-1]),
                }
            )
            rows.append(row)

    matrix = pd.DataFrame(rows).sort_values(
        ["core_fraction", "case_id"], kind="stable"
    )
    matrix.to_csv(args.output / "matrix.csv", index=False, float_format="%.17g")
    summaries = aggregate(rows, fractions)
    selected = choose_structure(summaries)

    source = source_identity()
    identity = {
        "status": "STRUCTURAL_SCREEN_MEASURED_NOT_ACCEPTED",
        "source": source,
        "source_commit": os.environ.get("QUANT_SOURCE_COMMIT", source["commit"]),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "data_sha256": market.fingerprint(),
        "catalog_sha256": file_hash(args.catalog),
        "contract_sha256": file_hash(args.contract),
        "request_sha256": file_hash(args.request),
        "contract_identity_sha256": canonical_hash(contract),
        "selection_window": contract["selection_window"],
        "final_evaluation_window": contract["final_evaluation_window"],
        "case_universes": cases,
        "execution": {
            "delay": 1,
            "initial_cash": 2_000_000.0,
            "commission_bps": 2.5,
            "slippage_bps": 10.0,
            "max_adv": 0.005,
        },
        "reference_gate": "NOT_EVALUATED_REFERENCE_RESULTS_INCOMPLETE",
        "production_default_changed": False,
    }
    selection = {
        "status": "STRUCTURE_SELECTED_FOR_FINAL_EVALUATION_NOT_ACCEPTED",
        "selected_parameters": {
            "core_fraction": selected["core_fraction"],
            "protection_trigger": "INDEPENDENTLY_CONFIRMED_CRISIS_ONLY",
            "open_execution": "DEFER_POSITIVE_GAP_PROTECTIVE_SELL",
            "partial_trim_floor": "ONE_PERCENT_OF_NAV",
            "recovery": "FULL_OWNERSHIP_ON_CONFIRMED_RISK_CLEAR",
        },
        "lexicographic_result": selected,
        "all_structures": summaries,
        "objective": contract["selection_objective"],
        "case_count": len(cases),
        "candidate_run_count": len(rows),
        "final_evaluation_used": False,
        "dynamic_reference_acceptance": "NOT_EVALUATED",
        "production_default_changed": False,
    }
    summary = {
        "status": "MEASURED_NOT_ACCEPTED",
        "selection": selection,
        "identity": identity,
        "passive_metrics": passive_by_case,
        "limitations": contract["limitations"],
    }
    write_json(args.output / "identity.json", identity)
    write_json(args.output / "selection.json", selection)
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "MANIFEST.json", build_manifest(args.output))
    print(json.dumps(selection, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

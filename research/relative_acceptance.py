"""Return-first acceptance against four case-matched references.

Every threshold is derived inside one exact case identity and evidence layer.
Mixed identities, rounded comparisons, or duplicate rows can never become a
pass.  A case missing any of the four references is skipped: it is not a pass
and it does not block campaign acceptance.

Terminal wealth is the only economic hard gate on complete cases.  Drawdown is
measured against the worst-drawdown reference in the same exact case and is
optimized only after the return gate is met.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REFERENCE_NAMES = ("chatgpt", "trae", "dumate", "workbuddy")
EVIDENCE_LAYERS = ("Native", "Corrected", "Normalized", "Equivalent-execution")
IDENTITY_FIELDS = (
    "evidence_layer",
    "data_sha256",
    "universe_and_order",
    "start_date",
    "end_date",
    "account_start_semantics",
    "initial_cash",
    "initial_positions",
    "seed",
    "costs",
    "slippage",
    "delay",
    "capacity",
    "board_lot",
    "tradability_rules",
    "corporate_action_semantics",
    "runner",
    "reference_source_sha",
    "runtime_identity",
)
TABLE_FIELDS = (
    "case_id",
    "evidence_layer",
    "case_identity_hash",
    "quant_terminal_wealth",
    "quant_max_drawdown",
    "chatgpt_terminal_wealth",
    "chatgpt_max_drawdown",
    "trae_terminal_wealth",
    "trae_max_drawdown",
    "dumate_terminal_wealth",
    "dumate_max_drawdown",
    "workbuddy_terminal_wealth",
    "workbuddy_max_drawdown",
    "return_floor",
    "return_floor_reference",
    "drawdown_target",
    "drawdown_target_reference",
    "return_ratio",
    "drawdown_margin",
    "return_pass",
    "drawdown_target_met",
    "case_pass",
    "status",
    "mandatory",
    "missing_references",
)


class AcceptanceError(ValueError):
    """The evidence cannot be compared without changing its meaning."""


def canonical_json(value: Any) -> str:
    """Serialize evidence identity deterministically and reject NaN/Infinity."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def case_identity_payload(identity: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in IDENTITY_FIELDS if field not in identity]
    extra = sorted(set(identity).difference(IDENTITY_FIELDS))
    if missing or extra:
        raise AcceptanceError(
            f"case identity fields differ; missing={missing}, extra={extra}"
        )
    payload = {field: identity[field] for field in IDENTITY_FIELDS}
    layer = payload["evidence_layer"]
    if layer not in EVIDENCE_LAYERS:
        raise AcceptanceError(f"unsupported evidence layer: {layer!r}")
    universe = payload["universe_and_order"]
    if (
        not isinstance(universe, list)
        or not universe
        or not all(isinstance(symbol, str) and symbol for symbol in universe)
        or len(set(universe)) != len(universe)
    ):
        raise AcceptanceError(
            "universe_and_order must be a non-empty unique string list"
        )
    if not isinstance(payload["initial_positions"], Mapping):
        raise AcceptanceError("initial_positions must be a mapping")
    initial_cash = _finite_number(payload["initial_cash"], "initial_cash")
    if initial_cash <= 0:
        raise AcceptanceError("initial_cash must be positive")
    payload["initial_cash"] = initial_cash
    canonical_json(payload)
    return payload


def case_identity_hash(identity: Mapping[str, Any]) -> str:
    payload = case_identity_payload(identity)
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AcceptanceError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AcceptanceError(f"{field} must be finite")
    return result


def _validate_metrics(
    metrics: Mapping[str, Any],
    *,
    owner: str,
    expected_hash: str,
    expected_layer: str,
) -> tuple[float, float]:
    if not isinstance(metrics, Mapping):
        raise AcceptanceError(f"{owner} metrics must be a mapping")
    if metrics.get("case_identity_hash") != expected_hash:
        raise AcceptanceError(
            f"{owner} case identity does not match the threshold case"
        )
    if metrics.get("evidence_layer") != expected_layer:
        raise AcceptanceError(
            f"{owner} evidence layer does not match the threshold case"
        )
    wealth = _finite_number(
        metrics.get("terminal_wealth"), f"{owner}.terminal_wealth"
    )
    drawdown = _finite_number(
        metrics.get("max_drawdown"), f"{owner}.max_drawdown"
    )
    if wealth <= 0:
        raise AcceptanceError(f"{owner}.terminal_wealth must be positive")
    if not 0 <= drawdown <= 1:
        raise AcceptanceError(
            f"{owner}.max_drawdown must be a loss magnitude in [0, 1]"
        )
    return wealth, drawdown


def _empty_row(
    *,
    case_id: str,
    layer: str,
    identity_hash: str,
    mandatory: bool,
    status: str,
    missing: Sequence[str] = (),
) -> dict[str, Any]:
    row = {field: None for field in TABLE_FIELDS}
    row.update(
        case_id=case_id,
        evidence_layer=layer,
        case_identity_hash=identity_hash,
        mandatory=mandatory,
        status=status,
        case_pass=False,
        return_pass=False,
        drawdown_target_met=False,
        missing_references=list(missing),
    )
    return row


def evaluate_case(
    record: Mapping[str, Any],
    *,
    reference_names: Sequence[str] = REFERENCE_NAMES,
) -> dict[str, Any]:
    """Evaluate one candidate against four references from its exact case.

    Return qualification is the hard gate.  The drawdown target is reported as
    a secondary optimization outcome and does not alter ``case_pass``.
    """
    if not isinstance(record, Mapping):
        raise AcceptanceError("case record must be a mapping")
    case_id = record.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise AcceptanceError("case_id must be a non-empty string")
    if type(record.get("mandatory", True)) is not bool:
        raise AcceptanceError("mandatory must be boolean")
    mandatory = record.get("mandatory", True)
    identity = record.get("identity")
    if not isinstance(identity, Mapping):
        raise AcceptanceError("identity must be a mapping")
    payload = case_identity_payload(identity)
    identity_hash = case_identity_hash(payload)
    declared_hash = record.get("case_identity_hash")
    if declared_hash is not None and declared_hash != identity_hash:
        raise AcceptanceError(
            "declared case_identity_hash does not match identity"
        )
    layer = payload["evidence_layer"]

    references = record.get("references")
    if not isinstance(references, Mapping):
        raise AcceptanceError("references must be a mapping")
    expected = tuple(reference_names)
    if tuple(sorted(references)) != tuple(sorted(set(references))):
        raise AcceptanceError("duplicate reference names are not allowed")
    extras = sorted(set(references).difference(expected))
    if extras:
        raise AcceptanceError(f"unknown references: {extras}")
    missing = [name for name in expected if name not in references]

    quant = _validate_metrics(
        record.get("quant"),
        owner="quant",
        expected_hash=identity_hash,
        expected_layer=layer,
    )
    if missing:
        row = _empty_row(
            case_id=case_id,
            layer=layer,
            identity_hash=identity_hash,
            mandatory=mandatory,
            status="REFERENCE_INCOMPLETE",
            missing=missing,
        )
        row["quant_terminal_wealth"], row["quant_max_drawdown"] = quant
        return row

    values: dict[str, tuple[float, float]] = {}
    for name in expected:
        values[name] = _validate_metrics(
            references[name],
            owner=name,
            expected_hash=identity_hash,
            expected_layer=layer,
        )

    return_floor = max(wealth for wealth, _ in values.values())
    drawdown_target = max(drawdown for _, drawdown in values.values())
    floor_sources = sorted(
        name for name, (wealth, _) in values.items() if wealth == return_floor
    )
    target_sources = sorted(
        name for name, (_, drawdown) in values.items()
        if drawdown == drawdown_target
    )
    quant_wealth, quant_drawdown = quant
    return_ratio = quant_wealth / return_floor
    drawdown_margin = drawdown_target - quant_drawdown
    return_pass = quant_wealth >= return_floor
    drawdown_target_met = quant_drawdown < drawdown_target
    case_pass = return_pass

    row = _empty_row(
        case_id=case_id,
        layer=layer,
        identity_hash=identity_hash,
        mandatory=mandatory,
        status="PASS" if case_pass else "FAIL_RETURN",
    )
    row.update(
        quant_terminal_wealth=quant_wealth,
        quant_max_drawdown=quant_drawdown,
        return_floor=return_floor,
        return_floor_reference=floor_sources,
        drawdown_target=drawdown_target,
        drawdown_target_reference=target_sources,
        return_ratio=return_ratio,
        drawdown_margin=drawdown_margin,
        return_pass=return_pass,
        drawdown_target_met=drawdown_target_met,
        case_pass=case_pass,
    )
    for name, (wealth, drawdown) in values.items():
        row[f"{name}_terminal_wealth"] = wealth
        row[f"{name}_max_drawdown"] = drawdown
    return row


def evaluate_matrix(
    records: Iterable[Mapping[str, Any]],
    *,
    required_layers: Sequence[str] = ("Native",),
    reference_names: Sequence[str] = REFERENCE_NAMES,
) -> dict[str, Any]:
    required = tuple(required_layers)
    if not required or any(
        layer not in EVIDENCE_LAYERS for layer in required
    ):
        raise AcceptanceError(
            "required_layers must contain supported evidence layers"
        )
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        row = evaluate_case(record, reference_names=reference_names)
        key = (
            row["case_id"],
            row["evidence_layer"],
            row["case_identity_hash"],
        )
        if key in seen:
            raise AcceptanceError(f"duplicate case evidence: {key}")
        seen.add(key)
        rows.append(row)
    rows.sort(key=lambda row: (row["evidence_layer"], row["case_id"]))

    required_missing = [
        layer
        for layer in required
        if not any(
            row["mandatory"] and row["evidence_layer"] == layer
            for row in rows
        )
    ]
    gating = [
        row
        for row in rows
        if row["mandatory"] and row["evidence_layer"] in required
    ]
    failed = [
        row["case_id"]
        for row in gating
        if row["status"] == "FAIL_RETURN"
    ]
    incomplete = [
        row["case_id"]
        for row in gating
        if row["status"] == "REFERENCE_INCOMPLETE"
    ]
    skipped_incomplete = list(incomplete)
    drawdown_target_not_met = [
        row["case_id"]
        for row in gating
        if row["return_ratio"] is not None and not row["drawdown_target_met"]
    ]
    complete = [row for row in gating if row["return_ratio"] is not None]
    weakest_return = (
        min(
            complete,
            key=lambda row: (row["return_ratio"], row["case_id"]),
        )
        if complete
        else None
    )
    weakest_drawdown = (
        min(
            complete,
            key=lambda row: (row["drawdown_margin"], row["case_id"]),
        )
        if complete
        else None
    )
    # Incomplete mandatory cases are skipped. Campaign MET when the required
    # layer is present and every complete mandatory case clears the return gate.
    met = not required_missing and not failed

    layer_summary: dict[str, dict[str, Any]] = {}
    for layer in EVIDENCE_LAYERS:
        layer_rows = [
            row for row in rows if row["evidence_layer"] == layer
        ]
        if not layer_rows:
            continue
        mandatory_rows = [row for row in layer_rows if row["mandatory"]]
        complete_rows = [
            row
            for row in mandatory_rows
            if row["return_ratio"] is not None
        ]
        layer_summary[layer] = {
            "cases": len(layer_rows),
            "mandatory_cases": len(mandatory_rows),
            "return_gate_passed": sum(
                bool(row["return_pass"]) for row in complete_rows
            ),
            "return_gate_failed": sum(
                not bool(row["return_pass"]) for row in complete_rows
            ),
            "drawdown_target_met": sum(
                bool(row["drawdown_target_met"]) for row in complete_rows
            ),
            "drawdown_target_not_met": sum(
                not bool(row["drawdown_target_met"])
                for row in complete_rows
            ),
            "reference_incomplete": sum(
                row["status"] == "REFERENCE_INCOMPLETE"
                for row in mandatory_rows
            ),
        }

    return {
        "schema_version": 3,
        "acceptance_priority": {
            "hard_gate": "terminal_wealth_at_least_best_reference",
            "secondary_objective": (
                "minimize_drawdown_after_return_gate"
            ),
            "ideal_drawdown_target": (
                "strictly_below_worst_reference_drawdown_in_same_case"
            ),
            "drawdown_target_is_merge_blocking": False,
        },
        "reference_names": list(reference_names),
        "required_evidence_layers": list(required),
        "economic_acceptance": "MET" if met else "NOT_MET",
        "mandatory_cases": len(gating),
        "required_layers_without_mandatory_cases": required_missing,
        "failed_return_cases": failed,
        "failed_cases": failed,
        "reference_incomplete_cases": incomplete,
        "skipped_incomplete_cases": skipped_incomplete,
        "drawdown_target_not_met_cases": drawdown_target_not_met,
        "drawdown_target_met_cases": [
            row["case_id"]
            for row in complete
            if row["drawdown_target_met"]
        ],
        "min_return_ratio": (
            weakest_return["return_ratio"]
            if weakest_return is not None
            else None
        ),
        "weakest_return_case": (
            weakest_return["case_id"]
            if weakest_return is not None
            else None
        ),
        "min_drawdown_margin": (
            weakest_drawdown["drawdown_margin"]
            if weakest_drawdown is not None
            else None
        ),
        "weakest_drawdown_case": (
            weakest_drawdown["case_id"]
            if weakest_drawdown is not None
            else None
        ),
        "all_mandatory_drawdown_targets_met": (
            bool(complete)
            and all(row["drawdown_target_met"] for row in complete)
        ),
        "layers": layer_summary,
        "rows": rows,
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=TABLE_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: _csv_value(row.get(field))
                    for field in TABLE_FIELDS
                }
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate return-first exact-case four-reference acceptance"
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument(
        "--required-layer",
        action="append",
        dest="required_layers",
        choices=EVIDENCE_LAYERS,
    )
    args = parser.parse_args(argv)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("cases"), list
    ):
        raise AcceptanceError(
            "input must be an object containing a cases list"
        )
    result = evaluate_matrix(
        payload["cases"],
        required_layers=args.required_layers
        or payload.get("required_evidence_layers", ["Native"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.csv, result["rows"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

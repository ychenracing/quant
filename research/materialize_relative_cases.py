"""Materialize the frozen exact-case registry for dynamic relative acceptance.

The historical plan contains aliases for identical economic cases. This tool
retains every historical name and group for traceability, but emits one hard
acceptance row per exact identity. It also adds the contract's explicit
selection and final-evaluation windows before any candidate result is measured.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.relative_acceptance import REFERENCE_NAMES, canonical_json, case_identity_hash

CASE_MAP_SHA256 = "ad464cf2212139c9b6c134dac71ebfb0147d9f7b4bc7fbcd5ffb373b653e5684"
WINDOW_METRICS_SHA256 = "fe6898043232d8c6befa15f1bbd04ca9ef996142dd3715990fb8feeacb276ed6"
NATIVE_VERIFICATION_SHA256 = "4228226d5d53b8d191634f51eb88db6dc923336755693bce79f139be9e088565"
MARKET_FINGERPRINT = "d9ded1d933f292933aa09145bc36a5696421c43dbf266fce018a34ef0b9f3f9b"
SOURCE_MAIN_SHA = "b0902e8abd360e2702ae350f4ccba3a696bcb71f"
REFERENCE_COMMIT = "5575d1b1b79fab405b92cfb7d164c7054a8bd826"
REFERENCE_TREES = {
    "chatgpt": "c06585d7837d32905ef4b954b8b2c22f4f3f51f9",
    "trae": "0ba9d9b776d1b01712e0086af12dfa81d9589b56",
    "dumate": "5b8f1863def6fd784284f9b7a782be28caa3f009",
    "workbuddy": "3c29e60758e0a6538a81ea8144a144d6faf966c4",
}
NATIVE_HARNESS_SHA256 = "32f310eac06c0c7c684c409d29028ea18312c4bd1400c132f2a50dd9b6ecb64e"
ORCHESTRATOR_SHA256 = "1605dc68292c90b6f5bd534728ffe1966dbf3516782d2051655570ccf05160c2"
NATIVE_ARCHIVE_SHA256 = "0a03c52f24dd08c2488b897b6e0a3f3d7211c467c49300645e1d722894ee620f"

# Order is frozen. Windows beginning at 2023-01-03 share the common empty
# CNY-2m account. Later windows are slices of the strategy's own causal full
# path, bound to the previous-session account state rather than falsely reset.
WINDOWS: tuple[dict[str, Any], ...] = (
    {
        "name": "selection",
        "start": "2023-01-03",
        "end": "2025-12-31",
        "first_session": "2023-01-03",
        "role": "STRUCTURE_SELECTION_ONLY",
        "account_start_semantics": "INDEPENDENT_EMPTY_ACCOUNT_AT_WINDOW_START",
        "historical_metric_name": None,
    },
    {
        "name": "full",
        "start": "2023-01-03",
        "end": "2026-09-11",
        "first_session": "2023-01-03",
        "role": "FINAL_AGGREGATE_EVALUATION",
        "account_start_semantics": "INDEPENDENT_EMPTY_ACCOUNT_AT_WINDOW_START",
        "historical_metric_name": "full",
    },
    {
        "name": "bull",
        "start": "2023-01-03",
        "end": "2026-06-30",
        "first_session": "2023-01-03",
        "role": "EVALUATION_ONLY_MAJOR_UPTREND",
        "account_start_semantics": "INDEPENDENT_EMPTY_ACCOUNT_AT_WINDOW_START",
        "historical_metric_name": "bull",
    },
    {
        "name": "final_evaluation",
        "start": "2026-01-01",
        "end": "2026-09-11",
        "first_session": "2026-01-05",
        "role": "FINAL_EVALUATION_ONLY_NO_PARAMETER_SELECTION",
        "account_start_semantics": "INHERITED_STRATEGY_FULL_PATH_WITH_PREVIOUS_CLOSE_BASE",
        "historical_metric_name": None,
        "state_as_of": "2025-12-31",
    },
    {
        "name": "late_june_august",
        "start": "2026-06-22",
        "end": "2026-08-31",
        "first_session": "2026-06-22",
        "role": "EVALUATION_ONLY_MAJOR_STRESS",
        "account_start_semantics": "INHERITED_STRATEGY_FULL_PATH_WITH_PREVIOUS_CLOSE_BASE",
        "historical_metric_name": "late_june_august",
        "state_as_of": "2026-06-19",
    },
    {
        "name": "july_august",
        "start": "2026-07-01",
        "end": "2026-08-31",
        "first_session": "2026-07-01",
        "role": "EVALUATION_ONLY_CORE_STRESS",
        "account_start_semantics": "INHERITED_STRATEGY_FULL_PATH_WITH_PREVIOUS_CLOSE_BASE",
        "historical_metric_name": "july_august",
        "state_as_of": "2026-06-30",
    },
    {
        "name": "recovery",
        "start": "2026-09-01",
        "end": "2026-09-11",
        "first_session": "2026-09-01",
        "role": "EVALUATION_ONLY_RECOVERY",
        "account_start_semantics": "INHERITED_STRATEGY_FULL_PATH_WITH_PREVIOUS_CLOSE_BASE",
        "historical_metric_name": "recovery",
        "state_as_of": "2026-08-31",
    },
)


class MaterializationError(ValueError):
    """Frozen source evidence is missing, changed, or internally inconsistent."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_sha256(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise MaterializationError(f"{path}: SHA256 {actual} != frozen {expected}")


def _base_key(case: Mapping[str, Any]) -> str:
    config = case["config"]
    value = {
        "effective_universe": case["effective_universe"],
        "data_sha256": case["data_sha256"],
        "source_sha": case["source_sha"],
        "start": case["start"],
        "end": case["end"],
        "initial_cash": case["initial_cash"],
        "initial_holdings": case["initial_holdings"],
        "cost_multiplier": case["cost_multiplier"],
        "delay": case["delay"],
        "commission_bps": config["commission_bps"],
        "slippage_bps": config["slippage_bps"],
        "max_adv": config["max_adv"],
    }
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def group_aliases(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in cases:
        buckets[_base_key(case)].append(case)
    groups: list[dict[str, Any]] = []
    for key, aliases in buckets.items():
        ordered = sorted(aliases, key=lambda item: int(item["index"]))
        exemplar = ordered[0]
        for other in ordered[1:]:
            for field in (
                "effective_universe", "data_sha256", "source_sha", "start", "end",
                "initial_cash", "initial_holdings", "cost_multiplier", "delay",
            ):
                if other[field] != exemplar[field]:
                    raise MaterializationError(f"alias mismatch for {key}: {field}")
            for field in ("commission_bps", "slippage_bps", "max_adv"):
                if other["config"][field] != exemplar["config"][field]:
                    raise MaterializationError(f"alias config mismatch for {key}: {field}")
        groups.append({
            "base_identity_hash": key,
            "canonical": exemplar,
            "aliases": [item["name"] for item in ordered],
            "historical_indices": [int(item["index"]) for item in ordered],
            "groups": sorted({item["group"] for item in ordered}),
            "alias_records": ordered,
        })
    return sorted(groups, key=lambda item: item["historical_indices"][0])


def load_window_metrics(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["case"], row["window"])
            if key in rows:
                raise MaterializationError(f"duplicate historical window metric: {key}")
            if row["status"] != "MEASURED":
                raise MaterializationError(f"unmeasured historical passive row: {key}")
            rows[key] = {
                "start": row["start"],
                "end": row["end"],
                "sessions": int(row["sessions"]),
                "terminal_wealth": float(row["wealth"]),
                "max_drawdown": float(row["max_drawdown"]),
                "inherited_drawdown": float(row["inherited_drawdown"]),
                "orders": int(row["orders"]),
                "fees": float(row["fees"]),
                "slippage": float(row["slippage"]),
            }
    return rows


def _same_metric(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(left[field] == right[field] for field in (
        "start", "end", "sessions", "terminal_wealth", "max_drawdown",
        "inherited_drawdown", "orders", "fees", "slippage",
    ))


def historical_metric_for_aliases(
    metrics: Mapping[tuple[str, str], Mapping[str, Any]],
    aliases: Sequence[str],
    window: Mapping[str, Any],
) -> dict[str, Any] | None:
    metric_name = window["historical_metric_name"]
    if metric_name is None:
        return None
    found = [metrics[(alias, metric_name)] for alias in aliases if (alias, metric_name) in metrics]
    if len(found) != len(aliases):
        missing = [alias for alias in aliases if (alias, metric_name) not in metrics]
        raise MaterializationError(f"missing passive metrics for {metric_name}: {missing}")
    first = found[0]
    if any(not _same_metric(first, other) for other in found[1:]):
        raise MaterializationError(f"exact aliases disagree on passive metric {metric_name}: {aliases}")
    if first["start"] != window["start"] or first["end"] != window["end"]:
        raise MaterializationError(f"window dates changed for {metric_name}")
    return dict(first)


def load_native_attempts(path: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("market_fingerprint") != MARKET_FINGERPRINT:
        raise MaterializationError("native verification market fingerprint changed")
    if payload.get("native_harness_sha256") != NATIVE_HARNESS_SHA256:
        raise MaterializationError("native harness identity changed")
    if payload.get("orchestrator_sha256") != ORCHESTRATOR_SHA256:
        raise MaterializationError("native orchestrator identity changed")
    if payload.get("archive_sha256") != NATIVE_ARCHIVE_SHA256:
        raise MaterializationError("native archive identity changed")
    attempts: dict[tuple[str, str], dict[str, Any]] = {}
    for row in payload.get("outcomes", []):
        key = (row["pool"], row["reference"])
        if key in attempts:
            raise MaterializationError(f"duplicate native attempt: {key}")
        attempts[key] = dict(row)
    return attempts, payload


def references_for_aliases(
    aliases: Sequence[str],
    attempts: Mapping[tuple[str, str], Mapping[str, Any]],
    identity_hash: str,
    window_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    measured: dict[str, Any] = {}
    audit: dict[str, Any] = {}
    for reference in REFERENCE_NAMES:
        rows = [attempts[(alias, reference)] for alias in aliases if (alias, reference) in attempts]
        if not rows:
            continue
        audit[reference] = [dict(row) for row in rows]
        valid = [row for row in rows if row.get("status") == "NATIVE_REPLAY_MEASURED"]
        if not valid or window_name != "full":
            continue
        first = valid[0]
        for other in valid[1:]:
            if (
                other["wealth"] != first["wealth"]
                or other["max_drawdown"] != first["max_drawdown"]
                or other.get("first") != first.get("first")
                or other.get("last") != first.get("last")
            ):
                raise MaterializationError(
                    f"duplicate alias native results disagree for {reference}: {aliases}"
                )
        measured[reference] = {
            "case_identity_hash": identity_hash,
            "evidence_layer": "Native",
            "terminal_wealth": float(first["wealth"]),
            "max_drawdown": float(first["max_drawdown"]),
            "source": {
                "verification_sha256": NATIVE_VERIFICATION_SHA256,
                "native_archive_sha256": NATIVE_ARCHIVE_SHA256,
                "pool": first["pool"],
                "status": first["status"],
            },
        }
    return measured, audit


def initial_positions(case_name: str, window: Mapping[str, Any]) -> dict[str, Any]:
    if window["account_start_semantics"].startswith("INDEPENDENT"):
        return {}
    return {
        "$state_rule": {
            "mode": "ENDOGENOUS_STRATEGY_ACCOUNT_STATE",
            "origin_case": case_name,
            "origin_start_date": "2023-01-03",
            "state_as_of": window["state_as_of"],
            "window_base": "PREVIOUS_SESSION_CLOSE_NAV",
            "result_specific_snapshot_binding_required": True,
        }
    }


def identity_for(case: Mapping[str, Any], window: Mapping[str, Any]) -> dict[str, Any]:
    config = case["config"]
    multiplier = float(case["cost_multiplier"])
    return {
        "evidence_layer": "Native",
        "data_sha256": case["data_sha256"],
        "universe_and_order": list(case["effective_universe"]),
        "start_date": window["start"],
        "end_date": window["end"],
        "account_start_semantics": window["account_start_semantics"],
        "initial_cash": float(case["initial_cash"]),
        "initial_positions": initial_positions(case["name"], window),
        "seed": 34,
        "costs": {
            "commission_bps": float(config["commission_bps"]),
            "transfer_bps": 0.1,
            "sell_stamp_rate": {
                "before_2023-08-28": 0.001,
                "from_2023-08-28": 0.0005,
            },
            "minimum_commission_cny": 5.0,
            "stress_multiplier": multiplier,
        },
        "slippage": {
            "base_bps": float(config["slippage_bps"]),
            "stress_multiplier": multiplier,
        },
        "delay": int(case["delay"]),
        "capacity": {
            "max_adv": float(config["max_adv"]),
            "volume_input": "QFQ_VOLUME_WITH_RAW_CLOSE_NOTIONAL_PROXY",
        },
        "board_lot": {
            "ordinary_buy_step": 100,
            "star_minimum": 200,
            "beijing_minimum": 100,
            "full_odd_lot_disposal_when_capacity_allows": True,
        },
        "tradability_rules": {
            "execution": "DECIDE_AFTER_CLOSE_EXECUTE_NEXT_ELIGIBLE_OPEN",
            "daily_limit_gap_buffer": 0.002,
            "blocked_and_partial_fill_retry": True,
            "t_plus_one_sellable_inventory": True,
            "no_short": True,
            "no_leverage": True,
        },
        "corporate_action_semantics": "FROZEN_QFQ_ADJUSTED_ECONOMIC_UNITS_NOT_ACTUAL_SHARE_LEDGER",
        "runner": {
            "comparison_orchestrator_sha256": ORCHESTRATOR_SHA256,
            "native_reference_harness_sha256": NATIVE_HARNESS_SHA256,
            "quant_runner": "techquant.engine.run",
        },
        "reference_source_sha": {
            "repository": "ychenracing/trades",
            "commit": REFERENCE_COMMIT,
            "selected_subtree_ids": REFERENCE_TREES,
        },
        "runtime_identity": {
            "python": "3.13.5",
            "numpy": "2.3.5",
            "pandas": "2.2.3",
            "market_fingerprint": MARKET_FINGERPRINT,
        },
    }


def materialize(
    cases: Sequence[Mapping[str, Any]],
    metrics: Mapping[tuple[str, str], Mapping[str, Any]],
    attempts: Mapping[tuple[str, str], Mapping[str, Any]],
    native_payload: Mapping[str, Any],
) -> dict[str, Any]:
    aliases = group_aliases(cases)
    output_cases: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for base in aliases:
        canonical = base["canonical"]
        for window in WINDOWS:
            identity = identity_for(canonical, window)
            identity_hash = case_identity_hash(identity)
            if identity_hash in seen_hashes:
                raise MaterializationError(f"duplicate exact case identity: {identity_hash}")
            seen_hashes.add(identity_hash)
            historical = historical_metric_for_aliases(metrics, base["aliases"], window)
            references, attempt_audit = references_for_aliases(
                base["aliases"], attempts, identity_hash, window["name"]
            )
            missing = [name for name in REFERENCE_NAMES if name not in references]
            output_cases.append({
                "case_id": f"{canonical['name']}::{window['name']}::Native",
                "mandatory": True,
                "evidence_layer": "Native",
                "case_identity_hash": identity_hash,
                "identity": identity,
                "base_case": {
                    "canonical_name": canonical["name"],
                    "aliases": base["aliases"],
                    "historical_indices": base["historical_indices"],
                    "source_groups": base["groups"],
                    "raw_universe_order": list(canonical["symbols"]),
                    "effective_execution_order": list(canonical["effective_universe"]),
                    "cost_multiplier": float(canonical["cost_multiplier"]),
                    "delay": int(canonical["delay"]),
                    "historical_identity_sha256": [
                        record["identity_sha256"] for record in base["alias_records"]
                    ],
                },
                "window": {
                    key: value for key, value in window.items()
                    if key != "historical_metric_name"
                },
                "historical_passive_seed": historical,
                "historical_passive_seed_status": (
                    "MEASURED" if historical is not None else "PENDING_EXACT_WINDOW_MEASUREMENT"
                ),
                "reference_results": references,
                "reference_attempt_audit": attempt_audit,
                "reference_status": (
                    "REFERENCE_COMPLETE" if not missing else "REFERENCE_INCOMPLETE"
                ),
                "missing_references": missing,
                "candidate_result": None,
                "candidate_status": "CANDIDATE_UNMEASURED",
                "case_acceptance": "NOT_MET",
            })
    window_order = {window["name"]: index for index, window in enumerate(WINDOWS)}
    output_cases.sort(key=lambda item: (
        item["base_case"]["historical_indices"][0],
        window_order[item["window"]["name"]],
    ))
    complete = [case for case in output_cases if case["reference_status"] == "REFERENCE_COMPLETE"]
    partial = [case for case in output_cases if case["reference_results"]]
    measured_seed = [case for case in output_cases if case["historical_passive_seed"] is not None]
    group_counts: dict[str, int] = defaultdict(int)
    for base in aliases:
        for group in base["groups"]:
            group_counts[group] += 1
    return {
        "schema_version": 2,
        "registry_id": "dynamic_relative_reference_case_matrix",
        "frozen_at": "2026-09-17",
        "source_main_sha": SOURCE_MAIN_SHA,
        "required_evidence_layers": ["Native"],
        "materialization_status": "FROZEN_MATERIALIZED",
        "economic_acceptance": "NOT_MET",
        "case_set_mutation_after_candidate_measurement": "FORBIDDEN",
        "source_evidence": {
            "case_map": {
                "sha256": CASE_MAP_SHA256,
                "git_blob": "4239f36852550047d3af3b1ef99f77f70f80dc67",
                "historical_rows": len(cases),
            },
            "passive_window_metrics": {
                "sha256": WINDOW_METRICS_SHA256,
                "git_blob": "dee913b569e2bacc87b7efd40554bb7ad71d2c5b",
            },
            "native_verification": {
                "sha256": NATIVE_VERIFICATION_SHA256,
                "git_blob": "bd1ccf1a5dc7aec9a398a309bab01ce4068699f1",
                "native_archive_sha256": NATIVE_ARCHIVE_SHA256,
                "attempts": len(native_payload.get("outcomes", [])),
            },
        },
        "case_count": {
            "historical_registrations": len(cases),
            "unique_base_identities": len(aliases),
            "collapsed_alias_registrations": len(cases) - len(aliases),
            "frozen_windows": len(WINDOWS),
            "mandatory_exact_cases": len(output_cases),
            "historical_passive_seed_measured": len(measured_seed),
            "historical_passive_seed_pending": len(output_cases) - len(measured_seed),
            "reference_complete": len(complete),
            "reference_partial_or_complete": len(partial),
            "reference_incomplete": len(output_cases) - len(complete),
            "candidate_unmeasured": len(output_cases),
        },
        "unique_base_groups": dict(sorted(group_counts.items())),
        "window_contract": [
            {key: value for key, value in window.items() if key != "historical_metric_name"}
            for window in WINDOWS
        ],
        "alias_policy": {
            "rule": "One hard gate per exact economic identity; retain all historical aliases and source groups for provenance.",
            "duplicate_aliases_do_not_increase_case_weight": True,
        },
        "inherited_state_policy": {
            "rule": "Later windows are causal slices of each strategy's own full path from the common empty account.",
            "actual_cash_and_positions_are_result_specific": True,
            "each_result_must_bind_the_previous_session_account_snapshot": True,
            "resetting_later_windows_to_cny_2m_empty_cash_is_a_different_case": True,
        },
        "cases": output_cases,
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-map", required=True, type=Path)
    parser.add_argument("--window-metrics", required=True, type=Path)
    parser.add_argument("--native-verification", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    require_sha256(args.case_map, CASE_MAP_SHA256)
    require_sha256(args.window_metrics, WINDOW_METRICS_SHA256)
    require_sha256(args.native_verification, NATIVE_VERIFICATION_SHA256)
    cases = json.loads(args.case_map.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or len(cases) != 199:
        raise MaterializationError("frozen case map must contain exactly 199 registrations")
    metrics = load_window_metrics(args.window_metrics)
    attempts, native_payload = load_native_attempts(args.native_verification)
    result = materialize(cases, metrics, attempts, native_payload)
    write_json(args.output, result)
    print(json.dumps(result["case_count"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

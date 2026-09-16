"""Fixed, sharded formal generalization matrix for fresh challenger authority.

The candidate is fixed before this runner is measured. This module reproduces
only the preregistered case inventory; it never reselects strategy parameters or
silently drops failed cases.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from techquant.config import Config
from techquant.data import file_hash, load_market
from techquant.engine import run
from techquant.evidence import metrics, save_result, source_identity
from research.offensive_fresh_challenger_authority import Owner, Parameters


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def plan_sha256(plan: list[dict]) -> str:
    payload = json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_case_plan(full_symbols: list[str], sectors: dict[str, str], catalog: dict, protocol: dict) -> list[dict]:
    full = list(full_symbols)
    if len(full) != 34 or len(set(full)) != 34:
        raise ValueError("formal matrix requires the frozen 34-name universe")
    if protocol.get("seed") != 34:
        raise ValueError("formal matrix requires the frozen seed")
    cases: list[dict] = []

    def add(name: str, group: str, symbols: list[str], cost: float = 1.0, delay: int = 1) -> None:
        symbols = list(symbols)
        if not symbols or any(symbol not in full for symbol in symbols):
            raise ValueError(f"invalid formal universe: {name}")
        policies = ["control", "treatment", "buy_hold", "equal_weight"] if group == "original_pool" else ["control", "treatment", "buy_hold"]
        cases.append({"index": len(cases), "name": name, "group": group, "symbols": symbols,
                      "cost_multiplier": float(cost), "delay": int(delay), "policies": policies})

    pools = dict(catalog["pools"])
    pools["union"] = full
    for name, symbols in pools.items():
        add(name, "original_pool", list(symbols))

    leaders = {"sz300308", "sz300502", "sz300394"}
    add("remove_optical_leaders", "joint_removal", [s for s in full if s not in leaders])
    for symbol in full:
        add("single_" + symbol, "single", [symbol])
        add("remove_" + symbol, "leave_one_out", [s for s in full if s != symbol])

    for index, sector in enumerate(sorted({sectors[s] for s in full})):
        add(f"remove_sector_{index}", "sector_removal", [s for s in full if sectors[s] != sector])

    rng = np.random.default_rng(protocol["seed"])
    sampled_sizes = (2, 3, 5, 8, 13, 22, 26, 33)
    for size in sampled_sizes:
        for repeat in range(6):
            add(f"sample_{size}_{repeat}", "sampled_subset", sorted(rng.choice(full, size=size, replace=False).tolist()))

    common = list(catalog["pools"]["chatgpt_5"])
    for size in range(1, len(common) + 1):
        for index, symbols in enumerate(itertools.combinations(common, size)):
            add(f"common_exhaustive_{size}_{index}", "common_five_exhaustive", list(symbols))

    extra_rng = np.random.default_rng(protocol["seed"] + 1)
    for size in range(2, len(full)):
        if size not in sampled_sizes:
            add(f"cardinality_{size}", "cardinality_coverage", sorted(extra_rng.choice(full, size=size, replace=False).tolist()))

    add("double_cost", "stress", full, cost=2.0)
    add("triple_cost", "stress", full, cost=3.0)
    add("delayed_open", "stress", full, delay=2)
    if len(cases) != 199 or len({row["name"] for row in cases}) != len(cases):
        raise AssertionError("formal case inventory drifted")
    return cases


def shard_plan(plan: list[dict], shard_index: int, shard_count: int) -> list[dict]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard")
    return [row for row in plan if int(row["index"]) % shard_count == shard_index]


def _policy_result(market, policy: str, cost: float, delay: int):
    cfg = Config()
    if policy == "control":
        return run(market, cfg, policy_factory=lambda current, _cfg: Owner(current, Parameters(False)), cost_multiplier=cost, delay=delay)
    if policy == "treatment":
        return run(market, cfg, policy_factory=lambda current, _cfg: Owner(current, Parameters(True)), cost_multiplier=cost, delay=delay)
    if policy in {"buy_hold", "equal_weight"}:
        return run(market, cfg, benchmark=policy, cost_multiplier=cost, delay=delay)
    raise ValueError(f"unknown policy {policy}")


def _prefix_check(market, full_result, enabled: bool) -> dict:
    cut = market.calendar[500]
    prefix = run(market.prefix(cut), Config(), policy_factory=lambda current, _cfg: Owner(current, Parameters(enabled)))
    pd.testing.assert_frame_equal(full_result.equity.loc[:cut], prefix.equity, check_freq=False)
    pd.testing.assert_frame_equal(full_result.targets.loc[:cut], prefix.targets, check_freq=False)
    return {"status": "PASS", "policy": "treatment" if enabled else "control", "cut": str(cut.date()), "sessions": 501}


def run_shard(data: Path, supplement: Path | None, output: Path, shard_index: int, shard_count: int) -> dict:
    root = Path(__file__).parent
    catalog = json.loads((root / "catalog.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    contract = json.loads((root / "fresh_challenger_formal_validation_contract.json").read_text())
    market = load_market(data, supplement=supplement, sectors=catalog["sectors"])
    if market.fingerprint() != contract["frozen_inputs"]["data_sha256"]:
        raise ValueError("formal validation frozen market identity mismatch")
    if str(market.calendar[0].date()) != contract["frozen_inputs"]["start"] or str(market.calendar[-1].date()) != contract["frozen_inputs"]["end"]:
        raise ValueError("formal validation date range mismatch")

    plan = build_case_plan(list(market.symbols), market.sectors, catalog, protocol)
    digest = plan_sha256(plan)
    selected = shard_plan(plan, shard_index, shard_count)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "identity.json", {
        "source": source_identity(), "candidate": contract["candidate"],
        "formal_contract_sha256": file_hash(root / "fresh_challenger_formal_validation_contract.json"),
        "runner_sha256": file_hash(Path(__file__)), "catalog_sha256": file_hash(root / "catalog.json"),
        "protocol_sha256": file_hash(root / "protocol.json"), "data_sha256": market.fingerprint(),
        "plan_sha256": digest, "shard_index": shard_index, "shard_count": shard_count})
    write_json(output / "case_plan.json", plan)
    write_json(output / "shard_plan.json", selected)

    windows = contract["time_windows"]
    rows: list[dict] = []
    prefix_checks: list[dict] = []
    raw_accounts = 0
    for position, case in enumerate(selected):
        subset = market.subset(case["symbols"])
        full_results = {}
        for policy in case["policies"]:
            result = _policy_result(subset, policy, case["cost_multiplier"], case["delay"])
            save_result(result, output / "runs" / f"{case['name']}__{policy}")
            raw_accounts += 1
            full_results[policy] = result
            average_exposure = float(result.equity.exposure.mean())
            for window, bounds in windows.items():
                measured = metrics(result, bounds[0], bounds[1])
                rows.append({"case_index": case["index"], "case": case["name"], "group": case["group"],
                             "policy": policy, "window": window, "universe_size": len(case["symbols"]),
                             "cost_multiplier": case["cost_multiplier"], "delay": case["delay"],
                             "average_exposure_full": average_exposure, **measured})
        if case["name"] == "union" and case["group"] == "original_pool" and case["cost_multiplier"] == 1.0 and case["delay"] == 1:
            prefix_checks.append(_prefix_check(subset, full_results["control"], False))
            prefix_checks.append(_prefix_check(subset, full_results["treatment"], True))
        pd.DataFrame(rows).to_csv(output / "matrix.csv", index=False, float_format="%.17g")
        if position % 10 == 0:
            print(json.dumps({"shard": shard_index, "completed": position + 1, "total": len(selected), "case": case["name"]}), flush=True)

    write_json(output / "prefix_checks.json", prefix_checks)
    status = {"status": "SHARD_COMPLETE", "shard_index": shard_index, "shard_count": shard_count,
              "case_count": len(selected), "raw_accounts": raw_accounts, "row_count": len(rows),
              "plan_sha256": digest, "prefix_checks": prefix_checks}
    write_json(output / "status.json", status)
    return status


def _ratio_summary(values: pd.Series) -> dict:
    values = values.astype(float)
    return {"count": int(len(values)), "at_least_one": int((values >= 1.0).sum()),
            "fraction_at_least_one": float((values >= 1.0).mean()), "minimum": float(values.min()),
            "median": float(values.median()), "maximum": float(values.max())}


def aggregate(inputs: Path, output: Path) -> dict:
    shard_dirs = sorted(p for p in inputs.iterdir() if p.is_dir() and (p / "status.json").exists())
    if len(shard_dirs) != 4:
        raise ValueError(f"expected four complete shards, found {len(shard_dirs)}")
    statuses = [json.loads((p / "status.json").read_text()) for p in shard_dirs]
    plan_hashes = {s["plan_sha256"] for s in statuses}
    if len(plan_hashes) != 1 or {s["shard_index"] for s in statuses} != {0, 1, 2, 3}:
        raise ValueError("formal shard identities are incomplete or inconsistent")
    plan = json.loads((shard_dirs[0] / "case_plan.json").read_text())
    if plan_sha256(plan) != next(iter(plan_hashes)):
        raise ValueError("formal plan digest mismatch")
    matrix = pd.concat([pd.read_csv(p / "matrix.csv") for p in shard_dirs], ignore_index=True).sort_values(["case_index", "policy", "window"]).reset_index(drop=True)
    output.mkdir(parents=True, exist_ok=False)
    matrix.to_csv(output / "matrix.csv", index=False, float_format="%.17g")
    write_json(output / "case_plan.json", plan)

    full = matrix[matrix.window == "full"].copy()
    control = full[full.policy == "control"].set_index("case")
    treatment = full[full.policy == "treatment"].set_index("case")
    buy_hold = full[full.policy == "buy_hold"].set_index("case")
    expected_cases = {row["name"] for row in plan}
    if set(control.index) != expected_cases or set(treatment.index) != expected_cases or set(buy_hold.index) != expected_cases:
        raise ValueError("missing formal control/treatment/buy-hold case")

    tc_ratio = treatment.wealth / control.wealth
    tb_ratio = treatment.wealth / buy_hold.wealth
    comparison = pd.DataFrame({"case": sorted(expected_cases)}).set_index("case")
    comparison["group"] = [next(row["group"] for row in plan if row["name"] == case) for case in comparison.index]
    comparison["treatment_wealth"] = treatment.loc[comparison.index, "wealth"]
    comparison["control_wealth"] = control.loc[comparison.index, "wealth"]
    comparison["buy_hold_wealth"] = buy_hold.loc[comparison.index, "wealth"]
    comparison["treatment_control_wealth_ratio"] = tc_ratio.loc[comparison.index]
    comparison["treatment_buy_hold_wealth_ratio"] = tb_ratio.loc[comparison.index]
    comparison["treatment_mdd"] = treatment.loc[comparison.index, "max_drawdown"]
    comparison["control_mdd"] = control.loc[comparison.index, "max_drawdown"]
    comparison.to_csv(output / "full_comparison.csv", float_format="%.17g")

    group_rows = []
    for group, part in comparison.groupby("group", sort=True):
        group_rows.append({"group": group, "cases": len(part),
                           "control_ratio_min": float(part.treatment_control_wealth_ratio.min()),
                           "control_ratio_median": float(part.treatment_control_wealth_ratio.median()),
                           "control_wins": int((part.treatment_control_wealth_ratio >= 1).sum()),
                           "buy_hold_ratio_min": float(part.treatment_buy_hold_wealth_ratio.min()),
                           "buy_hold_ratio_median": float(part.treatment_buy_hold_wealth_ratio.median()),
                           "buy_hold_wins": int((part.treatment_buy_hold_wealth_ratio >= 1).sum()),
                           "mdd_not_worse": int((part.treatment_mdd <= part.control_mdd).sum())})
    pd.DataFrame(group_rows).to_csv(output / "group_summary.csv", index=False, float_format="%.17g")

    prefix_checks = [item for p in shard_dirs for item in json.loads((p / "prefix_checks.json").read_text())]
    if len(prefix_checks) != 2 or any(row["status"] != "PASS" for row in prefix_checks):
        raise ValueError("formal prefix causality did not pass")
    write_json(output / "prefix_checks.json", prefix_checks)
    comparison[comparison.group == "original_pool"].to_csv(output / "original_pools.csv", float_format="%.17g")
    comparison[comparison.group == "stress"].to_csv(output / "stress.csv", float_format="%.17g")

    summary = {
        "status": "FORMAL_GENERALIZATION_MEASURED_NOT_FINAL_ACCEPTANCE", "economic_acceptance": "NOT_ESTABLISHED",
        "case_count": len(plan), "account_count": int(sum(s["raw_accounts"] for s in statuses)),
        "matrix_rows": int(len(matrix)), "plan_sha256": next(iter(plan_hashes)),
        "treatment_vs_control": _ratio_summary(tc_ratio), "treatment_vs_buy_hold": _ratio_summary(tb_ratio),
        "worst_control_case": str(tc_ratio.idxmin()), "worst_buy_hold_case": str(tb_ratio.idxmin()),
        "prefix_checks": prefix_checks,
        "candidate_numeric_parameter_stability": "NOT_APPLICABLE_NO_TUNABLE_NUMERIC_CANDIDATE_PARAMETER",
        "mdd_diagnostic": {"not_worse_than_control": int((treatment.max_drawdown <= control.max_drawdown).sum()), "cases": len(plan)},
        "limitations": ["Retrospective fixed-snapshot validation is not a future return guarantee.",
                        "No post-selection numeric candidate neighborhood exists because the promoted candidate adds no tunable numeric strategy parameter.",
                        "Reference-strategy native/corrected/normalized comparison is a separate final acceptance step and is not inferred from this matrix."]}
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    shard = sub.add_parser("shard")
    shard.add_argument("--data", type=Path, required=True); shard.add_argument("--supplement", type=Path)
    shard.add_argument("--output", type=Path, required=True); shard.add_argument("--shard-index", type=int, required=True); shard.add_argument("--shard-count", type=int, required=True)
    agg = sub.add_parser("aggregate")
    agg.add_argument("--inputs", type=Path, required=True); agg.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_shard(args.data, args.supplement, args.output, args.shard_index, args.shard_count) if args.command == "shard" else aggregate(args.inputs, args.output)
    print(json.dumps(result, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()

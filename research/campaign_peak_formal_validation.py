"""Fixed 199-case formal matrix for the promoted campaign-peak alpha candidate.

Case generation and aggregation are imported from the already-used fresh-
challenger formal runner so universe/seed/cost/delay semantics cannot drift.
Only the control/treatment policy class changes to the promoted source-bound
candidate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from techquant.config import Config
from techquant.data import file_hash, load_market
from techquant.engine import run
from techquant.evidence import metrics, save_result, source_identity
from research.offensive_campaign_peak_authority import Owner, Parameters
from research.fresh_challenger_formal_validation import (
    aggregate as aggregate_existing,
    build_case_plan,
    plan_sha256,
    shard_plan,
    write_json,
)


def policy_result(market, policy: str, cost: float, delay: int):
    cfg = Config()
    if policy == "control":
        return run(market, cfg, policy_factory=lambda current, _cfg: Owner(current, Parameters(False)),
                   cost_multiplier=cost, delay=delay)
    if policy == "treatment":
        return run(market, cfg, policy_factory=lambda current, _cfg: Owner(current, Parameters(True)),
                   cost_multiplier=cost, delay=delay)
    if policy in {"buy_hold", "equal_weight"}:
        return run(market, cfg, benchmark=policy, cost_multiplier=cost, delay=delay)
    raise ValueError(f"unknown policy {policy}")


def prefix_check(market, full_result, enabled: bool) -> dict:
    cut = market.calendar[500]
    prefix = run(market.prefix(cut), Config(),
                 policy_factory=lambda current, _cfg: Owner(current, Parameters(enabled)))
    pd.testing.assert_frame_equal(full_result.equity.loc[:cut], prefix.equity, check_freq=False)
    pd.testing.assert_frame_equal(full_result.targets.loc[:cut], prefix.targets, check_freq=False)
    return {"status": "PASS", "policy": "treatment" if enabled else "control",
            "cut": str(cut.date()), "sessions": 501}


def run_shard(data: Path, supplement: Path | None, output: Path,
              shard_index: int, shard_count: int) -> dict:
    root = Path(__file__).parent
    catalog = json.loads((root / "catalog.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    contract = json.loads((root / "campaign_peak_formal_validation_contract.json").read_text())
    market = load_market(data, supplement=supplement, sectors=catalog["sectors"])
    frozen = contract["frozen_inputs"]
    if market.fingerprint() != frozen["data_sha256"]:
        raise ValueError("campaign peak formal frozen market identity mismatch")
    if str(market.calendar[0].date()) != frozen["start"] or str(market.calendar[-1].date()) != frozen["end"]:
        raise ValueError("campaign peak formal date range mismatch")

    plan = build_case_plan(list(market.symbols), market.sectors, catalog, protocol)
    digest = plan_sha256(plan)
    selected = shard_plan(plan, shard_index, shard_count)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "identity.json", {
        "source": source_identity(), "candidate": contract["candidate"],
        "formal_contract_sha256": file_hash(root / "campaign_peak_formal_validation_contract.json"),
        "runner_sha256": file_hash(Path(__file__)), "catalog_sha256": file_hash(root / "catalog.json"),
        "protocol_sha256": file_hash(root / "protocol.json"), "data_sha256": market.fingerprint(),
        "plan_sha256": digest, "shard_index": shard_index, "shard_count": shard_count,
    })
    write_json(output / "case_plan.json", plan)
    write_json(output / "shard_plan.json", selected)

    windows = contract["time_windows"]
    rows, prefix_checks = [], []
    raw_accounts = 0
    for position, case in enumerate(selected):
        subset = market.subset(case["symbols"])
        full_results = {}
        for policy in case["policies"]:
            result = policy_result(subset, policy, case["cost_multiplier"], case["delay"])
            save_result(result, output / "runs" / f"{case['name']}__{policy}")
            raw_accounts += 1
            full_results[policy] = result
            exposure = float(result.equity.exposure.mean())
            for window, bounds in windows.items():
                rows.append({
                    "case_index": case["index"], "case": case["name"], "group": case["group"],
                    "policy": policy, "window": window, "universe_size": len(case["symbols"]),
                    "cost_multiplier": case["cost_multiplier"], "delay": case["delay"],
                    "average_exposure_full": exposure, **metrics(result, bounds[0], bounds[1]),
                })
        if case["name"] == "union" and case["group"] == "original_pool" and case["cost_multiplier"] == 1.0 and case["delay"] == 1:
            prefix_checks.append(prefix_check(subset, full_results["control"], False))
            prefix_checks.append(prefix_check(subset, full_results["treatment"], True))
        pd.DataFrame(rows).to_csv(output / "matrix.csv", index=False, float_format="%.17g")
        if position % 10 == 0:
            print(json.dumps({"shard": shard_index, "completed": position + 1,
                              "total": len(selected), "case": case["name"]}), flush=True)

    write_json(output / "prefix_checks.json", prefix_checks)
    status = {"status": "SHARD_COMPLETE", "shard_index": shard_index, "shard_count": shard_count,
              "case_count": len(selected), "raw_accounts": raw_accounts, "row_count": len(rows),
              "plan_sha256": digest, "prefix_checks": prefix_checks}
    write_json(output / "status.json", status)
    return status


def aggregate(inputs: Path, output: Path) -> dict:
    return aggregate_existing(inputs, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    shard = sub.add_parser("shard")
    shard.add_argument("--data", type=Path, required=True)
    shard.add_argument("--supplement", type=Path)
    shard.add_argument("--output", type=Path, required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--shard-count", type=int, required=True)
    agg = sub.add_parser("aggregate")
    agg.add_argument("--inputs", type=Path, required=True)
    agg.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "shard":
        print(json.dumps(run_shard(args.data, args.supplement, args.output,
                                   args.shard_index, args.shard_count)), flush=True)
    else:
        print(json.dumps(aggregate(args.inputs, args.output)), flush=True)


if __name__ == "__main__":
    main()

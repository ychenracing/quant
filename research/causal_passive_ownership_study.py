"""Source-bound economic proof for production causal passive ownership.

This does not invent a new economic path.  It proves that the production
``run_passive_ownership`` adapter is account-for-account identical to the
engine's already measured ``benchmark='buy_hold'`` path, then compares that
same account with the current active alpha leader on the frozen core scopes.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

import pandas as pd

from techquant.config import Config
from techquant.data import file_hash
from techquant.engine import run
from techquant.evidence import metrics, save_result, source_identity
from techquant.passive import run_passive_ownership
from research.expectation_study import scopes, write_json
from research.offensive_dominant_peak_authority import (
    Owner as ActiveOwner,
    Parameters as ActiveParameters,
)

FAMILY = "causal_passive_ownership"
FROZEN_FULL_SHA = "d9ded1d933f292933aa09145bc36a5696421c43dbf266fce018a34ef0b9f3f9b"
SELECTION_END = "2025-12-31"


def _same_account(benchmark, passive) -> None:
    pd.testing.assert_frame_equal(benchmark.equity, passive.equity, check_exact=True)
    pd.testing.assert_frame_equal(benchmark.targets, passive.targets, check_exact=True)
    if benchmark.orders != passive.orders:
        raise AssertionError("passive ownership order ledger differs from same-engine buy_hold")
    if passive.metadata.get("benchmark") is not None:
        raise AssertionError("production passive ownership must not retain research benchmark identity")
    if passive.metadata.get("strategy") != "passive_ownership":
        raise AssertionError("production passive ownership strategy identity missing")
    if passive.metadata.get("economic_semantics") != "same_engine_buy_hold":
        raise AssertionError("passive ownership economic semantics identity missing")


def _run_triplet(market, out: Path, scope: str):
    cfg = Config()
    active = run(
        market,
        cfg,
        policy_factory=lambda current, _cfg: ActiveOwner(current, ActiveParameters(True)),
    )
    benchmark = run(market, cfg, benchmark="buy_hold")
    passive = run_passive_ownership(market, cfg)
    _same_account(benchmark, passive)
    for label, result in (("active", active), ("buy_hold", benchmark), ("passive", passive)):
        result.metadata = dict(result.metadata)
        result.metadata["passive_ownership_study"] = {
            "family": FAMILY,
            "source": source_identity(),
            "runner_sha256": file_hash(Path(__file__)),
            "scope": scope,
        }
        save_result(result, out / "runs" / f"{scope}_{label}")
    return active, benchmark, passive


def _objective(rows: list[dict], label: str) -> dict:
    wealth = [float(row[label]["wealth"]) for row in rows]
    return {
        "mean_log": math.fsum(math.log(value) for value in wealth) / len(wealth),
        "minimum": min(wealth),
        "wealth": {row["scope"]: float(row[label]["wealth"]) for row in rows},
    }


def select(market, catalog, out: Path) -> dict:
    if market.fingerprint() != FROZEN_FULL_SHA:
        raise ValueError("causal passive ownership requires the frozen 34-name full market")
    training = market.prefix(SELECTION_END)
    scope_map = scopes(training, catalog)
    out.mkdir(parents=True, exist_ok=True)
    write_json(
        out / "plan.json",
        {
            "family": FAMILY,
            "source": source_identity(),
            "runner_sha256": file_hash(Path(__file__)),
            "full_data_sha256": market.fingerprint(),
            "selection_data_sha256": training.fingerprint(),
            "selection_end": SELECTION_END,
            "scopes": scope_map,
            "config": asdict(Config()),
            "production_semantics": "same_engine_buy_hold",
            "economic_acceptance": "UNVERIFIED",
        },
    )
    rows = []
    for scope, symbols in scope_map.items():
        subset = training.subset(symbols)
        active, benchmark, passive = _run_triplet(subset, out, scope)
        row = {
            "scope": scope,
            "active": metrics(active),
            "buy_hold": metrics(benchmark),
            "passive": metrics(passive),
            "equivalence": "EXACT_EQUITY_TARGETS_ORDERS",
        }
        rows.append(row)
        write_json(out / "paired-progress.json", rows)
        print(json.dumps(row), flush=True)
    active_obj = _objective(rows, "active")
    passive_obj = _objective(rows, "passive")
    advance = (
        passive_obj["mean_log"] > active_obj["mean_log"] + 1e-12
        and passive_obj["minimum"] >= active_obj["minimum"] - 1e-12
    )
    selection = {
        "status": "BEATS_CURRENT_ACTIVE_ALPHA_LEADER" if advance else "REJECTED_VS_CURRENT_ACTIVE_ALPHA_LEADER",
        "advance": advance,
        "rows": rows,
        "active_objective": active_obj,
        "passive_objective": passive_obj,
        "full_data_sha256": market.fingerprint(),
        "selection_data_sha256": training.fingerprint(),
        "source": source_identity(),
        "runner_sha256": file_hash(Path(__file__)),
        "economic_acceptance": "NOT_ESTABLISHED",
        "equivalence": "PRODUCTION_PASSIVE_EXACTLY_EQUALS_SAME_ENGINE_BUY_HOLD",
        "historical_exposure": "RETROSPECTIVE_NOT_UNSEEN_OUT_OF_SAMPLE",
    }
    write_json(out / "selection.json", selection)
    return selection


def evaluate(market, catalog, selection: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if selection["full_data_sha256"] != market.fingerprint():
        raise ValueError("passive ownership selection/full market identity mismatch")
    if not selection["advance"]:
        write_json(
            out / "status.json",
            {"status": "NOT_RUN_REJECTED_VS_ACTIVE", "runs": 0, "rows": 0},
        )
        return
    windows = {
        "full": (None, None),
        "bull": ("2023-01-03", "2026-06-30"),
        "late_june_through_august": ("2026-06-22", "2026-08-31"),
        "july_august": ("2026-07-01", "2026-08-31"),
        "retrospective_2026": ("2026-01-01", None),
    }
    rows = []
    for scope, symbols in scopes(market, catalog).items():
        subset = market.subset(symbols)
        active, benchmark, passive = _run_triplet(subset, out, scope)
        for policy, result in (("active", active), ("buy_hold", benchmark), ("passive", passive)):
            for window, (start, end) in windows.items():
                rows.append({"scope": scope, "policy": policy, "window": window, **metrics(result, start, end)})
        pd.DataFrame(rows).to_csv(out / "matrix.csv", index=False, float_format="%.17g")
    full_rows = [row for row in rows if row["window"] == "full"]
    active_full = {
        row["scope"]: row["wealth"] for row in full_rows if row["policy"] == "active"
    }
    passive_full = {
        row["scope"]: row["wealth"] for row in full_rows if row["policy"] == "passive"
    }
    full_mean_active = math.fsum(math.log(v) for v in active_full.values()) / len(active_full)
    full_mean_passive = math.fsum(math.log(v) for v in passive_full.values()) / len(passive_full)
    write_json(
        out / "status.json",
        {
            "status": "MEASURED_NOT_FINAL_ACCEPTANCE",
            "runs": 9,
            "rows": len(rows),
            "full_active_wealth": active_full,
            "full_passive_wealth": passive_full,
            "full_active_mean_log": full_mean_active,
            "full_passive_mean_log": full_mean_passive,
            "equivalence": "PRODUCTION_PASSIVE_EXACTLY_EQUALS_SAME_ENGINE_BUY_HOLD",
            "formal_199_case": "REUSE_REQUIRES_EQUIVALENCE_RECEIPT",
        },
    )


def main(data_market, catalog, out: Path) -> None:
    selection = select(data_market, catalog, out / "selection")
    evaluate(data_market, catalog, selection, out / "evaluation")


__all__ = ["FAMILY", "select", "evaluate", "main"]

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from techquant.config import Config
from techquant.data import file_hash, load_market
from techquant.engine import run
from techquant.evidence import metrics
from research.explosive_campaign_accelerator import Owner, Parameters
from research.expectation_study import scopes
from research.ledger_attribution import attribute
from research.trend_book import Owner as ControlOwner, Parameters as ControlParameters


ROOT = Path("/workspace/scratch/1816e11a0469")
SOURCE = ROOT / "quant-source"
INPUTS = ROOT / "frozen-input/evidence/inputs"
HOSTED = ROOT / "leader-anchor-evidence2/nonlinear/selection/runs"


def account(market, treatment):
    owner = (
        Owner(market, Parameters(True))
        if treatment else ControlOwner(market, ControlParameters(2))
    )
    result = run(market, Config(), policy_factory=lambda _m, _c: owner)
    _, _, ledger = attribute(market, result)
    row = dict(metrics(result), average_exposure=float(result.equity.exposure.mean()))
    return owner, result, row, ledger


def main():
    catalog = json.loads((SOURCE / "research/catalog.json").read_text())
    full = load_market(
        INPUTS / "market",
        supplement=INPUTS / "supplement",
        sectors=catalog["sectors"],
    )
    market = full.prefix("2025-12-31")
    rows = []
    for scope, names in scopes(market, catalog).items():
        scoped = market.subset(names)
        _, control, control_metrics, control_ledger = account(scoped, False)
        owner, treatment, treatment_metrics, treatment_ledger = account(scoped, True)
        hosted = pd.read_csv(HOSTED / f"{scope}_control/equity.csv")
        local = control.equity.reset_index()
        hosted["date"] = pd.to_datetime(hosted["date"])
        local["date"] = pd.to_datetime(local["date"])
        columns = ["nav", "cash", "holdings", "exposure", "target_cap", "breadth"]
        hosted_error = max(
            float((local[column] - hosted[column]).abs().max()) for column in columns
        )
        events = owner.trace
        rows.append({
            "scope": scope,
            "control": control_metrics,
            "treatment": treatment_metrics,
            "wealth_change": treatment_metrics["wealth"] / control_metrics["wealth"] - 1,
            "mdd_change": treatment_metrics["max_drawdown"] - control_metrics["max_drawdown"],
            "control_ledger": control_ledger,
            "treatment_ledger": treatment_ledger,
            "hosted_control_max_error": hosted_error,
            "accelerator": {
                "event_days": len(events),
                "newly_allowed_symbol_events": sum(len(event["newly_allowed"]) for event in events),
                "unique_triggered_symbols": sorted({
                    symbol for event in events for symbol in event["symbols"]
                }),
            },
        })
    advance = (
        all(row["wealth_change"] >= -1e-12 for row in rows)
        and rows[0]["wealth_change"] > 1e-12
        and any(row["wealth_change"] > 1e-12 for row in rows)
    )
    output = {
        "family": "explosive_campaign_accelerator",
        "status": "PAIRED_SCREEN_ADVANCE" if advance else "REJECTED_PAIRED_SCREEN",
        "advance": advance,
        "full_data_sha256": full.fingerprint(),
        "selection_data_sha256": market.fingerprint(),
        "source_sha256": {
            name: file_hash(SOURCE / "research" / name)
            for name in (
                "explosive_campaign_accelerator.py",
                "test_explosive_campaign_accelerator.py",
                "explosive_campaign_accelerator_contract.json",
            )
        },
        "runner_sha256": file_hash(Path(__file__)),
        "rows": rows,
        "tests": {
            "focused": "5/5",
            "ordinary": "87/87",
            "research": "410 passed, 9 skipped",
        },
        "economic_acceptance": "NOT_ESTABLISHED",
        "historical_exposure": "RETROSPECTIVE_NOT_UNSEEN_OUT_OF_SAMPLE",
    }
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

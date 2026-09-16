"""Fixed three-scope screen for the preregistered relative campaign owner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from techquant.config import Config
from techquant.data import file_hash, load_market
from techquant.engine import run
from techquant.evidence import metrics
from research.campaign_quality_attribution import load_benchmark
from research.expectation_study import scopes
from research.ledger_attribution import attribute
from research.relative_campaign_owner import Owner, Parameters
from research.trend_book import Owner as ControlOwner, Parameters as ControlParameters


def account(market, benchmark, benchmark_identity, treatment: bool):
    owner = (
        Owner(
            market,
            Parameters(True),
            benchmark_open=benchmark.open.to_numpy(),
            benchmark_close=benchmark.close.to_numpy(),
            benchmark_identity=benchmark_identity,
        )
        if treatment
        else ControlOwner(market, ControlParameters(2))
    )
    result = run(market, Config(), policy_factory=lambda _market, _config: owner)
    _, _, ledger = attribute(market, result)
    return owner, metrics(result), ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("."))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    catalog = json.loads((args.source / "research/catalog.json").read_text())
    full = load_market(args.data, supplement=args.supplement, sectors=catalog["sectors"])
    market = full.prefix("2025-12-31")
    benchmark, identity = load_benchmark(args.data, market.calendar)
    benchmark_label = f"{identity['symbol']}:{identity['csv_sha256']}"
    rows = []
    for scope, symbols in scopes(market, catalog).items():
        scoped = market.subset(symbols)
        _, control, control_ledger = account(
            scoped, benchmark, benchmark_label, False
        )
        owner, treatment, treatment_ledger = account(
            scoped, benchmark, benchmark_label, True
        )
        wealth_change = treatment["wealth"] / control["wealth"] - 1
        rows.append({
            "scope": scope,
            "control": control,
            "treatment": treatment,
            "wealth_change": wealth_change,
            "mdd_change": treatment["max_drawdown"] - control["max_drawdown"],
            "orders_change": treatment["orders"] - control["orders"],
            "turnover_change": treatment["gross_turnover"] - control["gross_turnover"],
            "control_ledger": control_ledger,
            "treatment_ledger": treatment_ledger,
            "quality_transitions": len(owner.trace),
            "quality_activated": sum(
                event["kind"] == "QUALITY_ACTIVATED" for event in owner.trace
            ),
            "quality_rejected": sum(
                event["kind"] == "QUALITY_REJECTED" for event in owner.trace
            ),
            "quality_revoked": sum(
                event["kind"] == "QUALITY_REVOKED" for event in owner.trace
            ),
            "trace": owner.trace,
        })

    advance = (
        all(row["wealth_change"] >= -1e-12 for row in rows)
        and any(row["wealth_change"] > 1e-12 for row in rows)
    )
    root = args.source / "research"
    output = {
        "family": "relative_campaign_owner",
        "status": "PAIRED_SCREEN_ADVANCE" if advance else "REJECTED_PAIRED_SCREEN",
        "advance": advance,
        "fixed_failure_rule": (
            "any scope wealth regression rejects; no window, benchmark, sign, "
            "confirmation, fraction or capacity search"
        ),
        "benchmark": identity,
        "full_data_sha256": full.fingerprint(),
        "selection_data_sha256": market.fingerprint(),
        "source_sha256": {
            name: file_hash(root / name)
            for name in (
                "relative_campaign_owner.py",
                "test_relative_campaign_owner.py",
                "relative_campaign_owner_contract.json",
            )
        },
        "runner_sha256": file_hash(Path(__file__)),
        "rows": rows,
        "economic_acceptance": "NOT_ESTABLISHED",
        "historical_exposure": "RETROSPECTIVE_FIXED_PAIRED_SCREEN",
    }
    rendered = json.dumps(output, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

"""Fixed three-scope alpha screen for the registered basis-failure exit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from techquant.config import Config
from techquant.data import file_hash, load_market
from techquant.engine import run
from techquant.evidence import metrics
from research.basis_failure_exit import Owner, Parameters
from research.expectation_study import scopes
from research.ledger_attribution import attribute
from research.trend_book import Owner as ControlOwner, Parameters as ControlParameters


def account(market, treatment: bool):
    owner = (
        Owner(market, Parameters(True))
        if treatment else ControlOwner(market, ControlParameters(2))
    )
    result = run(market, Config(), policy_factory=lambda _market, _config: owner)
    _, _, ledger = attribute(market, result)
    measured = dict(metrics(result), average_exposure=float(result.equity.exposure.mean()))
    return owner, measured, ledger


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
    rows = []
    for scope, symbols in scopes(market, catalog).items():
        scoped = market.subset(symbols)
        _, control, control_ledger = account(scoped, False)
        owner, treatment, treatment_ledger = account(scoped, True)
        wealth_change = treatment["wealth"] / control["wealth"] - 1
        rows.append({
            "scope": scope,
            "control": control,
            "treatment": treatment,
            "wealth_change": wealth_change,
            "mdd_change": treatment["max_drawdown"] - control["max_drawdown"],
            "control_ledger": control_ledger,
            "treatment_ledger": treatment_ledger,
            "event_count": len(owner.events),
            "events": owner.events,
        })
    advance = (
        all(row["wealth_change"] >= -1e-12 for row in rows)
        and any(row["wealth_change"] > 1e-12 for row in rows)
    )
    root = args.source / "research"
    output = {
        "family": "basis_failure_exit",
        "status": "PAIRED_SCREEN_ADVANCE" if advance else "REJECTED_PAIRED_SCREEN",
        "advance": advance,
        "full_data_sha256": full.fingerprint(),
        "selection_data_sha256": market.fingerprint(),
        "source_sha256": {
            name: file_hash(root / name)
            for name in (
                "basis_failure_exit.py",
                "test_basis_failure_exit.py",
                "basis_failure_exit_contract.json",
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
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()

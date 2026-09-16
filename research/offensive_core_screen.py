"""Fixed pre-2026 three-scope alpha screen for the offensive core candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from techquant.config import Config
from techquant.data import file_hash, load_market
from techquant.engine import run
from techquant.evidence import metrics, save_result
from research.expectation_study import scopes
from research.ledger_attribution import attribute
from research.offensive_core import Owner, Parameters


CUTOFF = "2025-12-31"


def alpha_decision(rows: list[dict]) -> dict:
    nonregression = all(row["wealth_change"] >= -1e-12 for row in rows)
    strict = any(row["wealth_change"] > 1e-12 for row in rows)
    return {
        "advance": bool(nonregression and strict),
        "wealth_nonregression_all_scopes": bool(nonregression),
        "strict_wealth_improvement": bool(strict),
    }


def account(market, output: Path, scope: str, treatment: bool):
    label = "treatment" if treatment else "control"
    owner = Owner(market, Parameters(treatment))
    result = run(
        market,
        Config(),
        policy_factory=lambda _market, _config: owner,
    )
    path = output / "runs" / f"{scope}_{label}"
    save_result(result, path)
    days, episodes, ledger = attribute(market, result)
    attribution = output / "attribution" / f"{scope}_{label}"
    attribution.mkdir(parents=True, exist_ok=True)
    days.to_csv(attribution / "pnl.csv", index=False, float_format="%.17g")
    episodes.to_csv(attribution / "episodes.csv", index=False, float_format="%.17g")
    return owner, {
        **metrics(result),
        "average_exposure": float(result.equity.exposure.mean()),
    }, ledger


def paired(full, catalog: dict, output: Path) -> dict:
    market = full.prefix(CUTOFF)
    rows = []
    for scope, symbols in scopes(market, catalog).items():
        scoped = market.subset(symbols)
        _, control, control_ledger = account(scoped, output, scope, False)
        owner, treatment, treatment_ledger = account(scoped, output, scope, True)
        wealth_change = treatment["wealth"] / control["wealth"] - 1.0
        rows.append({
            "scope": scope,
            "control": control,
            "treatment": treatment,
            "wealth_change": float(wealth_change),
            "mdd_change": float(treatment["max_drawdown"] - control["max_drawdown"]),
            "orders_change": int(treatment["orders"] - control["orders"]),
            "turnover_change": float(treatment["gross_turnover"] - control["gross_turnover"]),
            "control_ledger": control_ledger,
            "treatment_ledger": treatment_ledger,
            "offensive_reviews": len(owner.trace),
            "full_exposure_selections": sum(
                row.get("action") == "FULL_EXPOSURE_SELECTION" for row in owner.trace
            ),
            "cash_no_eligible": sum(
                row.get("action") == "CASH_NO_ELIGIBLE_NAME" for row in owner.trace
            ),
        })

    decision = alpha_decision(rows)
    root = Path(__file__).parent
    result = {
        "family": "offensive_core",
        "status": "PAIRED_SCREEN_ADVANCE" if decision["advance"] else "REJECTED_PAIRED_SCREEN",
        **decision,
        "selection_end": CUTOFF,
        "full_data_sha256": full.fingerprint(),
        "selection_data_sha256": market.fingerprint(),
        "source_sha256": {
            name: file_hash(root / name)
            for name in (
                "offensive_core.py",
                "test_offensive_core.py",
                "offensive_core_contract.json",
                "offensive_core_screen.py",
            )
        },
        "rows": rows,
        "economic_acceptance": "NOT_ESTABLISHED",
        "historical_exposure": "RETROSPECTIVE_FIXED_PAIRED_SCREEN",
    }
    return result


def hosted(output: Path, data: Path, supplement: Path) -> dict:
    catalog = json.loads(Path("research/catalog.json").read_text())
    full = load_market(data, supplement=supplement, sectors=catalog["sectors"])
    selection = output / "selection"
    selection.mkdir(parents=True, exist_ok=True)
    result = paired(full, catalog, selection)
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    (selection / "selection.json").write_text(rendered, encoding="utf-8")
    (selection / "plan.json").write_text(
        json.dumps({
            "family": "offensive_core",
            "window": ["2023-01-03", CUTOFF],
            "scopes": ["union", "chatgpt_5", "joint_optical_leader_removal"],
            "pair_members": ["control", "treatment"],
            "alpha_screen": "wealth non-regression all scopes and strict improvement at least one",
            "full_data_sha256": full.fingerprint(),
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = hosted(args.output, args.data, args.supplement)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

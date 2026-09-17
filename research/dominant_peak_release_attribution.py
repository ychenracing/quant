"""Read-only source-bound attribution for dominant peak authority events."""
from __future__ import annotations

import json
from pathlib import Path

from techquant.config import Config
from techquant.engine import run
from techquant.evidence import metrics, source_identity
from research.expectation_study import scopes
from research.offensive_campaign_peak_authority import Owner as ControlOwner, Parameters as ControlParameters
from research.offensive_dominant_peak_authority import Owner as TreatmentOwner, Parameters as TreatmentParameters


def first_filled_order(result, *, symbol: str, side: str, after_date: str):
    for row in result.orders:
        if row.get("status") != "FILLED" or row.get("symbol") != symbol or row.get("side") != side:
            continue
        if str(row.get("date")) >= after_date:
            return {k: row.get(k) for k in ("date", "symbol", "side", "units", "price", "notional", "fee")}
    return None


def attribute(market, catalog, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for scope, names in scopes(market, catalog).items():
        subset = market.subset(names)
        control_owner = ControlOwner(subset, ControlParameters(True))
        treatment_owner = TreatmentOwner(subset, TreatmentParameters(True))
        control = run(subset, Config(), policy_factory=lambda _m, _c: control_owner)
        treatment = run(subset, Config(), policy_factory=lambda _m, _c: treatment_owner)
        events = [r for r in treatment_owner.trace if r.get("kind") == "DOMINANT_PEAK_AUTHORITY_EVENT"]
        enriched = []
        for event in events:
            item = dict(event)
            symbol = event.get("symbol")
            challenger = event.get("challenger")
            date = str(event.get("date"))
            if symbol:
                item["next_incumbent_sell"] = first_filled_order(treatment, symbol=symbol, side="SELL", after_date=date)
            if challenger:
                item["next_challenger_buy"] = first_filled_order(treatment, symbol=challenger, side="BUY", after_date=date)
            enriched.append(item)
        row = {
            "scope": scope,
            "source": source_identity(),
            "control": metrics(control),
            "treatment": metrics(treatment),
            "terminal_wealth_delta": float(metrics(treatment)["wealth"] - metrics(control)["wealth"]),
            "events": enriched,
        }
        rows.append(row)
        (output / f"{scope}.json").write_text(json.dumps(row, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    summary = {
        "status": "READ_ONLY_RETROSPECTIVE_ATTRIBUTION_NOT_A_TRADING_RULE",
        "source": source_identity(),
        "rows": rows,
        "limitations": [
            "Later fills are retrospective diagnostics and are not valid inputs to the historical decision.",
            "This attribution does not itself authorize a new rule; any next mechanism must be preregistered from causal state only."
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return summary


__all__ = ["attribute"]

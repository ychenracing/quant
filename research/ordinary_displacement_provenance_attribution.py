"""Read-only recovered-campaign provenance at ordinary displacement events."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from techquant.config import Config
from techquant.data import load_market
from techquant.engine import run
from research.expectation_study import scopes
from research.fresh_challenger_formal_validation import build_case_plan
from research.ledger_attribution import attribute
from research.offensive_fresh_challenger_authority import Owner, Parameters
from research.ordinary_displacement_state_attribution import (
    DIAGNOSTIC_FORMAL_CASES,
    SELECTION_END,
    group,
    label_challenger_episode,
    write_json,
)


def rows(owner: Owner, episodes):
    recovered = set()
    output = []
    release_actions = {
        "RECOVERED_CAMPAIGN_ACTUAL_ESTABLISHMENT",
        "RECOVERED_CAMPAIGN_FRESH_EPOCH",
        "TREND_EDGE_REARM",
    }
    for event in owner.trace:
        if event.get("kind") == "FRESH_CHALLENGER_AUTHORITY_EVENT":
            action = event.get("action")
            if action == "REFERENCE_ALPHA_REARM":
                recovered.update(event.get("symbols", []))
            elif action in release_actions:
                recovered.difference_update(event.get("symbols", []))
        if event.get("kind") == "ALPHA_DECAY_DISPLACEMENT_EVENT" and event.get("action") == "ALPHA_DECAY_DISPLACEMENT":
            output.append({
                "date": event["date"],
                "session": int(event["session"]),
                "incumbent": event["symbol"],
                "challenger": event["challenger"],
                "challenger_recovered_without_fresh_epoch": event["challenger"] in recovered,
                "incumbent_reference": float(event["owned_alpha_reference"]),
                "incumbent_score": float(event["incumbent_score"]),
                "challenger_score": float(event["challenger_score"]),
                **label_challenger_episode(owner.market, episodes, event),
            })
    return output


def analyze(name, market):
    owner = Owner(market, Parameters(True))
    result = run(market, Config(), policy_factory=lambda _m, _c: owner)
    _, episodes, _ = attribute(market, result)
    events = rows(owner, episodes)
    return {
        "case": name,
        "events": len(events),
        "by_recovered_without_fresh_epoch": group(events, "challenger_recovered_without_fresh_epoch"),
        "ordinary_displacements": events,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--supplement", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).parent
    catalog = json.loads((root / "catalog.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    full = load_market(a.data, supplement=a.supplement, sectors=catalog["sectors"])
    selected = full.prefix(SELECTION_END)
    case_symbols = {name: list(symbols) for name, symbols in scopes(selected, catalog).items()}
    formal = {row["name"]: row for row in build_case_plan(list(full.symbols), full.sectors, catalog, protocol)}
    for name in DIAGNOSTIC_FORMAL_CASES:
        case_symbols[name] = list(formal[name]["symbols"])
    result = {
        "status": "READ_ONLY_RECOVERED_PROVENANCE_ATTRIBUTION_NOT_CANDIDATE",
        "full_data_sha256": full.fingerprint(),
        "selection_data_sha256": selected.fingerprint(),
        "selection_end": SELECTION_END,
        "interpretation": "Recovered provenance is reconstructed only from causal trace state. Later campaign PnL is retrospective attribution and never a decision input.",
        "cases": [analyze(name, selected.subset(symbols)) for name, symbols in case_symbols.items()],
    }
    write_json(a.output, result)
    print(json.dumps({"status": result["status"], "cases": [{"case": row["case"], "groups": row["by_recovered_without_fresh_epoch"]} for row in result["cases"]]}), flush=True)


if __name__ == "__main__":
    main()

"""Read-only state attribution for ordinary full-book alpha displacement.

This diagnostic never changes a target.  It observes the current promoted
fresh-challenger owner and records causal state available at each ordinary
ALPHA_DECAY_DISPLACEMENT.  Actual later challenger campaign PnL is retained only
as a retrospective outcome label, never as an input to a decision rule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from techquant.config import Config
from techquant.data import file_hash, load_market
from techquant.engine import run
from techquant.evidence import metrics, source_identity
from research.expectation_study import scopes
from research.fresh_challenger_formal_validation import build_case_plan
from research.ledger_attribution import attribute
from research.offensive_fresh_challenger_authority import Owner, Parameters


SELECTION_END = "2025-12-31"
DIAGNOSTIC_FORMAL_CASES = ("cardinality_16", "cardinality_14", "sample_8_4")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def label_challenger_episode(market, episodes: pd.DataFrame, event: dict) -> dict:
    i = int(event["session"])
    if i + 2 >= len(market.calendar):
        return {"campaign_funded": False, "campaign_label": "RIGHT_CENSORED"}
    signal = str(market.calendar[i + 1].date())
    challenger = event["challenger"]
    matches = episodes[(episodes.symbol == challenger) & (episodes.entry_signal == signal)]
    if len(matches) != 1:
        return {"campaign_funded": False, "campaign_label": "NO_EXACT_NEXT_CLOSE_FILL"}
    row = matches.iloc[0]
    return {
        "campaign_funded": True,
        "campaign_label": "ACTUAL_FUNDED_EPISODE",
        "campaign_entry": str(row.entry),
        "campaign_exit": str(row.exit),
        "campaign_pnl_cny": float(row.pnl),
        "campaign_return_on_buys": float(row.return_on_buys),
    }


def event_state(owner: Owner, event: dict, episodes: pd.DataFrame) -> dict:
    b = owner.base
    i = int(event["session"])
    incumbent = owner.market.symbols.index(event["symbol"])
    challenger = owner.market.symbols.index(event["challenger"])
    p = b.price_signals
    score = b.features.score
    trend = b.trend
    incumbent_health = bool(
        p.ready[i, incumbent]
        and trend.entry[i, incumbent]
        and p.price[i, incumbent] > p.ema20[i, incumbent]
        and p.momentum5[i, incumbent] > 0
        and np.isfinite(score[i, incumbent])
        and score[i, incumbent] > 0
        and not trend.exit[i, incumbent]
        and p.ret1[i, incumbent] > -0.08
    )
    challenger_edge = bool(trend.entry[i, challenger] and (i == 0 or not trend.entry[i - 1, challenger]))
    challenger_rising = bool(i > 0 and np.isfinite(score[i - 1, challenger]) and score[i, challenger] > score[i - 1, challenger])
    incumbent_falling = bool(i > 0 and np.isfinite(score[i - 1, incumbent]) and score[i, incumbent] < score[i - 1, incumbent])
    return {
        "date": event["date"], "session": i,
        "incumbent": event["symbol"], "challenger": event["challenger"],
        "incumbent_reference": float(event["owned_alpha_reference"]),
        "incumbent_score": float(event["incumbent_score"]),
        "challenger_score": float(event["challenger_score"]),
        "incumbent_admission_health": incumbent_health,
        "challenger_fresh_entry_edge": challenger_edge,
        "challenger_score_rising": challenger_rising,
        "incumbent_score_falling": incumbent_falling,
        **label_challenger_episode(owner.market, episodes, event),
    }


def group(rows: list[dict], key: str) -> dict:
    out = {}
    for state in (False, True):
        chosen = [r for r in rows if r[key] is state]
        funded = [r for r in chosen if r["campaign_funded"]]
        pnl = [r["campaign_pnl_cny"] for r in funded]
        out[str(state).lower()] = {
            "events": len(chosen), "funded_campaigns": len(funded),
            "wins": int(sum(value > 0 for value in pnl)),
            "win_rate": float(np.mean([value > 0 for value in pnl])) if pnl else None,
            "median_pnl_cny": float(np.median(pnl)) if pnl else None,
            "total_pnl_cny": float(sum(pnl)),
        }
    return out


def analyze_case(name: str, market) -> dict:
    owner = Owner(market, Parameters(True))
    result = run(market, Config(), policy_factory=lambda _m, _c: owner)
    _, episodes, ledger = attribute(market, result)
    ordinary = [r for r in owner.trace if r.get("kind") == "ALPHA_DECAY_DISPLACEMENT_EVENT" and r.get("action") == "ALPHA_DECAY_DISPLACEMENT"]
    rows = [event_state(owner, event, episodes) for event in ordinary]
    return {
        "case": name,
        "metrics": {**metrics(result), "average_exposure": float(result.equity.exposure.mean())},
        "ledger": ledger,
        "ordinary_displacements": len(rows),
        "by_incumbent_admission_health": group(rows, "incumbent_admission_health"),
        "by_challenger_fresh_entry_edge": group(rows, "challenger_fresh_entry_edge"),
        "by_challenger_score_rising": group(rows, "challenger_score_rising"),
        "by_incumbent_score_falling": group(rows, "incumbent_score_falling"),
        "events": rows,
    }


def main() -> None:
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
    core = scopes(selected, catalog)
    plan = build_case_plan(list(full.symbols), full.sectors, catalog, protocol)
    by_name = {row["name"]: row for row in plan}
    cases = {name: list(symbols) for name, symbols in core.items()}
    for name in DIAGNOSTIC_FORMAL_CASES:
        cases[name] = list(by_name[name]["symbols"])
    result = {
        "status": "READ_ONLY_ORDINARY_DISPLACEMENT_STATE_ATTRIBUTION_NOT_CANDIDATE",
        "source": source_identity(), "source_sha256": file_hash(Path(__file__)),
        "full_data_sha256": full.fingerprint(), "selection_data_sha256": selected.fingerprint(),
        "selection_end": SELECTION_END,
        "formal_tail_cases": list(DIAGNOSTIC_FORMAL_CASES),
        "interpretation": "Later campaign PnL is retrospective attribution only. Causal booleans are computed solely from the signal close and prior close.",
        "cases": [],
    }
    for name, symbols in cases.items():
        result["cases"].append(analyze_case(name, selected.subset(symbols)))
    write_json(a.output, result)
    print(json.dumps({"status": result["status"], "cases": [{"case": r["case"], "events": r["ordinary_displacements"]} for r in result["cases"]]}), flush=True)


if __name__ == "__main__":
    main()

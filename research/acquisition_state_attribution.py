from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from techquant.config import Config
from techquant.data import load_market
from research.expectation_study import scopes


ROOT = Path("/workspace/scratch/1816e11a0469")
SOURCE = ROOT / "quant-source"
INPUTS = ROOT / "frozen-input/evidence/inputs"
RUNS = ROOT / "coherent-evidence/payload/nonlinear/evaluation/runs"
CUTOFF = pd.Timestamp("2025-12-31")
EPSILON = 1e-8


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def closed_campaigns(folder: Path) -> list[dict]:
    orders = pd.read_csv(folder / "orders.csv")
    orders["date"] = pd.to_datetime(orders["date"])
    orders["signal_date"] = pd.to_datetime(orders["signal_date"])
    orders = orders[(orders.date <= CUTOFF) & orders.status.eq("FILLED")].copy()
    campaigns: list[dict] = []
    for symbol, symbol_orders in orders.groupby("symbol", sort=True):
        units = 0.0
        current = None
        for row in symbol_orders.sort_values(["date", "side"]).itertuples():
            signed = row.units if row.side == "BUY" else -row.units
            if units <= EPSILON and signed > 0:
                current = {
                    "symbol": symbol,
                    "signal_date": row.signal_date,
                    "entry_date": row.date,
                    "exit_date": None,
                    "cash_pnl": 0.0,
                    "orders": 0,
                }
            if current is None:
                raise AssertionError(f"sale without an open campaign: {symbol} {row.date}")
            current["orders"] += 1
            if row.side == "BUY":
                current["cash_pnl"] -= row.notional + row.fee
            else:
                current["cash_pnl"] += row.notional - row.fee
            units += signed
            if units < -EPSILON:
                raise AssertionError(f"negative inventory: {symbol} {row.date} {units}")
            if units <= EPSILON:
                current["exit_date"] = row.date
                current["holding_sessions"] = None
                campaigns.append(current)
                current = None
                units = 0.0
        # Open campaigns are right-censored and deliberately excluded. Their
        # terminal mark is not used to make this acquisition-state decision.
    return campaigns


def prefix_state(close: pd.DataFrame, campaign: dict, config: Config) -> dict:
    date = campaign["signal_date"]
    symbol = campaign["symbol"]
    location = close.index.get_loc(date)
    if not isinstance(location, (int, np.integer)):
        raise AssertionError(f"non-unique signal date: {date}")
    if location < config.slow:
        return {"mature": False, "accelerating_persistence": False}
    p0 = float(close.iloc[location][symbol])
    pfast = float(close.iloc[location - config.fast][symbol])
    pslow = float(close.iloc[location - config.slow][symbol])
    if not (np.isfinite(p0) and np.isfinite(pfast) and np.isfinite(pslow)
            and min(p0, pfast, pslow) > 0):
        return {"mature": False, "accelerating_persistence": False}
    recent_velocity = np.log(p0 / pfast) / config.fast
    prior_velocity = np.log(pfast / pslow) / (config.slow - config.fast)
    return {
        "mature": True,
        "accelerating_persistence": bool(recent_velocity > prior_velocity > 0),
        "recent_log_velocity": float(recent_velocity),
        "prior_log_velocity": float(prior_velocity),
    }


def summarize(rows: list[dict], state: bool) -> dict:
    selected = [row for row in rows if row["accelerating_persistence"] is state]
    pnl = np.array([row["cash_pnl"] for row in selected], dtype=float)
    positive_pnl = float(np.maximum(pnl, 0).sum()) if len(pnl) else 0.0
    negative_pnl = float(np.minimum(pnl, 0).sum()) if len(pnl) else 0.0
    return {
        "campaigns": len(selected),
        "positive_campaigns": int((pnl > 0).sum()),
        "win_rate": float((pnl > 0).mean()) if len(pnl) else None,
        "median_cash_pnl": float(np.median(pnl)) if len(pnl) else None,
        "total_cash_pnl": float(pnl.sum()),
        "positive_cash_pnl": positive_pnl,
        "negative_cash_pnl": negative_pnl,
    }


def finite(value):
    value = float(value)
    return value if np.isfinite(value) else None


def main() -> None:
    config = Config()
    catalog = json.loads((SOURCE / "research/catalog.json").read_text())
    full = load_market(
        INPUTS / "market", supplement=INPUTS / "supplement", sectors=catalog["sectors"]
    )
    market = full.prefix(CUTOFF)
    output = {
        "status": "READ_ONLY_ACQUISITION_STATE_ATTRIBUTION_NOT_CANDIDATE",
        "hypothesis": (
            "At the first buy signal, the latest fast-window log-return velocity exceeds "
            "the preceding slow-minus-fast velocity, and both are positive."
        ),
        "state": (
            "log(P[t]/P[t-fast])/fast > "
            "log(P[t-fast]/P[t-slow])/(slow-fast) > 0"
        ),
        "windows": {"fast": config.fast, "slow": config.slow},
        "selection_end": str(CUTOFF.date()),
        "data_sha256": market.fingerprint(),
        "account_source": "previously authenticated incumbent accounts",
        "account_manifest_sha256": {},
        "scopes": [],
    }
    for scope, symbols in scopes(market, catalog).items():
        scoped = market.subset(symbols)
        folder = RUNS / f"{scope}_incumbent"
        close = scoped.panel("close").ffill()
        rows = []
        for campaign in closed_campaigns(folder):
            state = prefix_state(close, campaign, config)
            if not state["mature"]:
                continue
            entry_i = int(scoped.calendar.get_loc(campaign["entry_date"]))
            exit_i = int(scoped.calendar.get_loc(campaign["exit_date"]))
            rows.append({
                "symbol": campaign["symbol"],
                "signal_date": str(campaign["signal_date"].date()),
                "entry_date": str(campaign["entry_date"].date()),
                "exit_date": str(campaign["exit_date"].date()),
                "holding_sessions": exit_i - entry_i + 1,
                "cash_pnl": finite(campaign["cash_pnl"]),
                "accelerating_persistence": state["accelerating_persistence"],
                "recent_log_velocity": finite(state["recent_log_velocity"]),
                "prior_log_velocity": finite(state["prior_log_velocity"]),
            })
        true_summary = summarize(rows, True)
        false_summary = summarize(rows, False)
        total_positive = true_summary["positive_cash_pnl"] + false_summary["positive_cash_pnl"]
        total_negative = true_summary["negative_cash_pnl"] + false_summary["negative_cash_pnl"]
        advancement = {
            "state_positive_median": (
                true_summary["median_cash_pnl"] is not None
                and true_summary["median_cash_pnl"] > 0
            ),
            "state_better_win_rate": (
                true_summary["win_rate"] is not None
                and false_summary["win_rate"] is not None
                and true_summary["win_rate"] > false_summary["win_rate"]
            ),
            "retains_at_least_half_positive_pnl": (
                total_positive > 0
                and true_summary["positive_cash_pnl"] / total_positive >= 0.5
            ),
            "rejects_majority_negative_pnl": (
                total_negative < 0
                and false_summary["negative_cash_pnl"] / total_negative >= 0.5
            ),
        }
        output["account_manifest_sha256"][scope] = sha256(folder / "manifest.json")
        output["scopes"].append({
            "scope": scope,
            "state_true": true_summary,
            "state_false": false_summary,
            "advancement": advancement,
            "scope_pass": all(advancement.values()),
            "rows": rows,
        })
    output["advance_to_implementation"] = all(row["scope_pass"] for row in output["scopes"])
    output["decision"] = (
        "ADVANCE_TO_PREREGISTRATION" if output["advance_to_implementation"]
        else "REJECTED_BEFORE_IMPLEMENTATION"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

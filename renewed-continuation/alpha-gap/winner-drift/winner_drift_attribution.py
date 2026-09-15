from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from techquant.data import load_market
from research.expectation_study import scopes


ROOT = Path("/workspace/scratch/1816e11a0469")
SOURCE = ROOT / "quant-source"
INPUTS = ROOT / "frozen-input/evidence/inputs"
RUNS = ROOT / "coherent-evidence/payload/nonlinear/evaluation/runs"
CUTOFF = pd.Timestamp("2025-12-31")


def account_stats(folder: Path, market):
    orders = pd.read_csv(folder / "orders.csv")
    orders["date"] = pd.to_datetime(orders["date"])
    orders = orders[(orders["date"] <= CUTOFF) & orders.status.eq("FILLED")].copy()
    equity = pd.read_csv(folder / "equity.csv")
    equity["date"] = pd.to_datetime(equity["date"])
    equity = equity[equity.date <= CUTOFF].set_index("date")
    dates = market.calendar[market.calendar <= CUTOFF]
    close = market.panel("close").ffill().reindex(dates)
    units = pd.DataFrame(0.0, index=dates, columns=market.symbols)
    running = np.zeros(len(market.symbols))
    index = {symbol: j for j, symbol in enumerate(market.symbols)}
    by_date = {date: rows for date, rows in orders.groupby("date")}
    for date in dates:
        if date in by_date:
            for row in by_date[date].itertuples():
                running[index[row.symbol]] += row.units * (1 if row.side == "BUY" else -1)
        units.loc[date] = running
    values = units * close
    weights = values.div(equity.nav.reindex(dates), axis=0).fillna(0.0)
    result = {}
    for symbol in market.symbols:
        selected = orders[orders.symbol.eq(symbol)]
        buys = selected[selected.side.eq("BUY")]
        sells = selected[selected.side.eq("SELL")]
        cash_flow = (
            (sells.notional - sells.fee).sum()
            - (buys.notional + buys.fee).sum()
        )
        terminal = float(np.nan_to_num(values[symbol].iloc[-1], nan=0.0))
        result[symbol] = {
            "pnl": float(cash_flow + terminal),
            "buy_notional": float(buys.notional.sum()),
            "first_buy": str(buys.date.min().date()) if len(buys) else None,
            "last_buy": str(buys.date.max().date()) if len(buys) else None,
            "ownership_sessions": int((units[symbol] > 1e-10).sum()),
            "average_weight": float(weights[symbol].mean()),
            "terminal_value": terminal,
        }
    return result


def finite(value):
    value = float(value)
    return value if np.isfinite(value) else None


def main():
    catalog = json.loads((SOURCE / "research/catalog.json").read_text())
    full = load_market(INPUTS / "market", supplement=INPUTS / "supplement", sectors=catalog["sectors"])
    market = full.prefix(CUTOFF)
    output = {
        "status": "READ_ONLY_CROSS_SECTIONAL_ATTRIBUTION_NOT_CANDIDATE",
        "selection_end": str(CUTOFF.date()),
        "data_sha256": market.fingerprint(),
        "account_source": "previously authenticated coherent-evidence accounts",
        "scopes": [],
    }
    for scope, names in scopes(market, catalog).items():
        scoped = market.subset(names)
        passive = account_stats(RUNS / f"{scope}_buy_hold", scoped)
        quant = account_stats(RUNS / f"{scope}_incumbent", scoped)
        rows = []
        for symbol in scoped.symbols:
            p, q = passive[symbol], quant[symbol]
            lag = None
            if p["first_buy"] and q["first_buy"]:
                lag = int(
                    scoped.calendar.searchsorted(pd.Timestamp(q["first_buy"]))
                    - scoped.calendar.searchsorted(pd.Timestamp(p["first_buy"]))
                )
            rows.append({
                "symbol": symbol,
                "sector": catalog["sectors"][symbol],
                "passive_pnl": finite(p["pnl"]),
                "quant_pnl": finite(q["pnl"]),
                "pnl_gap": finite(p["pnl"] - q["pnl"]),
                "passive_first_buy": p["first_buy"],
                "quant_first_buy": q["first_buy"],
                "quant_entry_lag_sessions": lag,
                "passive_average_weight": finite(p["average_weight"]),
                "quant_average_weight": finite(q["average_weight"]),
                "passive_ownership_sessions": p["ownership_sessions"],
                "quant_ownership_sessions": q["ownership_sessions"],
            })
        rows.sort(key=lambda row: (-row["passive_pnl"], row["symbol"]))
        positive = [row for row in rows if row["passive_pnl"] > 0]
        top = rows[:5]
        total_positive = sum(row["passive_pnl"] for row in positive)
        output["scopes"].append({
            "scope": scope,
            "summary": {
                "symbols": len(rows),
                "passive_positive_contributors": len(positive),
                "passive_total_pnl": sum(row["passive_pnl"] for row in rows),
                "quant_total_symbol_pnl": sum(row["quant_pnl"] for row in rows),
                "quant_negative_symbol_pnl": sum(
                    row["quant_pnl"] for row in rows if row["quant_pnl"] < 0
                ),
                "positive_passive_names_negative_in_quant": sum(
                    row["passive_pnl"] > 0 and row["quant_pnl"] < 0 for row in rows
                ),
                "top5_share_of_positive_passive_pnl": (
                    sum(max(0.0, row["passive_pnl"]) for row in top) / total_positive
                    if total_positive > 0 else None
                ),
                "top5_quant_never_bought": sum(row["quant_first_buy"] is None for row in top),
                "top5_median_entry_lag": float(np.median([
                    row["quant_entry_lag_sessions"] for row in top
                    if row["quant_entry_lag_sessions"] is not None
                ])) if any(row["quant_entry_lag_sessions"] is not None for row in top) else None,
                "top5_passive_pnl": sum(row["passive_pnl"] for row in top),
                "top5_quant_pnl": sum(row["quant_pnl"] for row in top),
            },
            "top_passive_contributors": top,
            "rows": rows,
        })
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

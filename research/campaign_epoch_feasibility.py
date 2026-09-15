"""Read-only feasibility screen for a standalone campaign-epoch owner.

This is not a strategy replay.  It asks whether one fixed, causal market-state
definition produces sparse epochs whose next-open, buy-once/hold-to-epoch-end
cohorts have positive economics in every core scope.  It never consumes a
reference strategy, changes the production owner, or searches parameters.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from techquant.config import Config
from techquant.data import file_hash, load_market
from research.expectation_study import scopes


SELECTION_END = "2025-12-31"


def _finite(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _signals(market):
    quoted = market.panel("close")
    volume = market.panel("volume")
    active = quoted.notna() & volume.gt(0)
    close = quoted.ffill()
    observed = active.cumsum()
    ready = active & observed.ge(60) & active.rolling(20, min_periods=20).mean().ge(0.8)

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema60 = close.ewm(span=60, adjust=False).mean()
    return60 = close.pct_change(60, fill_method=None)
    qualified = ready & close.gt(ema20) & ema20.gt(ema60) & return60.gt(0)
    breadth = qualified.sum(axis=1).div(ready.sum(axis=1).replace(0, np.nan))

    # The supplied-universe equal-weight market record is causal.  Missing or
    # stale quotes cannot create a return observation.
    returns = close.pct_change(fill_method=None).where(active)
    market_index = (1.0 + returns.mean(axis=1, skipna=True).fillna(0.0)).cumprod()
    market_fast = market_index.ewm(span=20, adjust=False).mean()
    market_slow = market_index.ewm(span=60, adjust=False).mean()
    median60 = return60.where(ready).median(axis=1)

    launch = (
        breadth.ge(0.5) & market_fast.gt(market_slow) & median60.gt(0)
    ).fillna(False)
    finish = (
        breadth.lt(0.5) & market_fast.lt(market_slow) & median60.le(0)
    ).fillna(False)
    return close, active, qualified, breadth, market_index, launch, finish


def _epochs(launch, finish):
    starts, ends = [], []
    live = False
    for session in range(len(launch)):
        if not live and bool(launch.iloc[session]):
            starts.append(session)
            live = True
        elif live and bool(finish.iloc[session]):
            ends.append(session)
            live = False
    if len(ends) < len(starts):
        ends.append(None)
    return list(zip(starts, ends))


def _scope(market, config):
    close, active, qualified, breadth, market_index, signals, finishes = _signals(market)
    opened = market.panel("open")
    raw_open = market.panel("raw_open")
    volume = market.panel("volume")
    one_way_cost = (config.commission_bps + config.slippage_bps) / 10_000.0
    rows = []
    for signal_session, finish_session in _epochs(signals, finishes):
        entry_session = signal_session + 1
        if entry_session >= len(close):
            continue
        executable = opened.iloc[entry_session].notna() & volume.iloc[entry_session].gt(0)
        members = np.flatnonzero(qualified.iloc[signal_session].to_numpy() & executable.to_numpy())
        if not len(members):
            continue

        settled = finish_session is not None and finish_session + 1 < len(close)
        exit_session = finish_session + 1 if settled else len(close) - 1
        exit_price = opened.iloc[exit_session] if settled else close.iloc[exit_session]
        entry_price = opened.iloc[entry_session]
        gross = exit_price.iloc[members].to_numpy() / entry_price.iloc[members].to_numpy()
        net_factor = float(np.mean(gross) * (1.0 - one_way_cost) / (1.0 + one_way_cost))

        per_name_notional = config.initial_cash / len(members)
        capacity = (
            raw_open.iloc[entry_session, members].to_numpy()
            * volume.iloc[entry_session, members].to_numpy()
            * config.max_adv
        )
        utilization = per_name_notional / capacity
        rows.append({
            "launch_signal": str(market.calendar[signal_session].date()),
            "entry_open": str(market.calendar[entry_session].date()),
            "finish_signal": (
                str(market.calendar[finish_session].date()) if finish_session is not None else None
            ),
            "exit_open_or_mark": str(market.calendar[exit_session].date()),
            "settled": settled,
            "duration_sessions": int(exit_session - entry_session),
            "launch_breadth": _finite(breadth.iloc[signal_session]),
            "members": int(len(members)),
            "net_return": net_factor - 1.0,
            "positive_member_fraction": float(np.mean(gross > 1.0)),
            "max_entry_adv_utilization": _finite(np.max(utilization)),
        })

    settled = [row for row in rows if row["settled"]]
    settled_returns = np.asarray([row["net_return"] for row in settled], dtype=float)
    compound = float(np.prod(1.0 + settled_returns)) if len(settled_returns) else 1.0
    summary = {
        "launches": len(rows),
        "settled_epochs": len(settled),
        "censored_epochs": len(rows) - len(settled),
        "settled_median_net_return": _finite(np.median(settled_returns)) if len(settled_returns) else None,
        "settled_positive_fraction": _finite(np.mean(settled_returns > 0)) if len(settled_returns) else None,
        "settled_compound_factor": compound,
        "market_index_factor": _finite(market_index.iloc[-1] / market_index.iloc[0]),
    }
    summary["scope_pass"] = bool(
        len(settled) >= 2
        and summary["settled_median_net_return"] > 0
        and summary["settled_positive_fraction"] >= 0.5
        and compound > 1.0
    )
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).parent
    catalog = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    full = load_market(args.data, supplement=args.supplement, sectors=catalog["sectors"])
    market = full.prefix(SELECTION_END)
    config = Config()
    results = []
    for name, symbols in scopes(market, catalog).items():
        summary, epochs = _scope(market.subset(symbols), config)
        results.append({"scope": name, "summary": summary, "epochs": epochs})

    advance = all(row["summary"]["scope_pass"] for row in results)
    output = {
        "diagnosis": "standalone causal campaign-epoch feasibility",
        "status": "FEASIBLE_FOR_PREREGISTRATION" if advance else "REJECTED_BEFORE_IMPLEMENTATION",
        "advance": advance,
        "selection_end": SELECTION_END,
        "data_sha256": market.fingerprint(),
        "source_sha256": file_hash(Path(__file__)),
        "config": asdict(config),
        "causal_rule": {
            "security_ready": "60 observations and at least 80% active in trailing 20 sessions",
            "security_qualified": "close > EMA20 > EMA60 and 60-session return > 0",
            "launch": "qualified breadth >= 1/2, equal-weight market EMA20 > EMA60, median 60-session return > 0",
            "finish": "qualified breadth < 1/2, equal-weight market EMA20 < EMA60, median 60-session return <= 0",
            "ownership": "one equal-notional cohort at next open; no membership change until next-open finish",
            "cost": "unchanged commission plus slippage on entry and exit",
        },
        "advance_rule": "every scope has at least two settled epochs, positive median settled net return, at least half positive, and settled compound factor above one",
        "limitations": [
            "read-only cohort opportunity diagnostic, not an executable account or economic acceptance",
            "the final open epoch is censored and excluded from the advance rule",
            "adjusted economic units are not verified actual-share corporate-action accounting",
            "all observed history is retrospective",
        ],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

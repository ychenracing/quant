"""Read-only bound for quarantining cash after an owned campaign fails.

This does not alter or replay an executable portfolio.  It observes the
unchanged two-position price book, identifies the first registered
acquisition-basis/EMA40 failure in each actually funded campaign, and asks a
single local question: if the existing units were sold at the next open and
the proceeds remained idle until that unchanged campaign actually ended,
would more cash have been preserved than by continuing the campaign?

The calculation deliberately excludes right-censored campaigns from the
decision aggregate and reports next-open liquidity/limit feasibility rather
than pretending an infeasible full exit could fill.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from techquant.config import Config
from techquant.data import file_hash, load_market
from techquant.engine import run
from techquant.evidence import metrics
from techquant.execution import daily_limit, fee
from techquant.policy import CloseObservation
from research.coherent import SignalInputs
from research.expectation_study import scopes
from research.trend_book import Owner as ParentOwner, Parameters as ParentParameters


EPSILON = 1e-8
CUTOFF = "2025-12-31"


class Observer(ParentOwner):
    """Observe first basis failure per funded campaign without changing it."""

    def __init__(self, market):
        super().__init__(market, ParentParameters(2))
        close = market.panel("close").ffill()
        self.ema40 = close.ewm(
            span=Config().slow, adjust=False, min_periods=Config().slow
        ).mean().to_numpy()
        self.open = market.panel("open").to_numpy()
        size = len(market.symbols)
        self.acquisition_basis = np.zeros(size)
        self.observed_units = np.zeros(size)
        self.failure_streak = np.zeros(size, dtype=int)
        self.campaign_latched = np.zeros(size, dtype=bool)
        self.events: list[dict] = []

    def _observe_fills(self, observation: CloseObservation) -> None:
        units = observation.units
        added = units > self.observed_units + 1e-10
        opened = added & (self.observed_units <= 1e-10)
        prices = self.open[observation.session]
        if np.any(added & (~np.isfinite(prices) | (prices <= 0))):
            raise ValueError("observed additions require a valid current open")
        self.acquisition_basis[opened] = prices[opened]
        enlarged = added & ~opened
        delta = units - self.observed_units
        self.acquisition_basis[enlarged] = (
            self.observed_units[enlarged] * self.acquisition_basis[enlarged]
            + delta[enlarged] * prices[enlarged]
        ) / units[enlarged]
        flat = units <= 1e-10
        self.acquisition_basis[flat] = 0.0
        self.failure_streak[flat] = 0
        self.campaign_latched[flat] = False
        self.observed_units = units.copy()

    def _signal_inputs(self, session: int) -> SignalInputs:
        observed = super()._signal_inputs(session)
        held = self.observed_units > 1e-10
        price = self.price_signals.price[session]
        failing = (
            held
            & (self.acquisition_basis > 0)
            & np.isfinite(price)
            & np.isfinite(self.ema40[session])
            & (price < self.acquisition_basis)
            & (price < self.ema40[session])
        )
        self.failure_streak = np.where(failing, self.failure_streak + 1, 0)
        new_event = held & (self.failure_streak == 2) & ~self.campaign_latched
        for index in np.flatnonzero(new_event):
            self.events.append({
                "signal_date": str(self.market.calendar[session].date()),
                "signal_session": int(session),
                "symbol": self.market.symbols[index],
                "signal_close": float(price[index]),
                "ema40": float(self.ema40[session, index]),
                "acquisition_basis": float(self.acquisition_basis[index]),
                "units": float(self.observed_units[index]),
            })
        self.campaign_latched |= new_event
        return observed

    def decide(self, observation: CloseObservation):
        self._observe_fills(observation)
        return super().decide(observation)


@dataclass(frozen=True)
class Campaign:
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp | None
    start_order: int
    end_order: int | None


def campaigns(orders: pd.DataFrame) -> list[Campaign]:
    found: list[Campaign] = []
    for symbol, selected in orders.groupby("symbol", sort=True):
        units = 0.0
        start = None
        entry = None
        rows = selected.sort_values(["date", "sequence"])
        for row in rows.itertuples():
            signed = row.units if row.side == "BUY" else -row.units
            if units <= EPSILON and signed > 0:
                start, entry = int(row.sequence), row.date
            units += signed
            if units < -EPSILON:
                raise AssertionError(f"negative inventory for {symbol} on {row.date}")
            if units <= EPSILON and start is not None:
                found.append(Campaign(symbol, entry, row.date, start, int(row.sequence)))
                units, start, entry = 0.0, None, None
        if start is not None:
            found.append(Campaign(symbol, entry, None, start, None))
    return found


def locate_campaign(event: dict, all_campaigns: list[Campaign]) -> Campaign:
    signal = pd.Timestamp(event["signal_date"])
    matches = [
        campaign for campaign in all_campaigns
        if campaign.symbol == event["symbol"]
        and campaign.entry_date <= signal
        and (campaign.exit_date is None or signal < campaign.exit_date)
    ]
    if len(matches) != 1:
        raise AssertionError(f"event-to-campaign mismatch: {event} {matches}")
    return matches[0]


def next_open_bound(market, result, event: dict, campaign: Campaign) -> dict:
    names = list(market.symbols)
    j = names.index(event["symbol"])
    signal_i = int(event["signal_session"])
    if signal_i + 1 >= len(market.calendar):
        return {"censored": True, "reason": "NO_NEXT_SESSION"}
    execution_i = signal_i + 1
    execution_date = market.calendar[execution_i]
    if campaign.exit_date is None:
        return {
            "censored": True,
            "reason": "ORIGINAL_CAMPAIGN_OPEN_AT_CUTOFF",
            "execution_date": str(execution_date.date()),
        }

    op = market.panel("open").to_numpy()
    rop = market.panel("raw_open").to_numpy()
    close = market.panel("close").ffill().to_numpy()
    raw_close = market.panel("raw_close").to_numpy()
    volume = market.panel("volume").to_numpy()
    adjusted_open = float(op[execution_i, j])
    raw_open = float(rop[execution_i, j])
    previous_close = float(close[execution_i - 1, j])
    gap = adjusted_open / previous_close - 1 if previous_close > 0 else 0.0
    limit_blocked = (
        not np.isfinite(adjusted_open)
        or not np.isfinite(raw_open)
        or gap <= -(daily_limit(event["symbol"]) - 0.002)
    )
    prior_turnover = volume[:, j] * raw_close[:, j]
    capacity = float(pd.Series(prior_turnover).rolling(20, min_periods=1).mean().shift().iloc[execution_i])
    capacity *= Config().max_adv
    raw_quantity = float(event["units"] * adjusted_open / raw_open) if raw_open > 0 else float("inf")
    raw_notional = raw_quantity * raw_open
    full_capacity = bool(np.isfinite(capacity) and capacity > 0 and raw_notional <= capacity + 1e-8)
    executable_next_open = bool(not limit_blocked and full_capacity)

    slipped = adjusted_open * (1 - Config().slippage_bps / 10_000)
    hypothetical_notional = float(event["units"] * slipped)
    hypothetical_fee = float(fee(
        hypothetical_notional,
        "SELL",
        str(execution_date.date()),
        Config().commission_bps,
    ))
    hypothetical_cash = hypothetical_notional - hypothetical_fee

    filled = pd.DataFrame([order for order in result.orders if order["status"] == "FILLED"])
    filled["date"] = pd.to_datetime(filled["date"])
    future = filled[
        filled.symbol.eq(event["symbol"])
        & filled.date.ge(execution_date)
        & filled.date.le(campaign.exit_date)
    ]
    actual_future_cash = float(
        (future.loc[future.side.eq("SELL"), "notional"]
         - future.loc[future.side.eq("SELL"), "fee"]).sum()
        - (future.loc[future.side.eq("BUY"), "notional"]
           + future.loc[future.side.eq("BUY"), "fee"]).sum()
    )
    preserved_cash = hypothetical_cash - actual_future_cash
    execution_session = int(market.calendar.get_loc(execution_date))
    exit_session = int(market.calendar.get_loc(campaign.exit_date))
    quarantine_sessions = exit_session - execution_session + 1
    nav = float(result.equity.loc[pd.Timestamp(event["signal_date"]), "nav"])
    return {
        "censored": False,
        "execution_date": str(execution_date.date()),
        "original_campaign_exit": str(campaign.exit_date.date()),
        "quarantine_sessions": int(quarantine_sessions),
        "event_units": float(event["units"]),
        "next_open": adjusted_open,
        "next_open_gap": float(gap),
        "prior_only_capacity_cny": capacity if np.isfinite(capacity) else None,
        "requested_raw_notional_cny": raw_notional,
        "capacity_utilization": raw_notional / capacity if capacity > 0 else None,
        "limit_blocked": bool(limit_blocked),
        "full_capacity": full_capacity,
        "executable_next_open": executable_next_open,
        "hypothetical_sale_notional_cny": hypothetical_notional,
        "hypothetical_sale_fee_cny": hypothetical_fee,
        "quarantined_cash_cny": hypothetical_cash,
        "quarantined_cash_share_of_signal_nav": hypothetical_cash / nav,
        "actual_future_campaign_cashflow_cny": actual_future_cash,
        "preserved_cash_at_original_exit_cny": preserved_cash,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("."))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    catalog = json.loads((args.source / "research/catalog.json").read_text())
    full = load_market(args.data, supplement=args.supplement, sectors=catalog["sectors"])
    market = full.prefix(CUTOFF)
    output = {
        "status": "READ_ONLY_NON_RECYCLED_CAPITAL_BOUND_NOT_CANDIDATE",
        "decision_rule": (
            "Observe the first owned basis/EMA40 failure in each unchanged control "
            "campaign; sell at the next open and hold the proceeds idle until that "
            "same campaign's unchanged exit. Exclude open campaigns from the decision."
        ),
        "interpretation": (
            "Local campaign cash-preservation bound only. It does not replay portfolio "
            "NAV, create spendable cash, or establish an executable strategy."
        ),
        "selection_end": CUTOFF,
        "full_data_sha256": full.fingerprint(),
        "selection_data_sha256": market.fingerprint(),
        "source_sha256": file_hash(Path(__file__)),
        "scopes": [],
    }
    for scope, symbols in scopes(market, catalog).items():
        scoped = market.subset(symbols)
        observer = Observer(scoped)
        result = run(scoped, Config(), policy_factory=lambda _market, _config: observer)
        filled = pd.DataFrame([order for order in result.orders if order["status"] == "FILLED"])
        filled["date"] = pd.to_datetime(filled["date"])
        filled["sequence"] = np.arange(len(filled))
        all_campaigns = campaigns(filled)
        rows = []
        for event in observer.events:
            campaign = locate_campaign(event, all_campaigns)
            rows.append({**event, **next_open_bound(scoped, result, event, campaign)})
        settled = [row for row in rows if not row["censored"]]
        executable = [row for row in settled if row["executable_next_open"]]
        preserved = sum(row["preserved_cash_at_original_exit_cny"] for row in settled)
        executable_preserved = sum(row["preserved_cash_at_original_exit_cny"] for row in executable)
        positive = sum(max(0.0, row["preserved_cash_at_original_exit_cny"]) for row in settled)
        sacrificed = -sum(min(0.0, row["preserved_cash_at_original_exit_cny"]) for row in settled)
        account_metrics = dict(metrics(result), average_exposure=float(result.equity.exposure.mean()))
        output["scopes"].append({
            "scope": scope,
            "control": account_metrics,
            "events": len(rows),
            "settled_events": len(settled),
            "censored_events": len(rows) - len(settled),
            "next_open_executable_events": len(executable),
            "preserved_cash_cny": preserved,
            "executable_subset_preserved_cash_cny": executable_preserved,
            "positive_preservation_cny": positive,
            "sacrificed_continuation_profit_cny": sacrificed,
            "median_quarantine_sessions": (
                float(np.median([row["quarantine_sessions"] for row in settled]))
                if settled else None
            ),
            "quarantined_cash_session_cny": sum(
                row["quarantined_cash_cny"] * row["quarantine_sessions"] for row in settled
            ),
            "all_settled_events_preserve_cash": bool(
                settled and all(row["preserved_cash_at_original_exit_cny"] >= 0 for row in settled)
            ),
            "rows": rows,
        })
    output["advance_to_preregistration"] = all(
        row["settled_events"] > 0
        and row["preserved_cash_cny"] > 0
        and row["next_open_executable_events"] == row["settled_events"]
        for row in output["scopes"]
    )
    output["decision"] = (
        "ADVANCE_TO_PREREGISTRATION"
        if output["advance_to_preregistration"]
        else "REJECTED_BEFORE_IMPLEMENTATION"
    )
    rendered = json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

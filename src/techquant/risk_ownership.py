"""Causal risk-aware ownership over the shared fill-aware execution engine.

The engine owns the original per-security cash sleeves, actual inventory, fees,
capacity and next-session execution.  This policy can only retain, protect or
restore already acquired economic units, and may reopen untouched sleeves only
after every unresolved risk event has obtained fresh recovery evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .data import Market, file_hash
from .engine import Result, run
from .policy import CloseDecision, CloseObservation


@dataclass(frozen=True)
class RiskOwnershipParameters:
    """Small preregistered structural surface; no per-case parameters."""

    core_fraction: float = 0.60
    risk_confirmation: int = 2
    recovery_confirmation: int = 2

    def __post_init__(self) -> None:
        if self.core_fraction not in (0.50, 0.60, 0.70):
            raise ValueError("core_fraction must be one of 0.50, 0.60 or 0.70")
        for name in ("risk_confirmation", "recovery_confirmation"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
                raise ValueError(f"{name} must be an integer from one to five")


class RiskAwareOwnershipPolicy:
    """Long-horizon ownership with event-latched selective protection."""

    def __init__(
        self,
        market: Market,
        config: Config,
        parameters: RiskOwnershipParameters | None = None,
    ):
        self.market = market
        self.config = config
        self.parameters = parameters or RiskOwnershipParameters()
        self._build_signals(market)

        size = len(market.symbols)
        self.state = "OPEN"
        self._last_session = -1
        self._risk_streak = 0
        self._healthy_streak = 0
        self._recovery_stage = 0
        self._last_protection_cap = 1.0
        self._peak_nav = 0.0
        self._nav_history: list[float] = []

        # Absolute-unit memories.  A blocked or partial fill therefore retries
        # one frozen goal instead of repeatedly cutting the surviving units.
        self.event_base_units = np.zeros(size)
        self.protection_goal = np.zeros(size)
        self.event_price = np.zeros(size)
        self.recovery_closes = np.zeros(size, dtype=int)
        self.requested_stage = np.zeros(size, dtype=int)
        self.last_target = np.zeros(size)

    def _build_signals(self, market: Market) -> None:
        quoted = market.panel("close")
        volume = market.panel("volume")
        active = quoted.notna() & volume.gt(0)
        price = quoted.ffill()
        returns = price.pct_change(fill_method=None).where(
            active & active.shift(fill_value=False)
        )
        market_return = returns.mean(axis=1).fillna(0.0)
        index = (1.0 + market_return).cumprod()
        index_ema20 = index.ewm(span=20, adjust=False).mean()
        index_ema60 = index.ewm(span=60, adjust=False).mean()

        ema20 = price.ewm(span=20, adjust=False).mean()
        ema60 = price.ewm(span=60, adjust=False).mean()
        ready20 = active & active.cumsum().ge(20)
        ready60 = active & active.cumsum().ge(60)
        breadth20 = (
            (ready20 & price.gt(ema20)).sum(axis=1)
            .div(ready20.sum(axis=1).replace(0, np.nan))
            .fillna(0.0)
        )
        breadth60 = (
            (ready60 & price.gt(ema60)).sum(axis=1)
            .div(ready60.sum(axis=1).replace(0, np.nan))
            .fillna(0.0)
        )
        observed = returns.notna().sum(axis=1).replace(0, np.nan)
        loss_fraction = returns.le(-0.04).sum(axis=1).div(observed).fillna(0.0)
        negative = market_return.clip(upper=0.0)
        downside_short = negative.pow(2).rolling(5, min_periods=3).mean().pow(0.5)
        downside_long = negative.pow(2).rolling(40, min_periods=20).mean().pow(0.5)

        index_ret3 = index.pct_change(3, fill_method=None).fillna(0.0)
        index_ret5 = index.pct_change(5, fill_method=None).fillna(0.0)
        index_ret10 = index.pct_change(10, fill_method=None).fillna(0.0)
        breadth_change5 = breadth20 - breadth20.shift(5).fillna(breadth20)

        self.price = price.to_numpy(dtype=float)
        self.ema20 = ema20.to_numpy(dtype=float)
        self.ema60 = ema60.to_numpy(dtype=float)
        self.ret5 = price.pct_change(5, fill_method=None).to_numpy(dtype=float)
        self.fresh = active.to_numpy(dtype=bool)
        self.trend_damage = (
            index.lt(index_ema60)
            & index_ema20.lt(index_ema60)
            & breadth60.lt(0.45)
        ).to_numpy(dtype=bool)
        self.breadth_damage = (
            breadth20.lt(0.35) & breadth_change5.le(-0.15)
        ).to_numpy(dtype=bool)
        self.volatility_damage = (
            downside_short.gt(np.maximum(0.012, 1.8 * downside_long))
            & index_ret5.lt(0.0)
        ).fillna(False).to_numpy(dtype=bool)
        self.market_shock = (
            index_ret3.le(-0.08)
            | index_ret10.le(-0.15)
            | (loss_fraction.ge(0.55) & market_return.le(-0.025))
        ).to_numpy(dtype=bool)
        self.market_healthy = (
            index.gt(index_ema20)
            & index_ema20.ge(index_ema60)
            & breadth20.ge(0.55)
            & index_ret5.ge(0.0)
            & loss_fraction.lt(0.20)
        ).to_numpy(dtype=bool)
        self.acute_damage = (
            price.pct_change(3, fill_method=None).le(-0.15)
            & price.lt(ema20)
            & price.lt(ema60)
            & price.pct_change(20, fill_method=None).le(-0.12)
            & ready20
        ).fillna(False).to_numpy(dtype=bool)
        strength = (
            price.div(ema60).sub(1.0)
            + 0.5 * ema20.div(ema60).sub(1.0)
            + 0.25 * price.pct_change(20, fill_method=None)
            - 0.5 * returns.rolling(20, min_periods=10).std()
        )
        self.strength = (
            strength.replace([np.inf, -np.inf], np.nan)
            .fillna(-np.inf)
            .to_numpy(dtype=float)
        )

    def _state_cap(self) -> float:
        core = self.parameters.core_fraction
        if self.state == "CRISIS":
            return core
        if self.state == "DEFENSIVE":
            return (1.0 + core) / 2.0
        if self.state == "RECOVERY":
            return core + (1.0 - core) / 2.0
        return 1.0

    def _advance_state(self, close: CloseObservation, reasons: list[str]) -> str:
        i = close.session
        previous = self.state
        self._peak_nav = max(self._peak_nav, close.nav)
        drawdown = 1.0 - close.nav / self._peak_nav if self._peak_nav else 0.0
        day_loss = close.nav / self._nav_history[-1] - 1.0 if self._nav_history else 0.0
        loss5 = (
            close.nav / self._nav_history[-5] - 1.0
            if len(self._nav_history) >= 5
            else 0.0
        )
        self._nav_history.append(close.nav)

        account_damage = drawdown >= 0.10 and (day_loss <= -0.03 or loss5 <= -0.05)
        channels = {
            "TREND": bool(self.trend_damage[i]),
            "BREADTH": bool(self.breadth_damage[i]),
            "VOLATILITY": bool(self.volatility_damage[i]),
            "ACCOUNT_PATH": bool(account_damage),
        }
        active = [name for name, enabled in channels.items() if enabled]
        crisis = bool(
            drawdown >= 0.18
            or (
                self.market_shock[i]
                and (channels["TREND"] or channels["BREADTH"] or account_damage)
            )
        )
        defensive = len(active) >= 2
        caution = bool(active or self.market_shock[i])

        if crisis:
            self.state = "CRISIS"
            self._risk_streak = 0
            self._healthy_streak = 0
            self._recovery_stage = 0
        elif defensive:
            self._risk_streak += 1
            self._healthy_streak = 0
            if previous in ("DEFENSIVE", "CRISIS"):
                self.state = previous
            elif previous == "RECOVERY":
                self.state = "DEFENSIVE"
                self._recovery_stage = 0
            elif self._risk_streak >= self.parameters.risk_confirmation:
                self.state = "DEFENSIVE"
            else:
                self.state = "CAUTION"
        elif caution:
            self._risk_streak = 0
            self._healthy_streak = 0
            if previous in ("DEFENSIVE", "CRISIS", "RECOVERY"):
                self.state = "DEFENSIVE"
                self._recovery_stage = 0
            else:
                self.state = "CAUTION"
        else:
            self._risk_streak = 0
            healthy = bool(self.market_healthy[i])
            self._healthy_streak = self._healthy_streak + 1 if healthy else 0
            if previous in ("DEFENSIVE", "CRISIS"):
                if self._healthy_streak >= self.parameters.recovery_confirmation:
                    self.state = "RECOVERY"
                    self._recovery_stage = 1
                    self._healthy_streak = 0
            elif previous == "RECOVERY":
                if self._healthy_streak >= self.parameters.recovery_confirmation:
                    self.state = "OPEN"
                    self._recovery_stage = 2
                    self._healthy_streak = 0
            elif previous == "CAUTION":
                if self._healthy_streak >= self.parameters.recovery_confirmation:
                    self.state = "OPEN"
                    self._healthy_streak = 0
            else:
                self.state = "OPEN"

        trigger = ",".join(
            (["SHOCK"] if self.market_shock[i] else []) + active
        ) or "NONE"
        if self.state != previous:
            reasons.append(f"STATE:{previous}->{self.state}")
        reasons.append(f"RISK_CHANNELS:{trigger}")
        return previous

    def _selective_goal(self, close: CloseObservation, cap: float) -> np.ndarray:
        current = close.units.copy()
        marks = np.where(
            np.isfinite(self.price[close.session]),
            self.price[close.session],
            0.0,
        )
        values = current * marks
        target_value = min(float(values.sum()), cap * close.nav)
        if values.sum() <= target_value + 1e-8:
            return current

        goal = current * self.parameters.core_fraction
        goal_value = goal * marks
        if goal_value.sum() > target_value and goal_value.sum() > 0:
            goal *= target_value / goal_value.sum()
            return goal

        remaining = target_value - float(goal_value.sum())
        order = sorted(
            np.flatnonzero(current > 1e-10),
            key=lambda j: (-self.strength[close.session, j], self.market.symbols[j]),
        )
        for j in order:
            room_value = (current[j] - goal[j]) * marks[j]
            if room_value <= 0:
                continue
            add_value = min(remaining, room_value)
            goal[j] += add_value / marks[j]
            remaining -= add_value
            if remaining <= 1e-8:
                break
        return goal

    def _record_event(
        self,
        mask: np.ndarray,
        proposed_goal: np.ndarray,
        close: CloseObservation,
        reasons: list[str],
        label: str,
        *,
        reset_existing: bool,
    ) -> None:
        if not mask.any():
            return
        existing = self.event_price > 0
        new = mask & ~existing
        old = mask & existing
        self.event_base_units[new] = close.units[new]
        self.protection_goal[new] = proposed_goal[new]
        if reset_existing:
            self.protection_goal[old] = np.minimum(
                self.protection_goal[old], proposed_goal[old]
            )
        affected = new | (old & reset_existing)
        self.event_price[affected] = np.maximum(
            self.event_price[affected],
            np.nan_to_num(self.price[close.session, affected], nan=0.0),
        )
        self.recovery_closes[affected] = 0
        self.requested_stage[affected] = 0
        self.last_target[affected] = self.protection_goal[affected]
        reasons.append(
            label + ":" + ",".join(self.market.symbols[j] for j in np.flatnonzero(mask))
        )

    def _complete_funded_recovery(
        self,
        close: CloseObservation,
        reasons: list[str],
    ) -> None:
        active = self.event_price > 0
        tolerance = np.maximum(1e-8, self.last_target * 1e-10)
        complete = (
            active
            & (self.requested_stage >= 2)
            & (self.last_target > self.protection_goal + tolerance)
            & (close.units >= self.last_target - tolerance)
        )
        if not complete.any():
            return
        self.event_base_units[complete] = 0.0
        self.protection_goal[complete] = 0.0
        self.event_price[complete] = 0.0
        self.recovery_closes[complete] = 0
        self.requested_stage[complete] = 0
        self.last_target[complete] = close.units[complete]
        reasons.append(
            "FUNDED_RECOVERY_COMPLETE:"
            + ",".join(self.market.symbols[j] for j in np.flatnonzero(complete))
        )

    def _recovery_target(self, close: CloseObservation) -> np.ndarray:
        active = self.event_price > 0
        target = close.units.copy()
        target[active] = self.protection_goal[active]
        if not active.any():
            return target

        if self.state == "RECOVERY":
            global_stage = 1
        elif self.state == "OPEN":
            global_stage = 2
        else:
            global_stage = 0
        symbol_stage = np.minimum(
            2, self.recovery_closes // self.parameters.recovery_confirmation
        )
        stage = np.minimum(global_stage, symbol_stage)
        fraction = np.where(stage >= 2, 1.0, np.where(stage == 1, 0.5, 0.0))
        desired = (
            self.protection_goal
            + fraction * (self.event_base_units - self.protection_goal)
        )
        target[active] = desired[active]

        # Never invent funding.  Keep already-held units, execute any remaining
        # protection, then allocate only the close-time cash budget to recovery.
        marks = np.where(
            np.isfinite(self.price[close.session]),
            self.price[close.session],
            0.0,
        )
        minimum = np.minimum(target, close.units)
        additions = np.maximum(0.0, target - minimum)
        minimum_value = float(minimum @ marks)
        available_value = max(0.0, close.nav - minimum_value)
        addition_value = additions * marks
        if addition_value.sum() > available_value and addition_value.sum() > 0:
            additions *= available_value / addition_value.sum()
        target = minimum + additions
        self.requested_stage[active] = stage[active]
        self.last_target[active] = target[active]
        return target

    def decide(self, close: CloseObservation) -> CloseDecision:
        if close.session <= self._last_session:
            raise ValueError("policy sessions must increase")
        self._last_session = close.session
        ownership = close.ownership
        if ownership is None:
            raise ValueError("risk-aware ownership requires engine-owned intent")
        if (
            not np.isfinite(close.nav)
            or close.nav <= 0
            or (close.units < 0).any()
            or (close.units > ownership.units + 1e-8).any()
        ):
            raise ValueError("invalid observed ownership account")

        reasons: list[str] = []
        self._complete_funded_recovery(close, reasons)
        active = self.event_price > 0
        symbol_healthy = (
            active
            & self.fresh[close.session]
            & (self.price[close.session] >= self.event_price)
            & (self.price[close.session] > self.ema20[close.session])
            & (np.nan_to_num(self.ret5[close.session], nan=-np.inf) > 0)
        )
        self.recovery_closes = np.where(
            active & symbol_healthy, self.recovery_closes + 1, 0
        )

        previous = self._advance_state(close, reasons)
        cap = self._state_cap()
        if self.state in ("DEFENSIVE", "CRISIS"):
            became_more_severe = (
                previous not in ("DEFENSIVE", "CRISIS")
                or cap < self._last_protection_cap - 1e-10
            )
            proposed = self._selective_goal(close, cap)
            tolerance = np.maximum(1e-8, close.units * 1e-10)
            reduced = proposed < close.units - tolerance
            if not became_more_severe:
                reduced &= self.event_price <= 0
            self._record_event(
                reduced,
                proposed,
                close,
                reasons,
                "SYSTEMIC_PROTECTION",
                reset_existing=became_more_severe,
            )
            if became_more_severe:
                self._last_protection_cap = cap

        acute = self.acute_damage[close.session] & (close.units > 1e-8)
        if acute.any():
            proposed = close.units.copy()
            proposed[acute] = 0.0
            self._record_event(
                acute,
                proposed,
                close,
                reasons,
                "ACUTE_HOLDING_DAMAGE",
                reset_existing=True,
            )

        active = self.event_price > 0
        target = self._recovery_target(close)
        marks = np.where(
            np.isfinite(self.price[close.session]),
            self.price[close.session],
            0.0,
        )
        weights = target * marks / close.nav
        if weights.sum() > 1 + 1e-10:
            raise ValueError("risk-aware target exceeds the observed account")

        allow_new = bool(self.state == "OPEN" and not active.any())
        decision_cap = 1.0 if allow_new else max(cap, float(weights.sum()))
        if active.any():
            reasons.append(
                "UNRESOLVED_RISK_MEMORY:"
                + ",".join(self.market.symbols[j] for j in np.flatnonzero(active))
            )
        if not reasons:
            reasons.append("OPEN_PASSIVE_OWNERSHIP")
        return CloseDecision(
            weights=weights,
            reason="|".join(reasons),
            cap=min(1.0, decision_cap),
            unit_targets=target,
            allow_new_ownership=allow_new,
        )

    def identity(self) -> dict[str, Any]:
        return {
            "name": "risk_aware_ownership",
            "parameters": asdict(self.parameters),
            "implementation_sha256": file_hash(Path(__file__)),
            "data_sha256": self.market.fingerprint(),
            "status": "RESEARCH_NOT_ACCEPTED",
        }


def run_risk_aware_ownership(
    market: Market,
    config: Config | None = None,
    *,
    parameters: RiskOwnershipParameters | None = None,
    start: str | None = None,
    end: str | None = None,
    delay: int = 1,
    cost_multiplier: float = 1.0,
) -> Result:
    """Run the isolated candidate; the production default remains passive."""
    chosen = parameters or RiskOwnershipParameters()

    def factory(bound_market: Market, bound_config: Config) -> RiskAwareOwnershipPolicy:
        return RiskAwareOwnershipPolicy(bound_market, bound_config, chosen)

    result = run(
        market,
        config,
        start=start,
        end=end,
        delay=delay,
        cost_multiplier=cost_multiplier,
        policy_factory=factory,
        ownership_mode=True,
    )
    result.metadata = dict(result.metadata)
    result.metadata["strategy"] = "risk_aware_ownership"
    result.metadata["economic_acceptance"] = "NOT_MEASURED"
    result.metadata["production_default_changed"] = False
    return result


__all__ = [
    "RiskAwareOwnershipPolicy",
    "RiskOwnershipParameters",
    "run_risk_aware_ownership",
]

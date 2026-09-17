"""Event-scoped systemic protection over engine-owned passive ownership.

The policy preserves the passive return source until a *new* systemic risk event
is confirmed.  One event requests one fixed, executable unit reduction.  A
historical account drawdown can strengthen a fresh loss signal, but can never
keep the account defensive by itself.  Recovery uses new market evidence in two
stages and never sells a restored unit again merely because the same episode
remains below its old account peak.
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
from .execution import round_quantity
from .policy import CloseDecision, CloseObservation


@dataclass(frozen=True)
class RiskOwnershipParameters:
    """Pre-registered coarse structure; no per-case or date-specific controls."""

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
    """Sparse systemic protection with fill-aware event memory."""

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
        self._early_recovery_streak = 0
        self._full_recovery_streak = 0
        self._crisis_active_last = False
        self._episode_level = 0
        self._recovery_stage = 0
        self._recovery_paused = False
        self._episode_base_units = np.zeros(size)
        self._protection_goal = np.zeros(size)
        self._peak_nav = 0.0
        self._nav_history: list[float] = []

    def _build_signals(self, market: Market) -> None:
        quoted = market.panel("close")
        raw_close = market.panel("raw_close")
        volume = market.panel("volume")
        active = quoted.notna() & volume.gt(0)
        price = quoted.ffill()
        raw = raw_close.ffill()
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
            (ready20 & price.gt(ema20))
            .sum(axis=1)
            .div(ready20.sum(axis=1).replace(0, np.nan))
            .fillna(0.0)
        )
        breadth60 = (
            (ready60 & price.gt(ema60))
            .sum(axis=1)
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
        ret20 = price.pct_change(20, fill_method=None)
        symbol_volatility = returns.rolling(20, min_periods=10).std()
        strength = (
            price.div(ema60).sub(1.0)
            + 0.5 * ema20.div(ema60).sub(1.0)
            + 0.25 * ret20
            - 0.5 * symbol_volatility
        )

        self.price = price.to_numpy(dtype=float)
        self.raw_close = raw.to_numpy(dtype=float)
        self.strength = (
            strength.replace([np.inf, -np.inf], np.nan)
            .fillna(-np.inf)
            .to_numpy(dtype=float)
        )
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
        self.recovery_early = (
            index.gt(index_ema20)
            & breadth20.ge(0.45)
            & breadth_change5.gt(0.0)
            & index_ret5.gt(0.0)
            & loss_fraction.lt(0.30)
        ).to_numpy(dtype=bool)
        self.recovery_full = (
            index.gt(index_ema20)
            & breadth20.ge(0.55)
            & index_ret10.gt(0.0)
            & loss_fraction.lt(0.20)
        ).to_numpy(dtype=bool)

    def _nominal_cap(self) -> float:
        if self._episode_level >= 2:
            return self.parameters.core_fraction
        if self._episode_level == 1:
            return (1.0 + self.parameters.core_fraction) / 2.0
        return 1.0

    def _account_acceleration(self, close: CloseObservation) -> tuple[bool, float]:
        self._peak_nav = max(self._peak_nav, close.nav)
        drawdown = 1.0 - close.nav / self._peak_nav if self._peak_nav else 0.0
        day_loss = (
            close.nav / self._nav_history[-1] - 1.0
            if self._nav_history
            else 0.0
        )
        loss5 = (
            close.nav / self._nav_history[-5] - 1.0
            if len(self._nav_history) >= 5
            else 0.0
        )
        self._nav_history.append(close.nav)
        # Absolute drawdown is context only. A fresh loss acceleration is
        # required, so an old peak cannot create a permanent defensive state.
        accelerated = drawdown >= 0.10 and (day_loss <= -0.04 or loss5 <= -0.08)
        return bool(accelerated), float(drawdown)

    def _risk_snapshot(
        self, close: CloseObservation
    ) -> tuple[list[str], bool, bool, float]:
        i = close.session
        account, drawdown = self._account_acceleration(close)
        channels = {
            "TREND": bool(self.trend_damage[i]),
            "BREADTH": bool(self.breadth_damage[i]),
            "VOLATILITY": bool(self.volatility_damage[i]),
            "ACCOUNT_ACCELERATION": account,
        }
        active = [name for name, enabled in channels.items() if enabled]
        defensive = len(active) >= 2
        crisis = bool(self.market_shock[i] and len(active) >= 1)
        return active, defensive, crisis, drawdown

    def _quantize_toward(
        self, current: np.ndarray, desired: np.ndarray, session: int
    ) -> np.ndarray:
        """Request only an executable lot delta; a sub-lot residual is held."""

        result = np.asarray(current, dtype=float).copy()
        desired = np.asarray(desired, dtype=float)
        adjusted = self.price[session]
        raw = self.raw_close[session]
        for j, symbol in enumerate(self.market.symbols):
            delta = float(desired[j] - current[j])
            if abs(delta) <= 1e-10:
                continue
            if (
                not np.isfinite(adjusted[j])
                or not np.isfinite(raw[j])
                or adjusted[j] <= 0
                or raw[j] <= 0
            ):
                continue
            if desired[j] <= 1e-10 and delta < 0:
                # The shared engine explicitly supports full odd-lot disposal.
                result[j] = 0.0
                continue
            raw_delta = abs(delta) * adjusted[j] / raw[j]
            executable_raw = round_quantity(symbol, raw_delta)
            if executable_raw <= 0:
                continue
            executable_units = executable_raw * raw[j] / adjusted[j]
            change = min(abs(delta), executable_units)
            result[j] = current[j] + (change if delta > 0 else -change)
        return result

    def _executable_goal_reached(
        self, current: np.ndarray, desired: np.ndarray, session: int
    ) -> bool:
        quantized = self._quantize_toward(current, desired, session)
        return bool(np.allclose(quantized, current, rtol=0.0, atol=1e-8))

    def _cash_funded_desired(
        self, close: CloseObservation, desired: np.ndarray
    ) -> np.ndarray:
        """Bound restoration by close-time account value before execution costs."""

        marks = np.where(
            np.isfinite(self.price[close.session]),
            self.price[close.session],
            0.0,
        )
        desired = np.asarray(desired, dtype=float)
        minimum = np.minimum(desired, close.units)
        additions = np.maximum(0.0, desired - minimum)
        minimum_value = float(minimum @ marks)
        available_value = max(0.0, close.nav - minimum_value)
        addition_value = additions * marks
        total_addition = float(addition_value.sum())
        if total_addition > available_value and total_addition > 0:
            additions *= available_value / total_addition
        return minimum + additions

    def _protection_target(
        self, close: CloseObservation, nominal_cap: float
    ) -> np.ndarray:
        """Keep causally strong holdings first at the same total risk cap.

        The event-level cap is identical to proportional protection.  Only the
        cross-sectional allocation changes: intact leadership is retained while
        the weakest observed holdings fund protection.  No future return, known
        date or case identity participates in this ordering.
        """

        marks = np.where(
            np.isfinite(self.price[close.session]),
            self.price[close.session],
            0.0,
        )
        current_value = float(close.units @ marks)
        target_value = min(current_value, nominal_cap * close.nav)
        if current_value <= target_value + 1e-8 or current_value <= 0:
            return close.units.copy()

        desired = np.zeros_like(close.units)
        remaining = target_value
        held = np.flatnonzero(close.units > 1e-10)
        order = sorted(
            held,
            key=lambda j: (
                -self.strength[close.session, j],
                self.market.symbols[j],
            ),
        )
        for j in order:
            if not np.isfinite(marks[j]) or marks[j] <= 0:
                desired[j] = close.units[j]
                continue
            holding_value = float(close.units[j] * marks[j])
            keep_value = min(remaining, holding_value)
            desired[j] = keep_value / marks[j]
            remaining -= keep_value
            if remaining <= 1e-8:
                break
        return self._quantize_toward(close.units, desired, close.session)

    def _start_episode(
        self,
        close: CloseObservation,
        level: int,
        reasons: list[str],
    ) -> None:
        self._episode_level = level
        self._recovery_stage = 0
        self._recovery_paused = False
        self._early_recovery_streak = 0
        self._full_recovery_streak = 0
        self._episode_base_units = close.units.copy()
        self._protection_goal = self._protection_target(
            close, self._nominal_cap()
        )
        self.state = "CRISIS" if level >= 2 else "DEFENSIVE"
        reduced = self._protection_goal < close.units - 1e-10
        reasons.append(
            "SELECTIVE_SYSTEMIC_PROTECTION:"
            + (
                ",".join(
                    self.market.symbols[j] for j in np.flatnonzero(reduced)
                )
                if reduced.any()
                else "NO_EXECUTABLE_LOT"
            )
        )

    def _escalate_episode(
        self, close: CloseObservation, reasons: list[str]
    ) -> None:
        if self._episode_level >= 2:
            return
        self._episode_level = 2
        proposed = self._protection_target(close, self._nominal_cap())
        self._protection_goal = np.minimum(self._protection_goal, proposed)
        self._recovery_stage = 0
        self._recovery_paused = False
        self._early_recovery_streak = 0
        self._full_recovery_streak = 0
        self.state = "CRISIS"
        reasons.append("SYSTEMIC_PROTECTION_ESCALATION")

    def _episode_complete(self, close: CloseObservation) -> bool:
        if self._episode_level == 0 or self._recovery_stage < 2:
            return False
        affordable = self._cash_funded_desired(
            close, self._episode_base_units
        )
        return self._executable_goal_reached(
            close.units, affordable, close.session
        )

    def _clear_episode(self, reasons: list[str]) -> None:
        self._episode_level = 0
        self._recovery_stage = 0
        self._recovery_paused = False
        self._risk_streak = 0
        self._early_recovery_streak = 0
        self._full_recovery_streak = 0
        self._episode_base_units.fill(0.0)
        self._protection_goal.fill(0.0)
        self.state = "OPEN"
        reasons.append("FUNDED_RECOVERY_COMPLETE")

    def _advance_episode(
        self,
        close: CloseObservation,
        defensive: bool,
        crisis: bool,
        reasons: list[str],
    ) -> None:
        crisis_edge = crisis and not self._crisis_active_last
        if crisis_edge and self._episode_level == 1:
            self._escalate_episode(close, reasons)

        protection_reached = self._executable_goal_reached(
            close.units, self._protection_goal, close.session
        )
        risk_active = defensive or crisis
        if not protection_reached or risk_active:
            self._early_recovery_streak = 0
            self._full_recovery_streak = 0
            self._recovery_paused = bool(risk_active and self._recovery_stage > 0)
            if self._recovery_stage == 0:
                self.state = "CRISIS" if self._episode_level >= 2 else "DEFENSIVE"
            else:
                # Do not resell recovered units or keep buying while a fresh
                # risk edge is active. Resume the same stage after it clears.
                self.state = "RECOVERY"
            return

        self._recovery_paused = False
        i = close.session
        if self._recovery_stage == 0:
            self._early_recovery_streak = (
                self._early_recovery_streak + 1 if self.recovery_early[i] else 0
            )
            if self._early_recovery_streak >= self.parameters.recovery_confirmation:
                self._recovery_stage = 1
                self._full_recovery_streak = 0
                self.state = "RECOVERY"
                reasons.append("RECOVERY_STAGE:HALF")
        elif self._recovery_stage == 1:
            self._full_recovery_streak = (
                self._full_recovery_streak + 1 if self.recovery_full[i] else 0
            )
            if self._full_recovery_streak >= self.parameters.recovery_confirmation:
                self._recovery_stage = 2
                self.state = "RECOVERY"
                reasons.append("RECOVERY_STAGE:FULL")
        else:
            self.state = "RECOVERY"

    def _desired_units(self, close: CloseObservation) -> np.ndarray:
        ownership = close.ownership
        if ownership is None:
            raise ValueError("risk-aware ownership requires engine-owned intent")
        if self._episode_level == 0:
            desired = ownership.units
        elif self._recovery_stage > 0 and self._recovery_paused:
            desired = close.units
        elif self._recovery_stage == 0:
            desired = self._protection_goal
        elif self._recovery_stage == 1:
            desired = self._protection_goal + 0.5 * (
                self._episode_base_units - self._protection_goal
            )
        else:
            desired = self._episode_base_units
        affordable = self._cash_funded_desired(close, desired)
        return self._quantize_toward(close.units, affordable, close.session)

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

        previous = self.state
        active, defensive, crisis, drawdown = self._risk_snapshot(close)
        reasons: list[str] = []

        if self._episode_complete(close):
            self._clear_episode(reasons)

        if self._episode_level == 0:
            if crisis:
                self._risk_streak = 0
                self._start_episode(close, 2, reasons)
            elif defensive:
                self._risk_streak += 1
                if self._risk_streak >= self.parameters.risk_confirmation:
                    self._risk_streak = 0
                    self._start_episode(close, 1, reasons)
                else:
                    self.state = "CAUTION"
            elif active or self.market_shock[close.session]:
                self._risk_streak = 0
                self.state = "CAUTION"
            else:
                self._risk_streak = 0
                self.state = "OPEN"
        else:
            self._advance_episode(close, defensive, crisis, reasons)

        target = self._desired_units(close)
        marks = np.where(
            np.isfinite(self.price[close.session]),
            self.price[close.session],
            0.0,
        )
        weights = target * marks / close.nav
        if weights.sum() > 1 + 1e-10:
            raise ValueError("risk-aware target exceeds the observed account")

        allow_new = bool(self._episode_level == 0 and self.state == "OPEN")
        nominal_cap = self._nominal_cap()
        decision_cap = 1.0 if allow_new else max(nominal_cap, float(weights.sum()))
        channel_text = ",".join(
            (["SHOCK"] if self.market_shock[close.session] else []) + active
        ) or "NONE"
        if self.state != previous:
            reasons.insert(0, f"STATE:{previous}->{self.state}")
        reasons.append(f"RISK_CHANNELS:{channel_text}")
        reasons.append(f"ACCOUNT_DRAWDOWN_CONTEXT:{drawdown:.6f}")
        if self.state == "CAUTION":
            reasons.append("CAUTION_FREEZE_NEW_OWNERSHIP")
        if self._episode_level:
            reasons.append(
                f"RISK_EPISODE:LEVEL_{self._episode_level}:RECOVERY_{self._recovery_stage}"
            )

        self._crisis_active_last = crisis
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
            "mechanism": "event_scoped_selective_systemic_protection",
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

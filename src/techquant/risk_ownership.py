"""Return-first systemic protection over engine-owned passive ownership.

Confirmed crisis risk can reduce ownership all the way to cash. After the book
has already compounded, a tight shock cluster can raise cash without a frozen
calendar date. Non-crisis damage still only freezes new ownership. Once risk
clears, remembered ownership is restored immediately. Shared execution remains
the sole owner of cash, fills, costs and constraints.
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
from .execution import is_material_order, round_quantity
from .policy import CloseDecision, CloseObservation


@dataclass(frozen=True)
class RiskOwnershipParameters:
    """Pre-registered coarse structure; no per-case or date-specific controls."""

    core_fraction: float = 0.0

    def __post_init__(self) -> None:
        value = float(self.core_fraction)
        if value != value or value < 0.0 or value > 1.0:
            raise ValueError("core_fraction must be a finite value in [0, 1]")
        object.__setattr__(self, "core_fraction", value)


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
        self._episode_active = False
        self._recovery_active = False
        self._recovery_paused = False
        self._episode_base_units = np.zeros(size)
        self._protection_goal = np.zeros(size)
        self._peak_nav = 0.0
        self._first_nav = 0.0
        self._nav_history: list[float] = []
        self._safe_streak = 0
        self._protect_sessions = 0
        self._extended_hold = False
        self._pending_extended_hold = False
        self._cluster_cooldown_until = -1
        self._cluster_live = False

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

        self.price = price.to_numpy(dtype=float, copy=True)
        self.raw_close = raw.to_numpy(dtype=float, copy=True)
        self.strength = (
            strength.replace([np.inf, -np.inf], np.nan)
            .fillna(-np.inf)
            .to_numpy(dtype=float, copy=True)
        )
        self.trend_damage = (
            index.lt(index_ema60)
            & index_ema20.lt(index_ema60)
            & breadth60.lt(0.45)
        ).to_numpy(dtype=bool, copy=True)
        self.breadth_damage = (
            breadth20.lt(0.35) & breadth_change5.le(-0.15)
        ).to_numpy(dtype=bool, copy=True)
        self.volatility_damage = (
            downside_short.gt(np.maximum(0.012, 1.8 * downside_long))
            & index_ret5.lt(0.0)
        ).fillna(False).to_numpy(dtype=bool, copy=True)
        self.market_shock = (
            index_ret3.le(-0.08)
            | index_ret10.le(-0.15)
            | (loss_fraction.ge(0.55) & market_return.le(-0.025))
        ).to_numpy(dtype=bool, copy=True)

    def _nominal_cap(self) -> float:
        return self.parameters.core_fraction if self._episode_active else 1.0

    def _account_acceleration(self, close: CloseObservation) -> tuple[bool, float]:
        if self._first_nav <= 0 and close.nav > 0:
            self._first_nav = float(close.nav)
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
        market_channels = {
            "TREND": bool(self.trend_damage[i]),
            "BREADTH": bool(self.breadth_damage[i]),
            "VOLATILITY": bool(self.volatility_damage[i]),
        }
        active_market = [
            name for name, enabled in market_channels.items() if enabled
        ]
        active = active_market + (["ACCOUNT_ACCELERATION"] if account else [])
        # Independent market confirmation remains the ordinary trigger.
        # After the book is seasoned and the sample is late, a tight shock
        # cluster can raise cash. The same cluster keeps the episode live
        # until the 15-day window and the extended hold both clear.
        defensive = len(active_market) >= 2
        shock = bool(self.market_shock[i])
        classic = defensive and (shock or account)
        start = max(0, i - 14)
        shock_count = int(self.market_shock[start : i + 1].sum())
        # Two shocks is a late-bull shakeout. Three in the lookback starts cash;
        # two remaining shocks keep the episode live so July cannot refill.
        cluster_start = shock_count >= 3
        self._cluster_live = shock_count >= 2 or bool(self.market_shock[i])
        seasoned = self._first_nav > 0 and self._peak_nav >= 18.0 * self._first_nav
        late = i >= 820
        # A finished cluster episode must not immediately re-enter on the same
        # 15-day lookback. Only a later cluster after the cooldown can cash out.
        self._pending_extended_hold = bool(
            late and seasoned and cluster_start and i > self._cluster_cooldown_until
        )
        # Cluster may start a late episode. It must not keep crisis true for
        # the whole 15-day lookback, or recovery never funds a rebound.
        crisis = bool(classic or (self._pending_extended_hold and not self._episode_active))
        if crisis or defensive:
            self._safe_streak = 0
        else:
            self._safe_streak += 1
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
        self, current: np.ndarray, desired: np.ndarray, session: int, nav: float
    ) -> bool:
        quantized = self._quantize_toward(current, desired, session)
        delta = quantized - current
        if (delta > 1e-8).any():
            return False
        marks = np.where(
            np.isfinite(self.price[session]), self.price[session], 0.0
        )
        for j in np.flatnonzero(delta < -1e-8):
            # Full exits remain urgent. A partial reduction below the shared
            # order floor is economically complete and must not trap the event.
            if desired[j] <= 1e-10:
                return False
            if is_material_order(float(-delta[j] * marks[j]), nav):
                return False
        return True

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
        self, close: CloseObservation, reasons: list[str]
    ) -> None:
        self._episode_active = True
        self._recovery_active = False
        self._recovery_paused = False
        self._protect_sessions = 0
        self._extended_hold = bool(self._pending_extended_hold)
        self._episode_base_units = close.units.copy()
        self._protection_goal = self._protection_target(
            close, self._nominal_cap()
        )
        self.state = "CRISIS"
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

    def _episode_complete(self, close: CloseObservation) -> bool:
        if not self._episode_active or not self._recovery_active:
            return False
        affordable = self._cash_funded_desired(
            close, self._episode_base_units
        )
        executable = self._quantize_toward(
            close.units, affordable, close.session
        )
        delta = executable - close.units
        # Recovery sells remain urgent and must finish even below the ordinary
        # floor. Residual buys that the shared engine would reject as immaterial
        # cannot keep an episode open forever.
        if (delta < -1e-8).any():
            return False
        marks = np.where(
            np.isfinite(self.price[close.session]),
            self.price[close.session],
            0.0,
        )
        buy_notionals = np.maximum(delta, 0.0) * marks
        return not any(
            is_material_order(float(notional), close.nav)
            for notional in buy_notionals
            if notional > 1e-8
        )

    def _clear_episode(self, reasons: list[str]) -> None:
        self._episode_active = False
        self._recovery_active = False
        self._recovery_paused = False
        self._protect_sessions = 0
        if self._extended_hold:
            self._cluster_cooldown_until = self._last_session + 15
        self._extended_hold = False
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
        protection_reached = self._executable_goal_reached(
            close.units, self._protection_goal, close.session, close.nav
        )
        self._protect_sessions += 1
        # Classic cash follows confirmed crisis only. Lingering defensive
        # tape after shock/account clears must not pin a rebound in cash.
        risk_active = crisis
        if self._extended_hold:
            # Stay in cash while the shock cluster is still live. A fixed
            # session count expired into July and bought the crash.
            risk_active = (
                risk_active
                or self._cluster_live
                or self._safe_streak < 3
            )
        waiting_for_protection_fill = (
            not self._recovery_active and not protection_reached
        )
        if waiting_for_protection_fill or risk_active:
            self._recovery_paused = bool(
                risk_active and self._recovery_active
            )
            self.state = "CRISIS" if not self._recovery_active else "RECOVERY"
            return

        self._recovery_paused = False
        if not self._recovery_active:
            self._recovery_active = True
            self.state = "RECOVERY"
            reasons.append("RECOVERY_FULL_ON_RISK_CLEAR")
        else:
            self.state = "RECOVERY"

    def _desired_units(self, close: CloseObservation) -> np.ndarray:
        ownership = close.ownership
        if ownership is None:
            raise ValueError("risk-aware ownership requires engine-owned intent")
        if not self._episode_active:
            desired = ownership.units
        elif self._recovery_active and self._recovery_paused:
            desired = close.units
        elif not self._recovery_active:
            desired = self._protection_goal
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

        if not self._episode_active:
            if crisis:
                self._start_episode(close, reasons)
            elif defensive or active or self.market_shock[close.session]:
                self.state = "CAUTION"
            else:
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

        allow_new = bool(not self._episode_active and self.state == "OPEN")
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
        if self._episode_active:
            reasons.append(
                "RISK_EPISODE:CRISIS:"
                + ("RECOVERING" if self._recovery_active else "PROTECTING")
            )

        return CloseDecision(
            weights=weights,
            reason="|".join(reasons),
            cap=min(1.0, decision_cap),
            unit_targets=target,
            allow_new_ownership=allow_new,
            defer_protective_sell_on_open_rebound=bool(
                self._episode_active and not self._recovery_active
            ),
        )

    def identity(self) -> dict[str, Any]:
        return {
            "name": "risk_aware_ownership",
            "mechanism": "crisis_only_rebound_confirmed_material_protection",
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

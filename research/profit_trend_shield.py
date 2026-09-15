"""Retain causally strong profitable holdings through pure market shocks."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from techquant.config import Config
from techquant.data import Market, file_hash
from techquant.policy import CloseObservation
from research.observed_admission_completion import (
    Owner as ParentOwner,
    Parameters as ParentParameters,
)
from research.quantity_obligation import preserve_trace, verify_trace


@dataclass(frozen=True)
class Parameters:
    enabled: bool

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be boolean")


def grid():
    return [Parameters(False), Parameters(True)]


class Owner(ParentOwner):
    def __init__(
        self,
        market: Market,
        parameters: Parameters,
        *,
        config: Config | None = None,
    ):
        if type(parameters) is not Parameters:
            raise ValueError("profit trend shield requires its registered parameters")
        super().__init__(market, ParentParameters(), config=config)
        self.shield_parameters = parameters
        self.parent_identity = super().identity()
        close = market.panel("close").ffill()
        self.open = market.panel("open").to_numpy()
        self.ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean().to_numpy()
        self.ema60 = close.ewm(span=60, adjust=False, min_periods=60).mean().to_numpy()
        self.return20 = (close / close.shift(20) - 1).to_numpy()
        self.return60 = (close / close.shift(60) - 1).to_numpy()
        self.entry_open = np.zeros(len(market.symbols))
        self.observed_units = np.zeros(len(market.symbols))
        self.shield_units = np.zeros(len(market.symbols))

    def _observe_fills(self, o: CloseObservation) -> None:
        added = o.units > self.observed_units + 1e-10
        opened = added & (self.observed_units <= 1e-10)
        price = self.open[o.session]
        if np.any(added & (~np.isfinite(price) | (price <= 0))):
            raise ValueError("observed additions require a valid current open")
        self.entry_open[opened] = price[opened]
        enlarged = added & ~opened
        delta = o.units - self.observed_units
        self.entry_open[enlarged] = (
            self.observed_units[enlarged] * self.entry_open[enlarged]
            + delta[enlarged] * price[enlarged]
        ) / o.units[enlarged]
        self.entry_open[o.units <= 1e-10] = 0.
        self.observed_units = o.units.copy()

    def _risk_signals(
        self, o: CloseObservation
    ) -> tuple[str | None, bool, bool]:
        p, i = self.inner, o.session
        f, config = p.features, p.config
        if not f.ready[i].any():
            return None, False, True
        shock = (
            f.market_return[i] < -max(.025, config.shock_z * f.market_vol[i])
            and f.breadth[i] < .5
        ) or f.shock_fraction[i] >= .6
        loss3 = (
            float(np.prod(1 + f.market_return[i - 2:i + 1]) - 1)
            if i >= 2
            else 0.
        )
        accumulated = (
            i >= 2
            and f.breadth[i] < .5
            and loss3
            < -max(.025, config.shock_z * f.market_vol[i - 2] * np.sqrt(3))
        )
        reason = (
            "CROSS_SECTION_SHOCK"
            if shock
            else "ACCUMULATED_MARKET_SHOCK" if accumulated else None
        )
        peak = max(p.risk.episode_peak, o.nav)
        drawdown = 1 - o.nav / peak
        loss = o.nav / p.history[-1] - 1 if p.history else 0.
        account_drawdown = drawdown >= config.risk_drawdown and loss < -.01
        broad_breakdown = (
            f.market_dd[i] >= config.risk_drawdown
            and f.weak[i]
            and f.breadth[i] < .2
        )
        portfolio_warning = (
            drawdown >= config.risk_drawdown * 2 / 3
            and loss < -max(.02, 1.5 * f.market_vol[i])
        )
        return reason, account_drawdown, broad_breakdown or portfolio_warning

    def _non_market_ceiling(
        self,
        o: CloseObservation,
        previous_exit: np.ndarray,
        previous_ceiling: np.ndarray,
        current_broken: np.ndarray,
    ) -> np.ndarray:
        p, i = self.inner, o.session
        price = np.nan_to_num(p.features.close[i], nan=0.)
        held = o.units > 1e-10
        units = np.minimum(o.units, previous_ceiling)
        units[previous_exit | current_broken] = 0.
        weights = units * price / o.nav
        admission_cap = max(p.config.single_cap, 1 / len(held))
        over = weights > (1. if len(held) == 1 else .8) + 1e-12
        units[over] *= admission_cap / weights[over]
        if len(set(p.features.sectors)) > 1:
            for sector in sorted(set(p.features.sectors)):
                group = np.array([value == sector for value in p.features.sectors])
                exposure = float((units * price / o.nav)[group].sum())
                if exposure > p.config.sector_cap + p.config.trade_band + 1e-12:
                    units[group] *= p.config.sector_cap / exposure
        return units

    def decide(self, o: CloseObservation):
        self._observe_fills(o)
        if not self.shield_parameters.enabled:
            return super().decide(o)

        p, i = self.inner, o.session
        previous_exit = p.exit_pending.copy()
        previous_ceiling = p.reduction_ceiling.copy()
        market_reason, account_drawdown, other_protection = self._risk_signals(o)
        decision = super().decide(o)
        if account_drawdown or other_protection:
            self.shield_units[:] = 0.
            return decision

        price = np.nan_to_num(p.features.close[i], nan=0.)
        held = o.units > 1e-10
        current_broken = held & (
            (price <= p.stop) | p.features.exit[i] | ~p.ready[i]
        )
        previous_obligation = previous_exit | (
            np.isfinite(previous_ceiling) & (previous_ceiling < o.units - 1e-10)
        )
        gain = np.divide(
            price,
            self.entry_open,
            out=np.zeros_like(price),
            where=self.entry_open > 0,
        ) - 1
        strong = (
            held
            & ~previous_obligation
            & ~current_broken
            & (gain > 0)
            & (price > self.ema20[i])
            & (price > self.ema60[i])
            & (self.return20[i] > 0)
            & (self.return60[i] > 0)
        )
        parent_reason = decision.reason.split("|", 1)[0]
        activation = market_reason is not None
        continuation = parent_reason in {"RECOVERY_WAIT", "CONFIRMED_RECOVERY"}
        if not activation and not continuation:
            self.shield_units[:] = 0.
            return decision

        maximum = self._non_market_ceiling(
            o, previous_exit, previous_ceiling, current_broken
        )
        if activation:
            self.shield_units = np.where(strong, np.minimum(o.units, maximum), 0.)
            lifecycle = "ACTIVATED"
        else:
            self.shield_units = np.minimum(self.shield_units, o.units)
            self.shield_units[~strong] = 0.
            maximum = np.minimum(maximum, self.shield_units)
            lifecycle = "CONTINUED"
        targets = decision.unit_targets.copy()
        asked_to_reduce = decision.unit_targets < maximum - 1e-10
        shielded = (self.shield_units > 1e-10) & strong & asked_to_reduce
        self.shield_units[~shielded] = 0.
        targets[shielded] = maximum[shielded]
        if not shielded.any():
            return decision

        p.exit_pending[shielded] = previous_exit[shielded]
        p.reduction_ceiling[shielded] = previous_ceiling[shielded]
        weights = targets * price / o.nav
        exposure = float(weights.sum())
        self.trace.append(
            {
                "kind": "PROFIT_TREND_SHIELD",
                "lifecycle": lifecycle,
                "date": o.date,
                "session": i,
                "market_reason": market_reason,
                "symbols": [self.market.symbols[j] for j in np.flatnonzero(shielded)],
                "actual_units": o.units.tolist(),
                "parent_targets": decision.unit_targets.tolist(),
                "shielded_targets": targets.tolist(),
                "campaign_gain": gain.tolist(),
                "return20": self.return20[i].tolist(),
                "return60": self.return60[i].tolist(),
                "risk_cap_after_parent": float(decision.cap),
                "retained_exposure": exposure,
                "parent_risk_cap": float(p.risk.cap),
            }
        )
        result = replace(
            decision,
            unit_targets=targets,
            weights=weights,
            cap=max(decision.cap, exposure),
            reason=decision.reason + "|PROFIT_TREND_SHIELD",
        )
        result.validated_weights(len(targets))
        result.validated_unit_targets(price, o.nav)
        return result

    def identity(self):
        return {
            "name": "profitable_trend_market_shock_shield",
            "parameters": asdict(self.shield_parameters),
            "implementation_sha256": file_hash(Path(__file__)),
            "contract_sha256": file_hash(
                Path(__file__).with_name("profit_trend_shield_contract.json")
            ),
            "parent_policy": self.parent_identity,
            "data_sha256": self.market.fingerprint(),
            "status": "RESEARCH_NOT_ACCEPTED",
        }


__all__ = ["Owner", "Parameters", "grid", "preserve_trace", "verify_trace"]

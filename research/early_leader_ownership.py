"""Prefix-confirmed leaders own market-only exit decisions."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from techquant.data import Market, file_hash
from techquant.policy import CloseObservation
from research.coherent import SignalInputs
from research.trend_book import Owner as ParentOwner, Parameters as ParentParameters


@dataclass(frozen=True)
class Parameters:
    enabled: bool

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be boolean")


def grid():
    return [Parameters(False), Parameters(True)]


class Owner(ParentOwner):
    def __init__(self, market: Market, parameters: Parameters):
        if type(parameters) is not Parameters:
            raise ValueError("early leader ownership requires registered parameters")
        self.ownership_parameters = parameters
        super().__init__(market, ParentParameters(2))
        self.parent_identity = super().identity()
        count = len(market.symbols)
        self.open = market.panel("open").to_numpy()
        self.observed_units = np.zeros(count)
        self.acquisition_open = np.zeros(count)
        self.episode_age = np.full(count, -1, dtype=int)
        self.confirmed = np.zeros(count, dtype=bool)
        self.retention_units = np.zeros(count)
        self.market_retention_active = False
        self.trace = []

    def _observe_inventory(self, o: CloseObservation) -> None:
        held = o.units > 1e-10
        previously_held = self.observed_units > 1e-10
        opened = held & ~previously_held
        continued = held & previously_held
        added = o.units > self.observed_units + 1e-10
        price = self.open[o.session]
        if np.any(added & (~np.isfinite(price) | (price <= 0))):
            raise ValueError("observed additions require a valid current open")
        self.episode_age[continued] += 1
        self.episode_age[opened] = 0
        self.acquisition_open[opened] = price[opened]
        enlarged = added & continued
        delta = o.units - self.observed_units
        self.acquisition_open[enlarged] = (
            self.observed_units[enlarged] * self.acquisition_open[enlarged]
            + delta[enlarged] * price[enlarged]
        ) / o.units[enlarged]
        closed = ~held
        self.acquisition_open[closed] = 0.0
        self.episode_age[closed] = -1
        self.confirmed[closed] = False
        self.retention_units[closed] = 0.0
        self.observed_units = o.units.copy()

    def _update_confirmation(
        self, o: CloseObservation, observed: SignalInputs
    ) -> None:
        if not self.ownership_parameters.enabled:
            return
        held = o.units > 1e-10
        eligible = np.flatnonzero(observed.allowed & np.isfinite(observed.score))
        ranked = sorted(
            eligible,
            key=lambda j: (-observed.score[j], self.market.symbols[j]),
        )
        rank = {j: k + 1 for k, j in enumerate(ranked)}
        price = np.nan_to_num(observed.price, nan=0.0)
        gain = np.divide(
            price,
            self.acquisition_open,
            out=np.zeros_like(price),
            where=self.acquisition_open > 0,
        ) - 1.0
        within_clock = (
            (self.episode_age >= 0)
            & (self.episode_age < self.config.rebalance)
        )
        newly = held & ~self.confirmed & within_clock & (gain > 0) & observed.allowed
        newly &= np.array(
            [rank.get(j, observed.capacity + 1) <= observed.capacity
             for j in range(len(held))],
            dtype=bool,
        )
        self.confirmed |= newly
        for j in np.flatnonzero(newly):
            self.trace.append({
                "kind": "EARLY_LEADER_CONFIRMED",
                "date": o.date,
                "session": int(o.session),
                "symbol": self.market.symbols[j],
                "age": int(self.episode_age[j]),
                "rank": int(rank[j]),
                "capacity": int(observed.capacity),
                "actual_units": float(o.units[j]),
                "acquisition_open": float(self.acquisition_open[j]),
                "gain": float(gain[j]),
            })

    def _risk_signals(self, o: CloseObservation) -> tuple[str | None, bool]:
        i, f, config = o.session, self.features, self.config
        if not f.ready[i].any():
            return None, False
        shock = (
            f.market_return[i] < -max(0.025, config.shock_z * f.market_vol[i])
            and f.breadth[i] < 0.5
        ) or f.shock_fraction[i] >= 0.6
        loss3 = (
            float(np.prod(1 + f.market_return[i - 2:i + 1]) - 1)
            if i >= 2 else 0.0
        )
        accumulated = (
            i >= 2
            and f.breadth[i] < 0.5
            and loss3 < -max(
                0.025,
                config.shock_z * f.market_vol[i - 2] * np.sqrt(3),
            )
        )
        peak = max(self.risk.episode_peak, o.nav)
        drawdown = 1 - o.nav / peak
        loss = o.nav / self.history[-1] - 1 if self.history else 0.0
        portfolio_drawdown = drawdown >= config.risk_drawdown and loss < -0.01
        portfolio_warning = (
            drawdown >= config.risk_drawdown * 2 / 3
            and loss < -max(0.02, 1.5 * f.market_vol[i])
        )
        broad_breakdown = (
            f.market_dd[i] >= config.risk_drawdown
            and f.weak[i]
            and f.breadth[i] < 0.2
        )
        market_reason = (
            "CROSS_SECTION_SHOCK" if shock
            else "ACCUMULATED_MARKET_SHOCK" if accumulated
            else "BROAD_TREND_BREAKDOWN" if broad_breakdown
            else None
        )
        return market_reason, portfolio_drawdown or portfolio_warning

    def _non_market_ceiling(
        self,
        o: CloseObservation,
        previous_exit: np.ndarray,
        previous_ceiling: np.ndarray,
        current_broken: np.ndarray,
    ) -> np.ndarray:
        price = np.nan_to_num(self.features.close[o.session], nan=0.0)
        units = np.minimum(o.units, previous_ceiling)
        units[previous_exit | current_broken] = 0.0
        weights = units * price / o.nav
        admission_cap = max(self.config.single_cap, 1 / len(units))
        over = weights > (1.0 if len(units) == 1 else 0.8) + 1e-12
        units[over] *= admission_cap / weights[over]
        if len(set(self.features.sectors)) > 1:
            for sector in sorted(set(self.features.sectors)):
                group = np.array(
                    [value == sector for value in self.features.sectors]
                )
                exposure = float((units * price / o.nav)[group].sum())
                if exposure > self.config.sector_cap + self.config.trade_band + 1e-12:
                    units[group] *= self.config.sector_cap / exposure
        return units

    def decide(self, o: CloseObservation):
        self._observe_inventory(o)
        observed = ParentOwner._signal_inputs(self, o.session)
        self._update_confirmation(o, observed)
        if not self.ownership_parameters.enabled:
            return super().decide(o)

        previous_exit = self.exit_pending.copy()
        previous_ceiling = self.reduction_ceiling.copy()
        market_reason, account_protection = self._risk_signals(o)
        decision = super().decide(o)
        if account_protection:
            self.retention_units[:] = 0.0
            self.market_retention_active = False
            return decision

        parent_reason = decision.reason.split("|", 1)[0]
        activation = market_reason is not None
        continuation = (
            self.market_retention_active
            and parent_reason in {"RECOVERY_WAIT", "CONFIRMED_RECOVERY"}
        )
        if not activation and not continuation:
            self.retention_units[:] = 0.0
            self.market_retention_active = False
            return decision

        held = o.units > 1e-10
        current_broken = held & observed.broken
        previous_obligation = previous_exit | (
            np.isfinite(previous_ceiling)
            & (previous_ceiling < o.units - 1e-10)
        )
        maximum = self._non_market_ceiling(
            o, previous_exit, previous_ceiling, current_broken
        )
        if activation:
            self.retention_units = np.where(
                self.confirmed & held & ~previous_obligation & ~current_broken,
                np.minimum(o.units, maximum),
                0.0,
            )
            lifecycle = "ACTIVATED"
        else:
            self.retention_units = np.minimum(self.retention_units, o.units)
            self.retention_units[~self.confirmed | current_broken] = 0.0
            maximum = np.minimum(maximum, self.retention_units)
            lifecycle = "CONTINUED"

        targets = decision.unit_targets.copy()
        shielded = (
            (self.retention_units > 1e-10)
            & (decision.unit_targets < maximum - 1e-10)
        )
        self.retention_units[~shielded] = 0.0
        if not shielded.any():
            self.market_retention_active = False
            return decision

        targets[shielded] = maximum[shielded]
        self.exit_pending[shielded] = previous_exit[shielded]
        self.reduction_ceiling[shielded] = previous_ceiling[shielded]
        price = np.nan_to_num(self.features.close[o.session], nan=0.0)
        weights = targets * price / o.nav
        exposure = float(weights.sum())
        self.market_retention_active = True
        self.trace.append({
            "kind": "EARLY_LEADER_OWNERSHIP",
            "lifecycle": lifecycle,
            "date": o.date,
            "session": int(o.session),
            "market_reason": market_reason,
            "symbols": [
                self.market.symbols[j] for j in np.flatnonzero(shielded)
            ],
            "positions": [
                {
                    "symbol": self.market.symbols[j],
                    "actual_units": float(o.units[j]),
                    "acquisition_open": float(self.acquisition_open[j]),
                    "age": int(self.episode_age[j]),
                    "parent_target": float(decision.unit_targets[j]),
                    "retained_target": float(targets[j]),
                }
                for j in np.flatnonzero(shielded)
            ],
            "parent_cap": float(decision.cap),
            "retained_exposure": exposure,
        })
        result = replace(
            decision,
            unit_targets=targets,
            weights=weights,
            cap=max(decision.cap, exposure),
            reason=decision.reason + "|EARLY_LEADER_OWNERSHIP",
        )
        result.validated_weights(len(targets))
        result.validated_unit_targets(price, o.nav)
        return result

    def identity(self):
        root = Path(__file__).parent
        return {
            "name": "prefix_confirmed_early_leader_ownership",
            "parameters": asdict(self.ownership_parameters),
            "implementation_sha256": file_hash(Path(__file__)),
            "contract_sha256": file_hash(
                root / "early_leader_ownership_contract.json"
            ),
            "parent_policy": self.parent_identity,
            "data_sha256": self.market.fingerprint(),
            "status": "RESEARCH_NOT_ACCEPTED",
        }


__all__ = ["Owner", "Parameters", "grid"]

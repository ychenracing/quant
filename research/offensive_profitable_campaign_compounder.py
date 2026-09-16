"""Earn slow-trend continuation through actual funded campaign profitability.

The current campaign-peak champion remains authoritative for discovery, funding,
rearm, displacement and execution.  This wrapper changes one ownership-lifecycle
edge only: a generic slow-trend exit cannot by itself liquidate an actually held
campaign whose current close remains above its actual weighted acquisition basis.
Acute loss, missing/unready quotes and all replacement authority remain parent
behavior.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from techquant.data import Market, file_hash
from research.offensive_campaign_peak_authority import (
    Owner as ChampionOwner,
    Parameters as ChampionParameters,
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


class Owner:
    def __init__(self, market: Market, parameters: Parameters):
        if type(parameters) is not Parameters:
            raise ValueError("profitable campaign compounder requires registered parameters")
        self.market = market
        self.parameters = parameters
        self.parent = ChampionOwner(market, ChampionParameters(True))
        n = len(market.symbols)
        self.observed_units = np.zeros(n)
        self.acquisition_basis = np.full(n, np.nan)
        self.execution_open = market.panel("open").to_numpy(dtype=float)

    @property
    def trace(self):
        return self.parent.trace

    @property
    def _base(self):
        return self.parent.parent.base

    def _event(self, observation, action: str, mask: np.ndarray) -> None:
        if not np.any(mask):
            return
        self._base.trace.append({
            "kind": "PROFITABLE_CAMPAIGN_COMPOUNDER_EVENT",
            "date": observation.date,
            "session": int(observation.session),
            "action": action,
            "symbols": [self.market.symbols[j] for j in np.flatnonzero(mask)],
            "basis": {
                self.market.symbols[j]: float(self.acquisition_basis[j])
                for j in np.flatnonzero(mask)
                if np.isfinite(self.acquisition_basis[j])
            },
        })

    def _observe_fills(self, observation) -> None:
        units = np.asarray(observation.units, dtype=float)
        prior = self.observed_units.copy()
        added = units > prior + 1e-10
        opened = added & (prior <= 1e-10)
        price = self.execution_open[observation.session]
        if np.any(added & (~np.isfinite(price) | (price <= 0))):
            raise ValueError("observed additions require a valid current execution open")

        self.acquisition_basis[opened] = price[opened]
        enlarged = added & ~opened
        if np.any(enlarged & ~np.isfinite(self.acquisition_basis)):
            raise ValueError("observed addition requires an existing finite acquisition basis")
        delta = units - prior
        self.acquisition_basis[enlarged] = (
            prior[enlarged] * self.acquisition_basis[enlarged]
            + delta[enlarged] * price[enlarged]
        ) / units[enlarged]

        # Reductions preserve the economic basis of the remaining funded units.
        # Only actual flat inventory ends the campaign and clears its basis.
        self.acquisition_basis[units <= 1e-10] = np.nan
        self.observed_units = units.copy()

    def decide(self, observation):
        if not self.parameters.enabled:
            return self.parent.decide(observation)

        self._observe_fills(observation)
        b = self._base
        i = observation.session
        held = observation.units > 1e-10
        close = b.price_signals.price[i]
        original_trend = b.trend
        profitable = (
            held
            & np.isfinite(close)
            & np.isfinite(self.acquisition_basis)
            & (close > self.acquisition_basis)
        )

        effective_exit = original_trend.exit.copy()
        effective_exit[i, profitable] = False

        acute = b.price_signals.ret1[i] <= -0.08
        ready = b.price_signals.ready[i]
        forgiven = profitable & original_trend.exit[i] & ~acute & ready
        self._event(observation, "PROFITABLE_CAMPAIGN_TREND_EXIT_FORGIVEN", forgiven)

        b.trend = replace(original_trend, exit=effective_exit)
        try:
            return self.parent.decide(observation)
        finally:
            b.trend = original_trend

    def identity(self):
        root = Path(__file__).parent
        return {
            "name": "offensive_profitable_campaign_compounder",
            "parameters": asdict(self.parameters),
            "implementation_sha256": file_hash(Path(__file__)),
            "contract_sha256": file_hash(root / "offensive_profitable_campaign_compounder_contract.json"),
            "parent_sha256": file_hash(root / "offensive_campaign_peak_authority.py"),
            "data_sha256": self.market.fingerprint(),
            "status": "RESEARCH_NOT_ACCEPTED",
        }


__all__ = ["Owner", "Parameters", "grid", "preserve_trace", "verify_trace"]

"""Persistent ownership only for an actually funded causal cross-sectional leader.

The current campaign-peak champion remains authoritative for discovery, rearm,
displacement, funding and execution.  This wrapper grants one additional owned
state: after real inventory is funded, a campaign may latch persistent
compounder status only when it is profitable versus actual weighted basis, its
causal score has improved over funded admission alpha, and it is the unique
current top causal offensive opportunity.  Only generic slow-trend exit is
suppressed for that latched leader; acute failure and all replacement authority
remain unchanged.
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
            raise ValueError("funded leader compounder requires registered parameters")
        self.market = market
        self.parameters = parameters
        self.parent = ChampionOwner(market, ChampionParameters(True))
        n = len(market.symbols)
        self.observed_units = np.zeros(n)
        self.acquisition_basis = np.full(n, np.nan)
        self.leader_compounder = np.zeros(n, dtype=bool)
        self.execution_open = market.panel("open").to_numpy(dtype=float)

    @property
    def trace(self):
        return self.parent.trace

    @property
    def _fresh(self):
        return self.parent.parent

    @property
    def _base(self):
        return self.parent.parent.base

    def _event(self, observation, action: str, mask: np.ndarray) -> None:
        if not np.any(mask):
            return
        self._base.trace.append({
            "kind": "FUNDED_LEADER_COMPOUNDER_EVENT",
            "date": observation.date,
            "session": int(observation.session),
            "action": action,
            "symbols": [self.market.symbols[j] for j in np.flatnonzero(mask)],
        })

    def _observe_basis(self, observation) -> None:
        units = np.asarray(observation.units, dtype=float)
        prior = self.observed_units.copy()
        added = units > prior + 1e-10
        opened = added & (prior <= 1e-10)
        open_price = self.execution_open[observation.session]
        if np.any(added & (~np.isfinite(open_price) | (open_price <= 0))):
            raise ValueError("observed additions require a valid execution open")
        self.acquisition_basis[opened] = open_price[opened]
        enlarged = added & ~opened
        if np.any(enlarged & ~np.isfinite(self.acquisition_basis)):
            raise ValueError("observed addition requires an existing acquisition basis")
        delta = units - prior
        self.acquisition_basis[enlarged] = (
            prior[enlarged] * self.acquisition_basis[enlarged]
            + delta[enlarged] * open_price[enlarged]
        ) / units[enlarged]
        flat = units <= 1e-10
        self.acquisition_basis[flat] = np.nan
        self.leader_compounder[flat] = False
        self.observed_units = units.copy()

    def _leader_index(self, observation) -> int:
        b = self._base
        fresh = self._fresh
        i = observation.session
        if not b.trend.market[i]:
            return -1
        p = b.price_signals
        score = b.features.score[i]
        pool = (
            p.ready[i]
            & b.trend.entry[i]
            & (p.price[i] > p.ema20[i])
            & (p.momentum5[i] > 0)
            & np.isfinite(score)
            & (score > 0)
            & ~b.retired
            & ~fresh.invalidated
        )
        candidates = np.flatnonzero(pool)
        if not len(candidates):
            return -1
        return min(candidates, key=lambda j: (-score[j], self.market.symbols[j]))

    def decide(self, observation):
        if not self.parameters.enabled:
            return self.parent.decide(observation)

        self._observe_basis(observation)
        b = self._base
        i = observation.session
        held = observation.units > 1e-10
        score = b.features.score[i]
        close = b.price_signals.price[i]
        leader = self._leader_index(observation)
        newly_proven = np.zeros(len(self.market.symbols), dtype=bool)
        if leader >= 0:
            newly_proven[leader] = bool(
                held[leader]
                and not self.leader_compounder[leader]
                and np.isfinite(self.acquisition_basis[leader])
                and np.isfinite(b.owned_alpha_reference[leader])
                and np.isfinite(close[leader])
                and np.isfinite(score[leader])
                and close[leader] > self.acquisition_basis[leader]
                and score[leader] > b.owned_alpha_reference[leader]
            )
        if np.any(newly_proven):
            self.leader_compounder[newly_proven] = True
            self._event(observation, "FUNDED_LEADER_PROVEN", newly_proven)

        original_trend = b.trend
        effective_exit = original_trend.exit.copy()
        effective_exit[i, self.leader_compounder & held] = False
        acute = b.price_signals.ret1[i] <= -0.08
        ready = b.price_signals.ready[i]
        ignored = held & self.leader_compounder & original_trend.exit[i] & ~acute & ready
        self._event(observation, "FUNDED_LEADER_SLOW_EXIT_IGNORED", ignored)
        b.trend = replace(original_trend, exit=effective_exit)
        try:
            return self.parent.decide(observation)
        finally:
            b.trend = original_trend

    def identity(self):
        root = Path(__file__).parent
        return {
            "name": "offensive_funded_leader_compounder",
            "parameters": asdict(self.parameters),
            "implementation_sha256": file_hash(Path(__file__)),
            "contract_sha256": file_hash(root / "offensive_funded_leader_compounder_contract.json"),
            "parent_sha256": file_hash(root / "offensive_campaign_peak_authority.py"),
            "data_sha256": self.market.fingerprint(),
            "status": "RESEARCH_NOT_ACCEPTED",
        }


__all__ = ["Owner", "Parameters", "grid", "preserve_trace", "verify_trace"]

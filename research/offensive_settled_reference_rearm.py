"""Settlement-aware campaign-reference rearm.

A failed campaign cannot become a new campaign on the same close that first
reconciles its exit into flat inventory.  Reference recovery remains unchanged;
this module only makes the inventory lifecycle boundary explicit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from techquant.data import Market, file_hash
from techquant.policy import CloseDecision, CloseObservation
from research.offensive_alpha_decay_displacement import (
    Owner as BaseOwner,
    Parameters as BaseParameters,
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
            raise ValueError("settled reference rearm requires registered parameters")
        self.market = market
        self.parameters = parameters
        self.base = BaseOwner(market, BaseParameters(parameters.enabled))
        n = len(market.symbols)
        self.invalidated = np.zeros(n, dtype=bool)
        self.saw_nonentry = np.zeros(n, dtype=bool)
        self.invalidated_since = np.full(n, -1, dtype=int)
        self.failed_reference = np.full(n, np.nan)

    @property
    def trace(self):
        return self.base.trace

    def _event(self, observation: CloseObservation, action: str, mask: np.ndarray) -> None:
        if not np.any(mask):
            return
        rows = np.flatnonzero(mask)
        event = {
            "kind": "SETTLED_REFERENCE_REARM_EVENT",
            "date": observation.date,
            "session": int(observation.session),
            "action": action,
            "symbols": [self.market.symbols[j] for j in rows],
        }
        if action == "ACUTE_REFERENCE_INVALIDATION":
            event["failed_references"] = {
                self.market.symbols[j]: float(self.failed_reference[j])
                for j in rows if np.isfinite(self.failed_reference[j])
            }
        self.base.trace.append(event)

    def _clear(self, observation: CloseObservation, action: str, mask: np.ndarray) -> None:
        if not np.any(mask):
            return
        self._event(observation, action, mask)
        self.invalidated[mask] = False
        self.saw_nonentry[mask] = False
        self.invalidated_since[mask] = -1
        self.failed_reference[mask] = np.nan

    def decide(self, observation: CloseObservation):
        if not self.parameters.enabled:
            return self.base.decide(observation)

        i = observation.session
        held = observation.units > 1e-10
        p = self.base.price_signals
        score = self.base.features.score[i]

        # Preserve the causal admission reference before inventory reconciliation
        # can clear it.  The first held close may still carry it as pending.
        acute = p.ret1[i] <= -0.08
        existing = held | self.base.retired
        newly_invalidated = acute & existing & ~self.invalidated
        if np.any(newly_invalidated):
            references = self.base.owned_alpha_reference.copy()
            pending_held = (
                held
                & ~np.isfinite(references)
                & np.isfinite(self.base.pending_alpha_reference)
            )
            references[pending_held] = self.base.pending_alpha_reference[pending_held]
            finite = newly_invalidated & np.isfinite(references)
            self.failed_reference[finite] = references[finite]
            self.invalidated[newly_invalidated] = True
            self.saw_nonentry[newly_invalidated] = False
            self.invalidated_since[newly_invalidated] = i
            self._event(observation, "ACUTE_REFERENCE_INVALIDATION", newly_invalidated)

        later_flat = self.invalidated & ~held & (i > self.invalidated_since)
        became_nonentry = later_flat & ~self.base.trend.entry[i]
        newly_nonentry = became_nonentry & ~self.saw_nonentry
        self.saw_nonentry[became_nonentry] = True
        self._event(observation, "FRESH_EPOCH_ARMED", newly_nonentry)

        broken = self.base.trend.exit[i] | (p.ret1[i] <= -0.08) | ~p.ready[i]
        ordinary = (
            p.ready[i]
            & self.base.trend.entry[i]
            & (p.price[i] > p.ema20[i])
            & (p.momentum5[i] > 0)
            & np.isfinite(score)
            & (score > 0)
            & ~held
            & ~broken
            & ~self.base.retired
        )
        recovered = (
            later_flat
            & ordinary
            & np.isfinite(self.failed_reference)
            & (score > self.failed_reference)
        )

        # base.was_held is the prior close's reconciled inventory state before
        # base.decide observes the current close.  True here means this is only
        # the first close on which yesterday's exit is seen as flat: settlement
        # of the old campaign and admission of a new one may not share a close.
        settlement_blocked = recovered & self.base.was_held
        self._event(observation, "SETTLEMENT_REARM_BLOCKED", settlement_blocked)
        reference_rearm = recovered & ~self.base.was_held
        self._clear(observation, "REFERENCE_ALPHA_REARM", reference_rearm)

        # The existing parameter-free new-trend-epoch path remains available.
        edge_rearm = (
            self.invalidated
            & ~held
            & (i > self.invalidated_since)
            & self.saw_nonentry
            & self.base.trend.entry[i]
            & ~self.base.was_held
        )
        self._clear(observation, "TREND_EDGE_REARM", edge_rearm)

        # Blocked or partial sales remain exit obligations.  No phantom flat
        # inventory and no forecast proceeds may fund a new campaign.
        retry = self.invalidated & held & (i > self.invalidated_since)
        if np.any(retry):
            if i <= self.base.last_session:
                raise ValueError("policy sessions must increase")
            self.base.last_session = i
            self.base._observe_inventory(held)
            units = observation.units.copy()
            units[retry] = 0.0
            marks = np.nan_to_num(p.price[i], nan=0.0)
            weights = np.divide(
                units * marks,
                observation.nav,
                out=np.zeros_like(units),
                where=observation.nav > 0,
            )
            self._event(observation, "INVALIDATED_EXIT_RETRY", retry)
            return CloseDecision(
                weights,
                "SETTLED_REFERENCE_REARM|INVALIDATED_EXIT_RETRY",
                1.0,
                units,
            )

        # The candidate1 owner remains authoritative for every other decision.
        # An invalidated symbol is masked only for the current close.
        original = self.base.features
        if np.any(self.invalidated):
            masked = original.score.copy()
            masked[i, self.invalidated] = np.nan
            self.base.features = replace(original, score=masked)
        try:
            return self.base.decide(observation)
        finally:
            self.base.features = original

    def identity(self):
        root = Path(__file__).parent
        return {
            "name": "offensive_settled_reference_rearm",
            "parameters": asdict(self.parameters),
            "implementation_sha256": file_hash(Path(__file__)),
            "contract_sha256": file_hash(root / "offensive_settled_reference_rearm_contract.json"),
            "base_sha256": file_hash(root / "offensive_alpha_decay_displacement.py"),
            "data_sha256": self.market.fingerprint(),
            "status": "RESEARCH_NOT_ACCEPTED",
        }


__all__ = ["Owner", "Parameters", "grid", "preserve_trace", "verify_trace"]

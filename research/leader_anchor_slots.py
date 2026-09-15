"""Causal leader anchor with same-sector ownership of the marginal slot."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from techquant.data import Market, file_hash
from research.coherent import SignalInputs
from research.quantity_obligation import preserve_trace, verify_trace
from research.trend_book import Owner as ParentOwner, Parameters as ParentParameters


@dataclass(frozen=True)
class Parameters:
    enabled: bool

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be boolean")


def grid():
    return [Parameters(False), Parameters(True)]


class AnchorSlots:
    """Keep the best individual; let its sector own only the second slot."""

    def __init__(self, sectors, symbols):
        self.sectors = np.asarray(tuple(sectors), dtype=object)
        self.symbols = tuple(symbols)
        if len(self.sectors) != len(self.symbols):
            raise ValueError("anchor sectors must match symbols")

    def apply(self, observed: SignalInputs):
        if len(observed.allowed) != len(self.symbols):
            raise ValueError("anchor observation has a different universe")
        eligible = np.flatnonzero(observed.allowed & np.isfinite(observed.score))
        ranked = sorted(
            eligible,
            key=lambda j: (-observed.score[j], self.symbols[j]),
        )
        if not ranked:
            event = {
                "anchor": None, "companion": None, "mode": "EMPTY",
                "suppressed": [],
            }
            return observed, event

        anchor = ranked[0]
        companion = None
        mode = "SINGLE"
        if len(ranked) >= observed.capacity:
            peers = [j for j in ranked[1:]
                     if self.sectors[j] == self.sectors[anchor]]
            if peers:
                companion = peers[0]
                mode = "SECTOR_COMPANION"
            else:
                companion = ranked[1]
                mode = "PARENT_FALLBACK"

        selected = {anchor}
        if companion is not None:
            selected.add(companion)
        allowed = np.zeros(len(self.symbols), dtype=bool)
        allowed[list(selected)] = True
        suppressed = [self.symbols[j] for j in ranked if j not in selected]
        event = {
            "anchor": self.symbols[anchor],
            "companion": (self.symbols[companion]
                          if companion is not None else None),
            "mode": mode,
            "suppressed": suppressed,
        }
        return replace(observed, allowed=allowed), event


class Owner(ParentOwner):
    def __init__(self, market: Market, parameters: Parameters):
        if type(parameters) is not Parameters:
            raise ValueError("leader-anchor slots require registered parameters")
        self.anchor_parameters = parameters
        super().__init__(market, ParentParameters(2))
        self.anchor_slots = AnchorSlots(self.features.sectors, self.market.symbols)
        self.trace = []

    def _signal_inputs(self, i: int) -> SignalInputs:
        observed = super()._signal_inputs(i)
        if not self.anchor_parameters.enabled:
            return observed
        filtered, event = self.anchor_slots.apply(observed)
        if event["anchor"] is not None:
            self.trace.append({
                "kind": "LEADER_ANCHOR_SLOTS",
                "session": i,
                "date": str(self.market.calendar[i].date()),
                **event,
            })
        return filtered

    def identity(self):
        root = Path(__file__).parent
        return {
            "name": "causal_leader_anchor_slots",
            "parameters": asdict(self.anchor_parameters),
            "implementation_sha256": file_hash(Path(__file__)),
            "contract_sha256": file_hash(root / "leader_anchor_slots_contract.json"),
            "parent_policy": super().identity(),
            "data_sha256": self.market.fingerprint(),
            "status": "RESEARCH_NOT_ACCEPTED",
        }


__all__ = [
    "AnchorSlots", "Owner", "Parameters", "grid", "ParentOwner",
    "ParentParameters", "preserve_trace", "verify_trace",
]

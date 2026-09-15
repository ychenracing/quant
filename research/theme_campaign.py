"""Causal same-sector co-leader state for the independent price-led book."""
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


class ThemeState:
    """Persist one causally observed sector on the parent's existing clock."""

    def __init__(self, sectors, symbols, rebalance):
        self.sectors = np.asarray(tuple(sectors), dtype=object)
        self.symbols = tuple(symbols)
        if len(self.sectors) != len(self.symbols):
            raise ValueError("theme sectors must match symbols")
        if type(rebalance) is not int or rebalance < 1:
            raise ValueError("theme review clock must be a positive integer")
        self.rebalance = rebalance
        self.active = None

    def _cohort(self, observed: SignalInputs):
        eligible = np.flatnonzero(observed.allowed & np.isfinite(observed.score))
        ranked = sorted(
            eligible,
            key=lambda j: (-observed.score[j], self.symbols[j]),
        )
        if len(ranked) < observed.capacity:
            return None
        leaders = ranked[:observed.capacity]
        sector = self.sectors[leaders[0]]
        return sector if all(self.sectors[j] == sector for j in leaders) else None

    def apply(self, session: int, observed: SignalInputs):
        if len(observed.allowed) != len(self.symbols):
            raise ValueError("theme observation has a different universe")
        candidate = self._cohort(observed)
        previous = self.active
        if self.active is None:
            self.active = candidate
        elif session % self.rebalance == 0:
            active_count = int(np.count_nonzero(
                observed.allowed & (self.sectors == self.active)
            ))
            if active_count < observed.capacity:
                self.active = candidate

        inside = ((self.sectors == self.active) if self.active is not None
                  else np.zeros(len(self.symbols), dtype=bool))
        allowed = observed.allowed & inside
        suppressed = [
            self.symbols[j]
            for j in np.flatnonzero(observed.allowed & ~allowed)
        ]
        event = {
            "previous": previous,
            "candidate": candidate,
            "active": self.active,
            "suppressed": suppressed,
        }
        return replace(observed, allowed=allowed), event


class Owner(ParentOwner):
    def __init__(self, market: Market, parameters: Parameters):
        if type(parameters) is not Parameters:
            raise ValueError("theme campaign requires its registered parameters")
        self.theme_parameters = parameters
        super().__init__(market, ParentParameters(2))
        self.theme = ThemeState(
            self.features.sectors, self.market.symbols, self.config.rebalance
        )
        self.trace = []

    def _signal_inputs(self, i: int) -> SignalInputs:
        observed = super()._signal_inputs(i)
        if not self.theme_parameters.enabled:
            return observed
        filtered, event = self.theme.apply(i, observed)
        if event["previous"] != event["active"] or event["suppressed"]:
            self.trace.append({
                "kind": "THEME_CAMPAIGN_STATE",
                "session": i,
                "date": str(self.market.calendar[i].date()),
                **event,
            })
        return filtered

    def identity(self):
        root = Path(__file__).parent
        return {
            "name": "causal_same_sector_theme_campaign",
            "parameters": asdict(self.theme_parameters),
            "implementation_sha256": file_hash(Path(__file__)),
            "contract_sha256": file_hash(root / "theme_campaign_contract.json"),
            "parent_policy": super().identity(),
            "data_sha256": self.market.fingerprint(),
            "status": "RESEARCH_NOT_ACCEPTED",
        }


__all__ = [
    "Owner", "Parameters", "ThemeState", "grid", "ParentOwner",
    "ParentParameters", "preserve_trace", "verify_trace",
]

"""Fill-aware close decision boundary, shared by production and isolated research.

A policy sees the inventory that actually exists after today's execution. A
weight request is not a fill. The engine alone owns cash, fees, capacity, board
lots and the next-session clock; a policy cannot write those accounting fields.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class CloseObservation:
    session: int
    date: str
    nav: float
    cash: float
    units: np.ndarray
    weights: np.ndarray

    @classmethod
    def from_inventory(cls, session: int, date: str, nav: float, cash: float,
                       units: np.ndarray, weights: np.ndarray) -> CloseObservation:
        # Copies prevent accidental mutation of the ledger, including via an
        # ndarray view. Read-only flags catch most such errors at their source.
        inventory, allocation = units.copy(), weights.copy()
        inventory.setflags(write=False)
        allocation.setflags(write=False)
        return cls(session, date, nav, cash, inventory, allocation)


@dataclass(frozen=True)
class CloseDecision:
    weights: np.ndarray
    reason: str
    cap: float = 1.

    def validated_weights(self, size: int) -> np.ndarray:
        result = np.asarray(self.weights, dtype=float)
        if (result.shape != (size,) or not np.isfinite(result).all()
                or (result < 0).any() or not np.isfinite(self.cap)
                or not 0 <= self.cap <= 1
                or result.sum() > self.cap + 1e-10):
            raise ValueError('policy decision must have exact shape and finite cash-funded weights')
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError('policy decision needs an explicit reason')
        return result.copy()


class ClosePolicy(Protocol):
    def decide(self, close: CloseObservation) -> CloseDecision:
        """Use completed-session information and actual inventory only."""
        ...

    def identity(self) -> dict[str, Any]:
        """Return serializable policy/source/configuration/data provenance."""
        ...

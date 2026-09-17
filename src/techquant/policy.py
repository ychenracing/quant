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
    unit_targets: np.ndarray | None = None

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


    def validated_unit_targets(self, prices: np.ndarray, nav: float) -> np.ndarray | None:
        """Validate optional fixed inventory against the same close allocation.

        Weight-target policies keep their opening-NAV semantics. An inventory
        policy can instead request close-time units without changing fees,
        affordability, lots, liquidity or the next-session execution clock.
        """
        if self.unit_targets is None:
            return None
        result = np.asarray(self.unit_targets, dtype=float)
        marks = np.asarray(prices, dtype=float)
        if (result.shape != marks.shape or result.ndim != 1
                or not np.isfinite(result).all() or (result < 0).any()
                or not np.isfinite(nav) or nav <= 0
                or ((result > 0) & (~np.isfinite(marks) | (marks <= 0))).any()):
            raise ValueError('unit targets need finite nonnegative inventory and valid close marks')
        values = np.zeros_like(result)
        np.multiply(result, marks, out=values, where=result > 0)
        weights = self.validated_weights(len(result))
        if not np.allclose(values / nav, weights, rtol=1e-9, atol=1e-10):
            raise ValueError('unit targets must match the validated close allocation')
        return result.copy()


class ClosePolicy(Protocol):
    def decide(self, close: CloseObservation) -> CloseDecision:
        """Use completed-session information and actual inventory only."""
        ...

    def identity(self) -> dict[str, Any]:
        """Return serializable policy/source/configuration/data provenance."""
        ...

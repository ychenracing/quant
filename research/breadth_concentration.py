"""Causal regime ownership: concentrate with broad strength, diversify without it.

This is a standalone price-led owner.  It does not wrap or override another
policy's exits, and it never consumes reference-strategy outputs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from techquant.config import Config
from techquant.data import Market, file_hash
from techquant.policy import CloseDecision, CloseObservation


@dataclass(frozen=True)
class Parameters:
    """The preregistered family has one source-defined configuration."""


@dataclass(frozen=True)
class OwnershipSignals:
    price: np.ndarray
    ready: np.ndarray
    breadth: np.ndarray
    qualified: np.ndarray
    broken: np.ndarray
    score: np.ndarray


def build_signals(market: Market) -> OwnershipSignals:
    quoted = market.panel("close")
    active = quoted.notna() & market.panel("volume").gt(0)
    price = quoted.ffill()
    observed = active.cumsum()
    ready = active & observed.ge(60) & active.rolling(20, min_periods=20).mean().ge(0.8)
    ema20 = price.ewm(span=20, adjust=False).mean()
    ema60 = price.ewm(span=60, adjust=False).mean()
    first = quoted.where(active & observed.eq(1)).ffill()
    since = price / first
    score = sum(
        np.log((price / price.shift(horizon)).fillna(since))
        for horizon in (20, 60, 120)
    ) / 3
    score = score.where(ready, -np.inf).fillna(-np.inf)
    breadth = (
        (price.gt(ema20) & ready)
        .sum(axis=1)
        .div(ready.sum(axis=1).replace(0, np.nan))
        .fillna(0.0)
    )
    return20 = price.pct_change(20, fill_method=None)
    qualified = ready & price.gt(ema20) & ema20.gt(ema60) & score.gt(0)
    broken = (~ready) | (price.lt(ema60) & return20.lt(0))
    return OwnershipSignals(
        price=price.to_numpy(),
        ready=ready.to_numpy(),
        breadth=breadth.to_numpy(),
        qualified=qualified.to_numpy(),
        broken=broken.to_numpy(),
        score=score.to_numpy(),
    )


def select_members(breadth, qualified, score, symbols):
    qualified = np.asarray(qualified, dtype=bool)
    score = np.asarray(score, dtype=float)
    if qualified.ndim != 1 or score.shape != qualified.shape or len(symbols) != len(score):
        raise ValueError("selection inputs must describe one exact universe")
    if not np.isfinite(breadth) or not 0 <= breadth <= 1:
        raise ValueError("breadth must be a finite fraction")
    eligible = np.flatnonzero(qualified & np.isfinite(score))
    ranked = sorted(eligible, key=lambda j: (-score[j], symbols[j]))
    if breadth >= 0.5:
        ranked = ranked[:2]
    return tuple(sorted(ranked))


class Owner:
    def __init__(self, market: Market, parameters: Parameters):
        if type(parameters) is not Parameters:
            raise ValueError("breadth concentration requires its registered singleton")
        self.market = market
        self.parameters = parameters
        self.config = Config()
        self.signals = build_signals(market)
        self.mode = None
        self.selected = tuple()
        self.pending_membership = False
        self.last_session = -1

    def decide(self, observation: CloseObservation) -> CloseDecision:
        i = observation.session
        if i <= self.last_session:
            raise ValueError("policy sessions must increase")
        self.last_session = i
        signals = self.signals
        held = observation.units > 1e-10
        mode = "CONCENTRATED" if signals.breadth[i] >= 0.5 else "DIVERSIFIED"
        broken = held & signals.broken[i]
        actual_members = set(np.flatnonzero(held))
        expected_members = set(self.selected)
        if self.pending_membership and actual_members == expected_members:
            self.pending_membership = False

        scheduled = i % self.config.rebalance == 0
        review = self.mode is None or mode != self.mode or scheduled or bool(broken.any())
        reasons = []
        if review:
            qualified = signals.qualified[i] & ~signals.broken[i]
            self.selected = select_members(
                signals.breadth[i], qualified, signals.score[i], self.market.symbols
            )
            self.mode = mode
            self.pending_membership = True
            reasons.append("REGIME_OWNERSHIP_REVIEW")
        if broken.any():
            reasons.append("BROKEN_HOLDING_EXIT")

        if self.pending_membership:
            weights = np.zeros(len(self.market.symbols), dtype=float)
            if self.selected:
                weights[list(self.selected)] = 1.0 / len(self.selected)
            reasons.append(mode + "_TARGET")
            return CloseDecision(weights, "|".join(reasons), 1.0)

        # No mode or membership change: preserve actual funded economic units,
        # not yesterday's target weights.  This prevents routine winner trims.
        reasons.append(mode + "_RETAIN_ACTUAL_UNITS")
        return CloseDecision(
            observation.weights.copy(),
            "|".join(reasons),
            1.0,
            observation.units.copy(),
        )

    def identity(self):
        root = Path(__file__).parent
        return {
            "name": "breadth_concentration_ownership",
            "parameters": asdict(self.parameters),
            "implementation_sha256": file_hash(Path(__file__)),
            "contract_sha256": file_hash(root / "breadth_concentration_contract.json"),
            "data_sha256": self.market.fingerprint(),
            "reference_runtime_inputs": False,
            "status": "RESEARCH_NOT_ACCEPTED",
        }


__all__ = [
    "Owner",
    "OwnershipSignals",
    "Parameters",
    "build_signals",
    "select_members",
]

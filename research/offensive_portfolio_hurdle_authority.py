"""Portfolio-level admission hurdle for ordinary full-book displacement.

The current fresh-challenger owner remains authoritative for campaign lifecycle,
failed-fill recovery and execution.  This wrapper changes only the ordinary
full-book replacement authority: a challenger must clear the strongest causal
admission reference already funded by the book before it may liquidate the
weakest decayed campaign.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from techquant.data import Market, file_hash
from research.offensive_fresh_challenger_authority import (
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


class Owner:
    def __init__(self, market: Market, parameters: Parameters):
        if type(parameters) is not Parameters:
            raise ValueError("portfolio hurdle authority requires registered parameters")
        self.market = market
        self.parameters = parameters
        self.parent = ParentOwner(market, ParentParameters(parameters.enabled))

    @property
    def trace(self):
        return self.parent.trace

    def _event(self, observation, action: str, **extra) -> None:
        self.parent.base.trace.append({
            "kind": "PORTFOLIO_HURDLE_AUTHORITY_EVENT",
            "date": observation.date,
            "session": int(observation.session),
            "action": action,
            **extra,
        })

    def _ordinary_displacement_context(self, observation):
        b = self.parent.base
        i = observation.session
        held = observation.units > 1e-10
        if int(held.sum()) != b.config.max_positions or not b.trend.market[i]:
            return None

        p = b.price_signals
        score = b.features.score[i]
        broken = b.trend.exit[i] | (p.ret1[i] <= -0.08) | ~p.ready[i]
        if np.any(held & broken):
            return None
        retry = self.parent.invalidated & held & (i > self.parent.invalidated_since)
        if np.any(retry):
            return None

        references = b.owned_alpha_reference.copy()
        pending = held & ~np.isfinite(references) & np.isfinite(b.pending_alpha_reference)
        references[pending] = b.pending_alpha_reference[pending]
        finite_held = held & np.isfinite(references)
        if not np.any(finite_held):
            return None

        decay = score - references
        decayed = held & np.isfinite(decay) & (decay < 0)
        allowed = (
            p.ready[i]
            & b.trend.entry[i]
            & (p.price[i] > p.ema20[i])
            & (p.momentum5[i] > 0)
            & np.isfinite(score)
            & (score > 0)
            & ~held
            & ~broken
            & ~b.retired
            & ~self.parent.invalidated
        )
        challengers = np.flatnonzero(allowed)
        if not decayed.any() or not len(challengers):
            return None

        incumbent = min(
            np.flatnonzero(decayed),
            key=lambda j: (decay[j], self.market.symbols[j]),
        )
        challenger = min(
            challengers,
            key=lambda j: (-score[j], self.market.symbols[j]),
        )
        incumbent_reference = references[incumbent]
        if not np.isfinite(incumbent_reference) or score[challenger] <= incumbent_reference:
            return None

        hurdle = float(np.max(references[finite_held]))
        return {
            "incumbent": int(incumbent),
            "challenger": int(challenger),
            "challenger_score": float(score[challenger]),
            "incumbent_reference": float(incumbent_reference),
            "portfolio_hurdle": hurdle,
        }

    def _hold_without_ordinary_displacement(self, observation):
        b = self.parent.base
        i = observation.session
        original = b.trend
        market = original.market.copy()
        market[i] = False
        b.trend = replace(original, market=market)
        try:
            return self.parent.decide(observation)
        finally:
            b.trend = original

    def decide(self, observation):
        if not self.parameters.enabled:
            return self.parent.decide(observation)

        context = self._ordinary_displacement_context(observation)
        if context is None:
            return self.parent.decide(observation)

        incumbent = context["incumbent"]
        challenger = context["challenger"]
        if context["challenger_score"] <= context["portfolio_hurdle"]:
            decision = self._hold_without_ordinary_displacement(observation)
            self._event(
                observation,
                "PORTFOLIO_HURDLE_BLOCK",
                incumbent=self.market.symbols[incumbent],
                challenger=self.market.symbols[challenger],
                challenger_score=context["challenger_score"],
                incumbent_reference=context["incumbent_reference"],
                portfolio_hurdle=context["portfolio_hurdle"],
            )
            return decision

        before = len(self.parent.base.trace)
        decision = self.parent.decide(observation)
        displaced = any(
            row.get("kind") == "ALPHA_DECAY_DISPLACEMENT_EVENT"
            and row.get("action") == "ALPHA_DECAY_DISPLACEMENT"
            for row in self.parent.base.trace[before:]
        )
        if displaced:
            self._event(
                observation,
                "PORTFOLIO_HURDLE_AUTHORIZED",
                incumbent=self.market.symbols[incumbent],
                challenger=self.market.symbols[challenger],
                challenger_score=context["challenger_score"],
                incumbent_reference=context["incumbent_reference"],
                portfolio_hurdle=context["portfolio_hurdle"],
            )
        return decision

    def identity(self):
        root = Path(__file__).parent
        return {
            "name": "offensive_portfolio_hurdle_authority",
            "parameters": asdict(self.parameters),
            "implementation_sha256": file_hash(Path(__file__)),
            "contract_sha256": file_hash(root / "offensive_portfolio_hurdle_authority_contract.json"),
            "parent_sha256": file_hash(root / "offensive_fresh_challenger_authority.py"),
            "data_sha256": self.market.fingerprint(),
            "status": "RESEARCH_NOT_ACCEPTED",
        }


__all__ = ["Owner", "Parameters", "grid", "preserve_trace", "verify_trace"]

"""Restrict same-close liquidation authority of a brand-new trend edge.

A fresh trend-entry edge remains a normal cash-funded opportunity.  This wrapper
changes only ordinary full-book alpha-decay displacement: on the first entry-edge
close, the exact challenger that the current alpha leader would use cannot
liquidate an incumbent whose causal score has not fallen versus the prior close.
No state, delay, score multiplier, or cooldown is carried forward.
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
            raise ValueError("fresh edge authority requires registered parameters")
        self.market = market
        self.parameters = parameters
        self.parent = ParentOwner(market, ParentParameters(parameters.enabled))

    @property
    def trace(self):
        return self.parent.trace

    def _ordinary_pair(self, observation):
        base = self.parent.base
        i = observation.session
        if i <= 0:
            return None
        held = observation.units > 1e-10
        if int(held.sum()) != base.config.max_positions or not base.trend.market[i]:
            return None
        p = base.price_signals
        score = base.features.score[i]
        broken = base.trend.exit[i] | (p.ret1[i] <= -0.08) | ~p.ready[i]
        if np.any(held & broken) or np.any(held & base.retired):
            return None
        decay = score - base.owned_alpha_reference
        decayed = held & np.isfinite(decay) & (decay < 0)
        allowed = (
            p.ready[i]
            & base.trend.entry[i]
            & (p.price[i] > p.ema20[i])
            & (p.momentum5[i] > 0)
            & np.isfinite(score)
            & (score > 0)
            & ~held
            & ~broken
            & ~base.retired
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
        reference = base.owned_alpha_reference[incumbent]
        if not np.isfinite(reference) or score[challenger] <= reference:
            return None
        return incumbent, challenger

    def _block_same_close_fresh_edge(self, observation, original_decide):
        pair = self._ordinary_pair(observation)
        if pair is None:
            return original_decide(observation)
        base = self.parent.base
        i = observation.session
        incumbent, challenger = pair
        score = base.features.score[i]
        prior_score = base.features.score[i - 1, incumbent]
        fresh_edge = bool(base.trend.entry[i, challenger] and not base.trend.entry[i - 1, challenger])
        incumbent_not_falling = bool(
            np.isfinite(score[incumbent])
            and np.isfinite(prior_score)
            and score[incumbent] >= prior_score
        )
        if not (fresh_edge and incumbent_not_falling):
            return original_decide(observation)

        original_features = base.features
        masked = original_features.score.copy()
        masked[i, challenger] = np.nan
        base.features = replace(original_features, score=masked)
        self.parent.base.trace.append({
            "kind": "FRESH_EDGE_AUTHORITY_EVENT",
            "date": observation.date,
            "session": int(i),
            "action": "FRESH_EDGE_AUTHORITY_BLOCK",
            "incumbent": self.market.symbols[incumbent],
            "challenger": self.market.symbols[challenger],
            "incumbent_prior_score": float(prior_score),
            "incumbent_score": float(score[incumbent]),
            "challenger_score": float(score[challenger]),
        })
        try:
            # Only this fresh challenger loses same-close liquidation authority.
            # The unchanged base owner may still select another eligible,
            # established challenger from the same causal close.
            return original_decide(observation)
        finally:
            base.features = original_features

    def decide(self, observation):
        if not self.parameters.enabled:
            return self.parent.decide(observation)

        base = self.parent.base
        original_decide = base.decide

        def authority_aware_decide(current_observation):
            return self._block_same_close_fresh_edge(current_observation, original_decide)

        # ParentOwner retains all acute/rearm/failed-fill branches.  Only its
        # eventual call into ordinary alpha-decay ownership is intercepted.
        base.decide = authority_aware_decide
        try:
            return self.parent.decide(observation)
        finally:
            base.decide = original_decide

    def identity(self):
        root = Path(__file__).parent
        return {
            "name": "offensive_fresh_edge_authority",
            "parameters": asdict(self.parameters),
            "implementation_sha256": file_hash(Path(__file__)),
            "contract_sha256": file_hash(root / "offensive_fresh_edge_authority_contract.json"),
            "parent_sha256": file_hash(root / "offensive_fresh_challenger_authority.py"),
            "data_sha256": self.market.fingerprint(),
            "status": "RESEARCH_NOT_ACCEPTED",
        }


__all__ = ["Owner", "Parameters", "grid", "preserve_trace", "verify_trace"]

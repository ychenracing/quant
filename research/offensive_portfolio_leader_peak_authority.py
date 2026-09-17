"""Release a weaker funded peak only to a new portfolio causal leader.

This treatment refines the current dominant-peak alpha leader at one decision
boundary only.  When dominant-peak would release a weaker secondary campaign
that is sitting at its actual funded closing peak for a fresh challenger, the
challenger must also be strictly stronger than every actually held campaign by
the same current causal score.  Otherwise that challenger is masked for this
close and the incumbent keeps the same peak protection it would have received
under the campaign-peak mechanism.  No persistent latch, threshold, cooldown,
capacity, funding, entry, exit or execution rule is added.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from techquant.data import Market, file_hash
from research.offensive_dominant_peak_authority import (
    Owner as ParentOwner,
    Parameters as ParentParameters,
)
from research.quantity_obligation import preserve_trace, verify_trace


@dataclass(frozen=True)
class Parameters:
    enabled: bool

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("portfolio leader peak authority requires a boolean")


def grid():
    return [Parameters(False), Parameters(True)]


class Owner:
    def __init__(self, market: Market, parameters: Parameters):
        if type(parameters) is not Parameters:
            raise ValueError("portfolio leader peak authority requires registered parameters")
        self.market = market
        self.parameters = parameters
        # The exact dominant-peak leader is the control in both modes.  Enabled
        # mode only intercepts the sparse weaker-peak release boundary below.
        self.parent = ParentOwner(market, ParentParameters(True))

    @property
    def trace(self):
        return self.parent.trace

    @property
    def _base(self):
        return self.parent.parent.base

    def _candidate(self, observation):
        # Observe the actual funded peak before deciding this close, matching the
        # parent ordering.  Re-observation inside parent.decide is idempotent.
        self.parent._observe_peak(observation)
        pair = self.parent._ordinary_pair(observation)
        if pair is None:
            return None

        incumbent, challenger = pair
        b = self._base
        i = observation.session
        held = observation.units > 1e-10
        score = b.features.score[i]
        fresh_edge = bool(
            b.trend.entry[i, challenger]
            and (i == 0 or not b.trend.entry[i - 1, challenger])
        )
        at_peak = bool(
            np.isfinite(self.parent.campaign_peak_close[incumbent])
            and np.isfinite(b.price_signals.price[i, incumbent])
            and b.price_signals.price[i, incumbent]
            >= self.parent.campaign_peak_close[incumbent]
        )
        other = np.flatnonzero(held & (np.arange(len(held)) != incumbent))
        finite_other = [j for j in other if np.isfinite(score[j])]
        incumbent_dominant = not finite_other or all(
            score[incumbent] >= score[j] for j in finite_other
        )
        if not (at_peak and fresh_edge and not incumbent_dominant):
            return None

        finite_held = [j for j in np.flatnonzero(held) if np.isfinite(score[j])]
        challenger_is_leader = bool(
            np.isfinite(score[challenger])
            and all(score[challenger] > score[j] for j in finite_held)
        )
        return incumbent, challenger, finite_held, challenger_is_leader

    def decide(self, observation):
        if not self.parameters.enabled:
            return self.parent.decide(observation)

        candidate = self._candidate(observation)
        if candidate is None:
            return self.parent.decide(observation)

        incumbent, challenger, finite_held, challenger_is_leader = candidate
        b = self._base
        i = observation.session
        score = b.features.score[i]

        if challenger_is_leader:
            self.parent.parent.base.trace.append({
                "kind": "PORTFOLIO_LEADER_PEAK_AUTHORITY_EVENT",
                "date": observation.date,
                "session": int(i),
                "action": "PORTFOLIO_LEADER_WEAKER_PEAK_RELEASE",
                "symbol": self.market.symbols[incumbent],
                "challenger": self.market.symbols[challenger],
                "incumbent_score": float(score[incumbent]),
                "held_scores": {
                    self.market.symbols[j]: float(score[j]) for j in finite_held
                },
                "challenger_score": float(score[challenger]),
            })
            return self.parent.decide(observation)

        original = b.features
        masked = original.score.copy()
        masked[i, challenger] = np.nan
        b.features = replace(original, score=masked)
        try:
            decision = self.parent.decide(observation)
        finally:
            b.features = original
        self.parent.parent.base.trace.append({
            "kind": "PORTFOLIO_LEADER_PEAK_AUTHORITY_EVENT",
            "date": observation.date,
            "session": int(i),
            "action": "NONLEADER_CHALLENGER_PEAK_BLOCK",
            "symbol": self.market.symbols[incumbent],
            "challenger": self.market.symbols[challenger],
            "incumbent_score": float(score[incumbent]),
            "held_scores": {
                self.market.symbols[j]: float(score[j]) for j in finite_held
            },
            "challenger_score": float(score[challenger]),
        })
        return decision

    def identity(self):
        root = Path(__file__).parent
        return {
            "name": "offensive_portfolio_leader_peak_authority",
            "parameters": asdict(self.parameters),
            "implementation_sha256": file_hash(Path(__file__)),
            "contract_sha256": file_hash(
                root / "offensive_portfolio_leader_peak_authority_contract.json"
            ),
            "parent_sha256": file_hash(root / "offensive_dominant_peak_authority.py"),
            "data_sha256": self.market.fingerprint(),
            "status": "RESEARCH_NOT_ACCEPTED",
        }


__all__ = ["Owner", "Parameters", "grid", "preserve_trace", "verify_trace"]

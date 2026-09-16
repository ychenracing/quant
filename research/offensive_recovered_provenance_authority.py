"""Restrict ordinary full-book liquidation authority by campaign provenance.

Reference-rearmed opportunities remain ordinary cash-funded opportunities.  The
only change is that, while their existing causal provenance still says
"rearmed without a fresh epoch", they cannot use the parent's ordinary full-book
alpha-decay path to liquidate funded inventory.  The parent remains authoritative
for every lifecycle, execution, acute-exit and failed-fill decision.
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
            raise ValueError("recovered provenance authority requires registered parameters")
        self.market = market
        self.parameters = parameters
        self.parent = ParentOwner(market, ParentParameters(parameters.enabled))

    @property
    def trace(self):
        return self.parent.trace

    def _event(self, observation, symbols) -> None:
        self.parent.base.trace.append({
            "kind": "RECOVERED_PROVENANCE_AUTHORITY_EVENT",
            "date": observation.date,
            "session": int(observation.session),
            "action": "RECOVERED_PROVENANCE_AUTHORITY_BLOCK",
            "symbols": list(symbols),
        })

    def decide(self, observation):
        if not self.parameters.enabled:
            return self.parent.decide(observation)

        base = self.parent.base
        original_decide = base.decide

        def provenance_aware_base_decide(current_observation):
            held = current_observation.units > 1e-10
            if int(held.sum()) != base.config.max_positions:
                return original_decide(current_observation)

            # ParentOwner updates actual-establishment and fresh-epoch provenance
            # before it reaches base.decide.  Reading the state here therefore
            # blocks only campaigns that are still recovered-without-fresh-epoch
            # at the exact ordinary displacement decision point.
            blocked = self.parent.reference_rearmed_without_fresh_epoch & ~held
            if not np.any(blocked):
                return original_decide(current_observation)

            i = current_observation.session
            original_features = base.features
            scores = original_features.score.copy()
            scores[i, blocked] = np.nan
            base.features = replace(original_features, score=scores)
            self._event(current_observation, [self.market.symbols[j] for j in np.flatnonzero(blocked)])
            try:
                return original_decide(current_observation)
            finally:
                base.features = original_features

        # FreshChallengerAuthority has custom acute/rearm/failed-fill branches.
        # Only its eventual call into the ordinary base owner is intercepted;
        # custom parent branches never see this wrapper and remain byte-behavior
        # equivalent to the current alpha leader.
        base.decide = provenance_aware_base_decide
        try:
            return self.parent.decide(observation)
        finally:
            base.decide = original_decide

    def identity(self):
        root = Path(__file__).parent
        return {
            "name": "offensive_recovered_provenance_authority",
            "parameters": asdict(self.parameters),
            "implementation_sha256": file_hash(Path(__file__)),
            "contract_sha256": file_hash(root / "offensive_recovered_provenance_authority_contract.json"),
            "parent_sha256": file_hash(root / "offensive_fresh_challenger_authority.py"),
            "data_sha256": self.market.fingerprint(),
            "status": "RESEARCH_NOT_ACCEPTED",
        }


__all__ = ["Owner", "Parameters", "grid", "preserve_trace", "verify_trace"]

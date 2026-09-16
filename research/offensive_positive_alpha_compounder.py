"""Let funded campaigns compound while their causal alpha remains positive.

This wrapper changes only security-specific lifecycle authority.  The current
campaign-peak champion remains authoritative for entry, funding, rearm,
displacement and execution.  A generic slow-trend exit alone cannot liquidate
an actually funded campaign whose current causal alpha score is still positive.
Conversely, a finite non-positive causal score is an owned-campaign self-failure
even when the generic slow-trend edge has not fired yet.  Acute one-session loss
and missing/unready quotes remain untouched in the parent owner.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from techquant.data import Market, file_hash
from research.offensive_campaign_peak_authority import (
    Owner as ChampionOwner,
    Parameters as ChampionParameters,
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
            raise ValueError("positive alpha compounder requires registered parameters")
        self.market = market
        self.parameters = parameters
        # Disabled treatment is the exact current campaign-peak alpha champion,
        # not the historical trend_book baseline.
        self.parent = ChampionOwner(market, ChampionParameters(True))

    @property
    def trace(self):
        return self.parent.trace

    @property
    def _base(self):
        # campaign_peak.parent -> fresh_challenger; fresh_challenger.base ->
        # alpha_decay owner whose trend/score state drives security exits.
        return self.parent.parent.base

    def _event(self, observation, action: str, mask: np.ndarray) -> None:
        if not np.any(mask):
            return
        self._base.trace.append({
            "kind": "POSITIVE_ALPHA_COMPOUNDER_EVENT",
            "date": observation.date,
            "session": int(observation.session),
            "action": action,
            "symbols": [self.market.symbols[j] for j in np.flatnonzero(mask)],
        })

    def decide(self, observation):
        if not self.parameters.enabled:
            return self.parent.decide(observation)

        b = self._base
        i = observation.session
        held = observation.units > 1e-10
        score = b.features.score[i]
        p = b.price_signals
        original_trend = b.trend
        original_exit_row = original_trend.exit[i]

        finite = np.isfinite(score)
        positive = held & finite & (score > 0)
        nonpositive = held & finite & (score <= 0)

        # Only change the generic slow-trend component. Acute loss and quote
        # readiness are independent parent predicates and remain authoritative.
        effective_exit = original_trend.exit.copy()
        effective_exit[i, positive] = False
        effective_exit[i, nonpositive] = True

        acute = p.ret1[i] <= -0.08
        ready = p.ready[i]
        forgiven = positive & original_exit_row & ~acute & ready
        self_failed = nonpositive & ~acute & ready
        self._event(observation, "POSITIVE_ALPHA_TREND_EXIT_FORGIVEN", forgiven)
        self._event(observation, "NONPOSITIVE_ALPHA_SELF_FAILURE", self_failed)

        b.trend = replace(original_trend, exit=effective_exit)
        try:
            return self.parent.decide(observation)
        finally:
            b.trend = original_trend

    def identity(self):
        root = Path(__file__).parent
        return {
            "name": "offensive_positive_alpha_compounder",
            "parameters": asdict(self.parameters),
            "implementation_sha256": file_hash(Path(__file__)),
            "contract_sha256": file_hash(root / "offensive_positive_alpha_compounder_contract.json"),
            "parent_sha256": file_hash(root / "offensive_campaign_peak_authority.py"),
            "data_sha256": self.market.fingerprint(),
            "status": "RESEARCH_NOT_ACCEPTED",
        }


__all__ = ["Owner", "Parameters", "grid", "preserve_trace", "verify_trace"]

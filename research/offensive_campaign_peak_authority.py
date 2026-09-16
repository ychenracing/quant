"""Use actual funded campaign closing-peak state for same-close displacement authority.

A fresh trend-entry challenger remains normally cash-admissible.  The only change
is to ordinary full-book liquidation authority: if the selected incumbent is
still at the highest close observed since its actual campaign became funded, a
brand-new trend edge cannot liquidate it on that same decision close.  No price
buffer, waiting period, score multiplier, or post-hoc label is used.
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
            raise ValueError("campaign peak authority requires registered parameters")
        self.market = market
        self.parameters = parameters
        self.parent = ParentOwner(market, ParentParameters(parameters.enabled))
        n = len(market.symbols)
        self.actual_was_held = np.zeros(n, dtype=bool)
        self.campaign_peak_close = np.full(n, np.nan)

    @property
    def trace(self):
        return self.parent.trace

    def _observe_campaign_state(self, observation) -> None:
        i = observation.session
        held = observation.units > 1e-10
        price = self.parent.base.price_signals.price[i]
        fresh = held & ~self.actual_was_held
        continuing = held & self.actual_was_held
        valid_fresh = fresh & np.isfinite(price)
        self.campaign_peak_close[valid_fresh] = price[valid_fresh]
        invalid_fresh = fresh & ~np.isfinite(price)
        self.campaign_peak_close[invalid_fresh] = np.nan
        valid_continuing = continuing & np.isfinite(price)
        prior = self.campaign_peak_close[valid_continuing]
        current = price[valid_continuing]
        self.campaign_peak_close[valid_continuing] = np.where(
            np.isfinite(prior), np.maximum(prior, current), current
        )
        self.campaign_peak_close[~held] = np.nan
        self.actual_was_held = held.copy()

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
        incumbent = min(np.flatnonzero(decayed), key=lambda j: (decay[j], self.market.symbols[j]))
        challenger = min(challengers, key=lambda j: (-score[j], self.market.symbols[j]))
        reference = base.owned_alpha_reference[incumbent]
        if not np.isfinite(reference) or score[challenger] <= reference:
            return None
        return incumbent, challenger

    def _campaign_peak_aware_decide(self, observation, original_decide):
        pair = self._ordinary_pair(observation)
        if pair is None:
            return original_decide(observation)
        base = self.parent.base
        i = observation.session
        incumbent, challenger = pair
        current_price = base.price_signals.price[i, incumbent]
        peak = self.campaign_peak_close[incumbent]
        fresh_edge = bool(base.trend.entry[i, challenger] and not base.trend.entry[i - 1, challenger])
        at_campaign_peak = bool(np.isfinite(current_price) and np.isfinite(peak) and current_price >= peak)
        if not (fresh_edge and at_campaign_peak):
            return original_decide(observation)

        original_features = base.features
        score = original_features.score.copy()
        score[i, challenger] = np.nan
        base.features = replace(original_features, score=score)
        self.parent.base.trace.append({
            "kind": "CAMPAIGN_PEAK_AUTHORITY_EVENT",
            "date": observation.date,
            "session": int(i),
            "action": "CAMPAIGN_PEAK_AUTHORITY_BLOCK",
            "incumbent": self.market.symbols[incumbent],
            "challenger": self.market.symbols[challenger],
            "incumbent_close": float(current_price),
            "campaign_peak_close": float(peak),
        })
        try:
            return original_decide(observation)
        finally:
            base.features = original_features

    def decide(self, observation):
        if not self.parameters.enabled:
            return self.parent.decide(observation)

        # Inventory at this close is actual observed state.  Update the funded
        # campaign peak before any policy target for this close is computed.
        self._observe_campaign_state(observation)
        base = self.parent.base
        original_decide = base.decide

        def authority_aware_decide(current_observation):
            return self._campaign_peak_aware_decide(current_observation, original_decide)

        # ParentOwner keeps all acute/rearm/failed-fill lifecycle authority. Only
        # its final ordinary alpha-decay owner call is intercepted.
        base.decide = authority_aware_decide
        try:
            return self.parent.decide(observation)
        finally:
            base.decide = original_decide

    def identity(self):
        root = Path(__file__).parent
        return {
            "name": "offensive_campaign_peak_authority",
            "parameters": asdict(self.parameters),
            "implementation_sha256": file_hash(Path(__file__)),
            "contract_sha256": file_hash(root / "offensive_campaign_peak_authority_contract.json"),
            "parent_sha256": file_hash(root / "offensive_fresh_challenger_authority.py"),
            "data_sha256": self.market.fingerprint(),
            "status": "RESEARCH_NOT_ACCEPTED",
        }


__all__ = ["Owner", "Parameters", "grid", "preserve_trace", "verify_trace"]

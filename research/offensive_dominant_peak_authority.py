"""Protect only the stronger funded campaign when a fresh challenger hits a peak block.

This is the current campaign-peak mechanism with one structural refinement: an
incumbent at its funded closing peak may veto an ordinary fresh-edge displacement
only when it is also the stronger currently held campaign by the same causal
score.  A weaker secondary campaign cannot hide behind its local price peak.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from techquant.data import Market, file_hash
from research.offensive_campaign_peak_authority import Owner as ChampionOwner, Parameters as ChampionParameters
from research.offensive_fresh_challenger_authority import Owner as ParentOwner, Parameters as ParentParameters
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
            raise ValueError("dominant peak authority requires registered parameters")
        self.market = market
        self.parameters = parameters
        self.control = ChampionOwner(market, ChampionParameters(True))
        self.parent = ParentOwner(market, ParentParameters(True))
        n = len(market.symbols)
        self.actual_was_held = np.zeros(n, dtype=bool)
        self.campaign_peak_close = np.full(n, np.nan)

    @property
    def trace(self):
        return self.control.trace if not self.parameters.enabled else self.parent.trace

    @property
    def _base(self):
        return self.parent.base

    def _observe_peak(self, observation) -> None:
        held = observation.units > 1e-10
        close = self._base.price_signals.price[observation.session]
        started = held & ~self.actual_was_held
        continuing = held & self.actual_was_held
        self.campaign_peak_close[started] = close[started]
        finite = continuing & np.isfinite(close)
        self.campaign_peak_close[finite] = np.where(
            np.isfinite(self.campaign_peak_close[finite]),
            np.maximum(self.campaign_peak_close[finite], close[finite]),
            close[finite],
        )
        self.campaign_peak_close[~held] = np.nan
        self.actual_was_held = held.copy()

    def _ordinary_pair(self, observation):
        b = self._base
        i = observation.session
        held = observation.units > 1e-10
        if int(held.sum()) != b.config.max_positions or not b.trend.market[i]:
            return None
        p = b.price_signals
        score = b.features.score[i]
        broken = b.trend.exit[i] | (p.ret1[i] <= -0.08) | ~p.ready[i]
        if np.any(held & broken):
            return None
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
        decay = score - b.owned_alpha_reference
        decayed = held & np.isfinite(decay) & (decay < 0)
        challengers = np.flatnonzero(allowed)
        if not np.any(decayed) or not len(challengers):
            return None
        incumbent = min(np.flatnonzero(decayed), key=lambda j: (decay[j], self.market.symbols[j]))
        challenger = min(challengers, key=lambda j: (-score[j], self.market.symbols[j]))
        if not np.isfinite(b.owned_alpha_reference[incumbent]) or score[challenger] <= b.owned_alpha_reference[incumbent]:
            return None
        return incumbent, challenger

    def decide(self, observation):
        if not self.parameters.enabled:
            return self.control.decide(observation)

        self._observe_peak(observation)
        pair = self._ordinary_pair(observation)
        if pair is None:
            return self.parent.decide(observation)
        incumbent, challenger = pair
        b = self._base
        i = observation.session
        score = b.features.score[i]
        held = observation.units > 1e-10
        fresh_edge = bool(b.trend.entry[i, challenger] and (i == 0 or not b.trend.entry[i - 1, challenger]))
        at_peak = bool(
            np.isfinite(self.campaign_peak_close[incumbent])
            and np.isfinite(b.price_signals.price[i, incumbent])
            and b.price_signals.price[i, incumbent] >= self.campaign_peak_close[incumbent]
        )
        other = np.flatnonzero(held & (np.arange(len(held)) != incumbent))
        finite_other = [j for j in other if np.isfinite(score[j])]
        dominant = not finite_other or all(score[incumbent] >= score[j] for j in finite_other)

        if at_peak and fresh_edge and not dominant:
            self.parent.base.trace.append({
                "kind": "DOMINANT_PEAK_AUTHORITY_EVENT",
                "date": observation.date,
                "session": int(i),
                "action": "WEAKER_PEAK_RELEASE",
                "symbol": self.market.symbols[incumbent],
                "challenger": self.market.symbols[challenger],
                "incumbent_score": float(score[incumbent]),
                "other_held_scores": {self.market.symbols[j]: float(score[j]) for j in finite_other},
                "challenger_score": float(score[challenger]),
            })
            return self.parent.decide(observation)

        if not (at_peak and fresh_edge and dominant):
            return self.parent.decide(observation)

        original = b.features
        masked = original.score.copy()
        masked[i, challenger] = np.nan
        b.features = replace(original, score=masked)
        try:
            decision = self.parent.decide(observation)
        finally:
            b.features = original
        self.parent.base.trace.append({
            "kind": "DOMINANT_PEAK_AUTHORITY_EVENT",
            "date": observation.date,
            "session": int(i),
            "action": "DOMINANT_FUNDED_PEAK_DISPLACEMENT_BLOCK",
            "symbol": self.market.symbols[incumbent],
            "challenger": self.market.symbols[challenger],
            "incumbent_score": float(score[incumbent]),
            "other_held_scores": {self.market.symbols[j]: float(score[j]) for j in finite_other},
            "challenger_score": float(score[challenger]),
            "campaign_peak_close": float(self.campaign_peak_close[incumbent]),
        })
        return decision

    def identity(self):
        root = Path(__file__).parent
        return {
            "name": "offensive_dominant_peak_authority",
            "parameters": asdict(self.parameters),
            "implementation_sha256": file_hash(Path(__file__)),
            "contract_sha256": file_hash(root / "offensive_dominant_peak_authority_contract.json"),
            "fresh_parent_sha256": file_hash(root / "offensive_fresh_challenger_authority.py"),
            "control_sha256": file_hash(root / "offensive_campaign_peak_authority.py"),
            "data_sha256": self.market.fingerprint(),
            "status": "RESEARCH_NOT_ACCEPTED",
        }


__all__ = ["Owner", "Parameters", "grid", "preserve_trace", "verify_trace"]

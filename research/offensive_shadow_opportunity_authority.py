"""Parameter-free shadow authority for ordinary full-book displacement.

The current offensive owner remains authoritative for every lifecycle and
execution decision.  This wrapper changes only when a new challenger is allowed
to liquidate an already funded incumbent: the first executable pair is a causal
nomination, and authority arrives only after the same challenger strengthens
above its own observed nomination score while the same incumbent remains weak.
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
            raise ValueError("shadow opportunity authority requires registered parameters")
        self.market = market
        self.parameters = parameters
        self.parent = ParentOwner(market, ParentParameters(parameters.enabled))
        self.shadow_incumbent = -1
        self.shadow_challenger = -1
        self.shadow_reference = np.nan

    @property
    def trace(self):
        return self.parent.trace

    def _event(self, observation, action: str, **extra) -> None:
        self.parent.base.trace.append({
            "kind": "SHADOW_OPPORTUNITY_AUTHORITY_EVENT",
            "date": observation.date,
            "session": int(observation.session),
            "action": action,
            **extra,
        })

    def _clear_shadow(self, observation, *, record: bool) -> None:
        if self.shadow_incumbent >= 0 and record:
            self._event(
                observation,
                "SHADOW_OPPORTUNITY_RESET",
                incumbent=self.market.symbols[self.shadow_incumbent],
                challenger=self.market.symbols[self.shadow_challenger],
                shadow_reference=float(self.shadow_reference),
            )
        self.shadow_incumbent = -1
        self.shadow_challenger = -1
        self.shadow_reference = np.nan

    def _pair(self, observation):
        b = self.parent.base
        i = observation.session
        held = observation.units > 1e-10
        if int(held.sum()) != b.config.max_positions or not b.trend.market[i]:
            return None
        p = b.price_signals
        broken = b.trend.exit[i] | (p.ret1[i] <= -0.08) | ~p.ready[i]
        if np.any(held & broken):
            return None
        retry = self.parent.invalidated & held & (i > self.parent.invalidated_since)
        if np.any(retry):
            return None

        score = b.features.score[i]
        references = b.owned_alpha_reference.copy()
        pending = held & ~np.isfinite(references) & np.isfinite(b.pending_alpha_reference)
        references[pending] = b.pending_alpha_reference[pending]
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
        if not np.isfinite(references[incumbent]) or score[challenger] <= references[incumbent]:
            return None
        return incumbent, challenger, float(score[challenger])

    def _hold_without_ordinary_displacement(self, observation):
        """Let the parent reconcile all state while withholding this one authority.

        With a full, unbroken book the base owner only consults market authority
        for voluntary displacement.  Temporarily suppressing that boolean leaves
        acute exits, inventory reconciliation and campaign state unchanged while
        preventing a nomination close from becoming a liquidation close.
        """
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

        pair = self._pair(observation)
        if pair is None:
            self._clear_shadow(observation, record=True)
            return self.parent.decide(observation)

        incumbent, challenger, challenger_score = pair
        same = (
            incumbent == self.shadow_incumbent
            and challenger == self.shadow_challenger
            and np.isfinite(self.shadow_reference)
        )
        if not same:
            self._clear_shadow(observation, record=True)
            self.shadow_incumbent = incumbent
            self.shadow_challenger = challenger
            self.shadow_reference = challenger_score
            self._event(
                observation,
                "SHADOW_OPPORTUNITY_NOMINATION",
                incumbent=self.market.symbols[incumbent],
                challenger=self.market.symbols[challenger],
                shadow_reference=challenger_score,
            )
            return self._hold_without_ordinary_displacement(observation)

        if challenger_score <= self.shadow_reference:
            return self._hold_without_ordinary_displacement(observation)

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
                "SHADOW_AUTHORIZED_DISPLACEMENT",
                incumbent=self.market.symbols[incumbent],
                challenger=self.market.symbols[challenger],
                shadow_reference=float(self.shadow_reference),
                challenger_score=challenger_score,
            )
            self._clear_shadow(observation, record=False)
        return decision

    def identity(self):
        root = Path(__file__).parent
        return {
            "name": "offensive_shadow_opportunity_authority",
            "parameters": asdict(self.parameters),
            "implementation_sha256": file_hash(Path(__file__)),
            "contract_sha256": file_hash(root / "offensive_shadow_opportunity_authority_contract.json"),
            "parent_sha256": file_hash(root / "offensive_fresh_challenger_authority.py"),
            "data_sha256": self.market.fingerprint(),
            "status": "RESEARCH_NOT_ACCEPTED",
        }


__all__ = ["Owner", "Parameters", "grid", "preserve_trace", "verify_trace"]

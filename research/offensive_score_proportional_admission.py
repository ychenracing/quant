"""Allocate simultaneous offensive vacancy admissions by causal-score share.

The current campaign-peak champion remains authoritative for every selection,
lifecycle and execution decision.  This wrapper changes only one funding action:
when the unchanged owner requests multiple VACANCY_FILL campaigns on the same
close, the exact observed cash is split in direct proportion to their positive
causal scores instead of equally.  One-vacancy fills and all funded inventory are
unchanged; there is no continuous score rebalancing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from techquant.data import Market, file_hash
from techquant.policy import CloseDecision
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
            raise ValueError("score proportional admission requires registered parameters")
        self.market = market
        self.parameters = parameters
        self.parent = ChampionOwner(market, ChampionParameters(True))

    @property
    def trace(self):
        return self.parent.trace

    @property
    def _base(self):
        return self.parent.parent.base

    def decide(self, observation):
        before = len(self._base.trace)
        decision = self.parent.decide(observation)
        if not self.parameters.enabled or decision.unit_targets is None:
            return decision

        fills = [
            row for row in self._base.trace[before:]
            if row.get("kind") == "ALPHA_DECAY_DISPLACEMENT_EVENT"
            and row.get("action") == "VACANCY_FILL"
        ]
        if not fills:
            return decision
        names = fills[-1].get("symbols", [])
        if len(names) <= 1:
            return decision
        chosen = [self.market.symbols.index(name) for name in names]
        score = self._base.features.score[observation.session]
        scores = np.asarray([score[j] for j in chosen], dtype=float)
        if (not np.isfinite(scores).all()) or (scores <= 0).any():
            raise ValueError("simultaneous admission requires finite positive causal scores")
        total = float(scores.sum())
        marks = np.nan_to_num(self._base.price_signals.price[observation.session], nan=0.0)
        if any(marks[j] <= 0 for j in chosen):
            raise ValueError("simultaneous admission requires valid close marks")

        units = np.asarray(decision.unit_targets, dtype=float).copy()
        shares = scores / total
        for j, share in zip(chosen, shares):
            units[j] = float(observation.cash) * float(share) / marks[j]
        weights = np.divide(
            units * marks,
            observation.nav,
            out=np.zeros_like(units),
            where=observation.nav > 0,
        )
        total_value = float((units * marks).sum())
        if total_value > observation.nav and total_value <= observation.nav * (1.0 + 1e-12):
            j = chosen[-1]
            units[j] = max(0.0, units[j] - (total_value - observation.nav) / marks[j])
            weights = np.divide(
                units * marks,
                observation.nav,
                out=np.zeros_like(units),
                where=observation.nav > 0,
            )
        self._base.trace.append({
            "kind": "SCORE_PROPORTIONAL_ADMISSION_EVENT",
            "date": observation.date,
            "session": int(observation.session),
            "action": "SCORE_PROPORTIONAL_ADMISSION",
            "symbols": names,
            "scores": {name: float(score[j]) for name, j in zip(names, chosen)},
            "cash_shares": {name: float(share) for name, share in zip(names, shares)},
            "observed_cash": float(observation.cash),
        })
        return CloseDecision(
            weights,
            decision.reason + "|SCORE_PROPORTIONAL_ADMISSION",
            decision.cap,
            units,
        )

    def identity(self):
        root = Path(__file__).parent
        return {
            "name": "offensive_score_proportional_admission",
            "parameters": asdict(self.parameters),
            "implementation_sha256": file_hash(Path(__file__)),
            "contract_sha256": file_hash(root / "offensive_score_proportional_admission_contract.json"),
            "parent_sha256": file_hash(root / "offensive_campaign_peak_authority.py"),
            "data_sha256": self.market.fingerprint(),
            "status": "RESEARCH_NOT_ACCEPTED",
        }


__all__ = ["Owner", "Parameters", "grid", "preserve_trace", "verify_trace"]

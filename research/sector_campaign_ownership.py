"""Persistent ownership of one causally observed technology-sector campaign."""
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
class CampaignSignals:
    price: np.ndarray
    ready: np.ndarray
    security_qualified: np.ndarray
    sector_score: np.ndarray
    sector_eligible: np.ndarray


def build_signals(market: Market, sectors: tuple[str, ...]) -> CampaignSignals:
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
    score = score.where(ready)
    security_qualified = ready & price.gt(ema20) & ema20.gt(ema60) & score.gt(0)

    unique = tuple(sorted(set(sectors)))
    sector_score = np.full((len(price), len(unique)), -np.inf)
    sector_eligible = np.zeros((len(price), len(unique)), dtype=bool)
    sector_array = np.asarray(sectors, dtype=object)
    ready_values = ready.to_numpy(dtype=bool)
    score_values = score.to_numpy(dtype=float)
    above_values = price.gt(ema20).to_numpy(dtype=bool)
    qualified_values = security_qualified.to_numpy(dtype=bool)
    for k, sector in enumerate(unique):
        member = sector_array == sector
        for i in range(len(price)):
            current = member & ready_values[i]
            if not current.any():
                continue
            current_scores = score_values[i, current]
            current_scores = current_scores[np.isfinite(current_scores)]
            if not len(current_scores):
                continue
            sector_score[i, k] = float(np.median(current_scores))
            breadth = float(np.mean(above_values[i, current]))
            sector_eligible[i, k] = bool(
                sector_score[i, k] > 0
                and breadth >= 0.5
                and np.any(member & qualified_values[i])
            )
    return CampaignSignals(
        price=price.to_numpy(dtype=float),
        ready=ready_values,
        security_qualified=qualified_values,
        sector_score=sector_score,
        sector_eligible=sector_eligible,
    )


def select_sector(eligible, score, sector_names):
    eligible = np.asarray(eligible, dtype=bool)
    score = np.asarray(score, dtype=float)
    if eligible.ndim != 1 or score.shape != eligible.shape or len(sector_names) != len(score):
        raise ValueError("sector selection inputs must describe one exact universe")
    candidates = np.flatnonzero(eligible & np.isfinite(score))
    if not len(candidates):
        return None
    ranked = sorted(candidates, key=lambda k: (-score[k], sector_names[k]))
    return sector_names[ranked[0]]


class Owner:
    def __init__(
        self,
        market: Market,
        parameters: Parameters,
        *,
        sectors: tuple[str, ...] | None = None,
    ):
        if type(parameters) is not Parameters:
            raise ValueError("sector campaign requires its registered singleton")
        self.market = market
        self.parameters = parameters
        self.config = Config()
        self.sectors = tuple(
            sectors
            if sectors is not None
            else (market.sectors.get(symbol, "unknown") for symbol in market.symbols)
        )
        if len(self.sectors) != len(market.symbols):
            raise ValueError("sector labels must match the supplied universe")
        self.sector_names = tuple(sorted(set(self.sectors)))
        self.signals = build_signals(market, self.sectors)
        self.active_sector = None
        self.selected = tuple()
        self.pending_membership = False
        self.last_session = -1

    def _members(self, session):
        if self.active_sector is None:
            return tuple()
        sector = np.asarray(self.sectors, dtype=object) == self.active_sector
        qualified = sector & self.signals.security_qualified[session]
        return tuple(np.flatnonzero(qualified))

    def decide(self, observation: CloseObservation) -> CloseDecision:
        i = observation.session
        if i <= self.last_session:
            raise ValueError("policy sessions must increase")
        self.last_session = i
        held = observation.units > 1e-10
        actual_members = set(np.flatnonzero(held))
        if self.pending_membership and actual_members == set(self.selected):
            self.pending_membership = False

        sector_index = (
            self.sector_names.index(self.active_sector)
            if self.active_sector is not None
            else None
        )
        active_valid = bool(
            sector_index is not None
            and self.signals.sector_eligible[i, sector_index]
        )
        scheduled = i % self.config.rebalance == 0
        transition = self.active_sector is None or (scheduled and not active_valid)
        previous = self.active_sector
        if transition:
            self.active_sector = select_sector(
                self.signals.sector_eligible[i],
                self.signals.sector_score[i],
                self.sector_names,
            )
            self.selected = self._members(i)
            self.pending_membership = True

        reasons = []
        if previous != self.active_sector:
            reasons.append("SECTOR_CAMPAIGN_TRANSITION")
        stale = held & ~self.signals.ready[i]
        if stale.any():
            reasons.append("STALE_HOLDING_EXIT")

        if self.pending_membership:
            weights = np.zeros(len(self.market.symbols), dtype=float)
            if self.selected:
                weights[list(self.selected)] = 1.0 / len(self.selected)
            reasons.append("SECTOR_CAMPAIGN_TARGET")
            return CloseDecision(weights, "|".join(reasons), 1.0)

        units = observation.units.copy()
        units[stale] = 0.0
        weights = observation.weights.copy()
        weights[stale] = 0.0
        reasons.append("SECTOR_CAMPAIGN_RETAIN_ACTUAL_UNITS")
        return CloseDecision(weights, "|".join(reasons), 1.0, units)

    def identity(self):
        root = Path(__file__).parent
        return {
            "name": "sector_campaign_ownership",
            "parameters": asdict(self.parameters),
            "implementation_sha256": file_hash(Path(__file__)),
            "contract_sha256": file_hash(root / "sector_campaign_ownership_contract.json"),
            "data_sha256": self.market.fingerprint(),
            "reference_runtime_inputs": False,
            "status": "RESEARCH_NOT_ACCEPTED",
        }


__all__ = [
    "CampaignSignals",
    "Owner",
    "Parameters",
    "build_signals",
    "select_sector",
]

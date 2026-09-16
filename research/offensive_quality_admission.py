"""Trend-quality vacancy discovery with absolute-score campaign ownership."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from techquant.data import Market, file_hash
from research.offensive_alpha_decay_displacement import (
    Owner as CampaignOwner,
    Parameters as CampaignParameters,
)
from research.offensive_trend_quality import trend_quality
from research.quantity_obligation import preserve_trace, verify_trace


@dataclass(frozen=True)
class Parameters:
    enabled: bool

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('enabled must be boolean')


def grid():
    return [Parameters(False), Parameters(True)]


class Owner:
    def __init__(self, market: Market, parameters: Parameters):
        if type(parameters) is not Parameters:
            raise ValueError('offensive quality admission requires registered parameters')
        self.market = market
        self.parameters = parameters
        self.base = CampaignOwner(market, CampaignParameters(parameters.enabled))
        self.ownership_features = self.base.features
        self.ownership_score = self.ownership_features.score
        self.discovery_score = trend_quality(market) if parameters.enabled else None
        if parameters.enabled:
            self.discovery_features = replace(
                self.ownership_features, score=self.discovery_score
            )
        else:
            self.discovery_features = None

    @property
    def trace(self):
        return self.base.trace

    def decide(self, observation):
        if not self.parameters.enabled:
            return self.base.decide(observation)

        held = observation.units > 1e-10
        vacancy = max(0, self.base.config.max_positions - int(held.sum()))
        if vacancy > 0:
            admission_score = np.where(
                np.isfinite(self.ownership_score), self.discovery_score, np.nan
            )
            self.base.features = replace(self.discovery_features, score=admission_score)
        else:
            self.base.features = self.ownership_features
        decision = self.base.decide(observation)

        if vacancy > 0:
            added = decision.unit_targets > observation.units + 1e-10
            for j in np.flatnonzero(added):
                score = self.ownership_score[observation.session, j]
                if not np.isfinite(score):
                    raise AssertionError('admitted symbol requires finite ownership score')
                self.base.pending_alpha_reference[j] = score
        return decision

    def identity(self):
        root = Path(__file__).parent
        return {
            'name': 'offensive_quality_admission',
            'parameters': asdict(self.parameters),
            'implementation_sha256': file_hash(Path(__file__)),
            'contract_sha256': file_hash(root / 'offensive_quality_admission_contract.json'),
            'campaign_owner_sha256': file_hash(root / 'offensive_alpha_decay_displacement.py'),
            'trend_quality_sha256': file_hash(root / 'offensive_trend_quality.py'),
            'trend_book_sha256': file_hash(root / 'trend_book.py'),
            'data_sha256': self.market.fingerprint(),
            'status': 'RESEARCH_NOT_ACCEPTED',
        }


__all__ = ['Owner', 'Parameters', 'grid', 'preserve_trace', 'verify_trace']

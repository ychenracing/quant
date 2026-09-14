"""Explicit fixed-unit adapter for the independent research policies only."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from techquant.data import Market, file_hash
from techquant.policy import CloseDecision, CloseObservation, ClosePolicy


class UnitIntent:
    def __init__(self, market: Market, underlying: ClosePolicy):
        self.underlying = underlying
        self.prices = market.panel('close').ffill().to_numpy()

    def decide(self, observation: CloseObservation) -> CloseDecision:
        request = self.underlying.decide(observation)
        weights = request.validated_weights(len(observation.units))
        prices = self.prices[observation.session]
        units = np.zeros_like(weights)
        if ((weights > 0) & (~np.isfinite(prices) | (prices <= 0))).any():
            raise ValueError('positive inventory request requires an observed close')
        np.divide(weights * observation.nav, prices, out=units, where=weights > 0)
        return CloseDecision(weights, request.reason, request.cap, units)

    def identity(self) -> dict:
        return {'name': 'fixed_close_inventory',
                'implementation_sha256': file_hash(Path(__file__)),
                'underlying': self.underlying.identity(), 'status': 'RESEARCH_NOT_ACCEPTED'}

"""Independent observed-trend authority for otherwise unchanged return forecasts.

A positive forecast is not permission to disregard a persistently broken price
trend. The new-risk gate is asymmetric: it can block purchases, never liquidate
an intact holding solely because the broad universe weakens.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import itertools
from pathlib import Path
import numpy as np
from techquant.data import Market, file_hash
from techquant.policy import CloseDecision, CloseObservation
from research.nonlinear import Owner as Learner, Parameters as ForecastParameters


@dataclass(frozen=True)
class Parameters:
    trend_span: int = 60
    require_market_trend: bool = False

    def __post_init__(self):
        if isinstance(self.trend_span, bool) or not isinstance(self.trend_span, int) or self.trend_span < 1:
            raise ValueError('trend span must be a positive integer')
        if not isinstance(self.require_market_trend, bool):
            raise ValueError('market trend requirement must be boolean')


def grid():
    return [Parameters(*values) for values in itertools.product((20, 60), (False, True))]


@dataclass(frozen=True)
class Signals:
    exit: np.ndarray
    entry: np.ndarray
    market: np.ndarray


def signals(market: Market, p: Parameters) -> Signals:
    quote = market.panel('close')
    active = quote.notna() & market.panel('volume').gt(0)
    price = quote.ffill()
    trend = price.ewm(span=p.trend_span, adjust=False).mean()
    below = price.lt(trend) & active
    broken = below.rolling(2, min_periods=2).sum().eq(2) & price.pct_change(20, fill_method=None).lt(0)
    entry = active & price.gt(trend) & trend.gt(trend.shift(5))
    fresh_returns = price.pct_change(fill_method=None).where(active & active.shift(fill_value=False))
    index = (1. + fresh_returns.mean(axis=1).fillna(0.)).cumprod()
    broad = index.ewm(span=60, adjust=False).mean()
    admitted = index.gt(broad) & broad.gt(broad.shift(5)) & fresh_returns.notna().any(axis=1)
    return Signals(broken.to_numpy(), entry.to_numpy(), admitted.to_numpy())


class Owner:
    def __init__(self, market: Market, p: Parameters):
        self.market, self.params = market, p
        self.inner = Learner(market, ForecastParameters(horizon=60, tail_threshold=.5, positions=4))
        self.s = signals(market, p)

    def decide(self, o: CloseObservation) -> CloseDecision:
        observed_exit = self.s.exit[o.session] & (o.units > 1e-10)
        # Feed the observed risk into the same full-exit latch before decisions:
        # unfilled sales remain pending and anticipated proceeds cannot buy names.
        self.inner.exit_pending |= observed_exit
        request = self.inner.decide(o)
        allowed = self.s.entry[o.session].copy()
        if self.params.require_market_trend and not self.s.market[o.session]:
            allowed[:] = False
        units = request.unit_targets.copy()
        denied = ~allowed & (units > o.units)
        units[denied] = o.units[denied]
        weights = request.weights.copy()
        weights[denied] = o.weights[denied]
        reasons = [request.reason]
        if observed_exit.any():
            reasons.append('OBSERVED_TREND_EXIT')
        if denied.any():
            reasons.append('WAIT_FOR_OBSERVED_RISING_TREND')
        return CloseDecision(weights, '|'.join(reasons), request.cap, units)

    def identity(self):
        return {'name': 'observed_trend_authority', 'parameters': asdict(self.params),
                'implementation_sha256': file_hash(Path(__file__)),
                'contract_sha256': file_hash(Path(__file__).with_name('observed_trend_contract.json')),
                'inner': self.inner.identity(), 'status': 'RESEARCH_NOT_ACCEPTED'}

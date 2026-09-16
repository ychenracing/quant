"""Deterministic causal trend-quality discovery over the offensive campaign owner."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from techquant.config import Config
from techquant.data import Market, file_hash
from research.offensive_alpha_decay_displacement import (
    Owner as CampaignOwner,
    Parameters as CampaignParameters,
)
from research.quantity_obligation import preserve_trace, verify_trace
from research.trend_book import price_signals


@dataclass(frozen=True)
class Parameters:
    enabled: bool

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('enabled must be boolean')


def grid():
    return [Parameters(False), Parameters(True)]


def _ordinal_percentiles(values: np.ndarray, eligible: np.ndarray,
                         symbols: tuple[str, ...]) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    idx = [j for j in range(len(values)) if eligible[j] and np.isfinite(values[j])]
    if not idx:
        return out
    order = sorted(idx, key=lambda j: (float(values[j]), symbols[j]))
    scale = float(len(order))
    for rank, j in enumerate(order, start=1):
        out[j] = rank / scale
    return out


def trend_quality(market: Market) -> np.ndarray:
    """Causal cross-sectional path quality using only closes through each session."""
    cfg = Config()
    quoted = market.panel('close')
    volume = market.panel('volume')
    active = quoted.notna() & volume.gt(0)
    close = quoted.ffill()
    log_close = np.log(close)
    log_ret = log_close.diff()

    progress = (log_close - log_close.shift(cfg.slow)).clip(lower=0.0)
    path = log_ret.abs().rolling(cfg.slow, min_periods=cfg.slow).sum()
    efficiency = progress.div(path.where(path > 0))

    prior_high = close.shift(1).rolling(cfg.slow, min_periods=cfg.slow).max()
    follow_through = close.div(prior_high)

    p = price_signals(market)
    ema20 = pd.DataFrame(p.ema20, index=close.index, columns=close.columns)
    persistence = (close > ema20).rolling(cfg.fast, min_periods=cfg.fast).mean()
    ready = pd.DataFrame(p.ready, index=close.index, columns=close.columns) & active

    components = [
        efficiency.to_numpy(dtype=float),
        follow_through.to_numpy(dtype=float),
        persistence.to_numpy(dtype=float),
    ]
    ready_values = ready.to_numpy(dtype=bool)
    quality = np.full(close.shape, np.nan, dtype=float)
    symbols = market.symbols
    for i in range(len(close.index)):
        pct = [
            _ordinal_percentiles(component[i], ready_values[i], symbols)
            for component in components
        ]
        stack = np.vstack(pct)
        valid = np.isfinite(stack).all(axis=0)
        if valid.any():
            quality[i, valid] = np.cbrt(np.prod(stack[:, valid], axis=0))
    return quality


class Owner:
    def __init__(self, market: Market, parameters: Parameters):
        if type(parameters) is not Parameters:
            raise ValueError('offensive trend quality requires registered parameters')
        self.market = market
        self.parameters = parameters
        self.base = CampaignOwner(market, CampaignParameters(parameters.enabled))
        self.quality = trend_quality(market) if parameters.enabled else None
        if parameters.enabled:
            self.base.features = replace(self.base.features, score=self.quality)

    @property
    def trace(self):
        return self.base.trace

    def decide(self, observation):
        return self.base.decide(observation)

    def identity(self):
        root = Path(__file__).parent
        return {
            'name': 'offensive_trend_quality',
            'parameters': asdict(self.parameters),
            'implementation_sha256': file_hash(Path(__file__)),
            'contract_sha256': file_hash(root / 'offensive_trend_quality_contract.json'),
            'campaign_owner_sha256': file_hash(root / 'offensive_alpha_decay_displacement.py'),
            'trend_book_sha256': file_hash(root / 'trend_book.py'),
            'data_sha256': self.market.fingerprint(),
            'status': 'RESEARCH_NOT_ACCEPTED',
        }


__all__ = ['Owner', 'Parameters', 'grid', 'trend_quality', 'preserve_trace', 'verify_trace']

"""Absolute causal leadership strength over the offensive campaign owner."""
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


def absolute_leadership_strength(market: Market) -> np.ndarray:
    """Causal trend strength in return units, suitable for cross-time comparison."""
    cfg = Config()
    quoted = market.panel('close')
    volume = market.panel('volume')
    active = quoted.notna() & volume.gt(0)
    close = quoted.ffill()
    log_close = np.log(close)
    log_ret = log_close.diff()

    short_progress = log_close - log_close.shift(cfg.fast)
    slow_progress = log_close - log_close.shift(cfg.slow)
    path = log_ret.abs().rolling(cfg.slow, min_periods=cfg.slow).sum()
    efficiency = slow_progress.clip(lower=0.0).div(path.where(path > 0))

    p = price_signals(market)
    ready = pd.DataFrame(p.ready, index=close.index, columns=close.columns) & active
    ready_slow = slow_progress.where(ready)
    market_slow = ready_slow.median(axis=1, skipna=True)
    slow_excess = slow_progress.sub(market_slow, axis=0)

    positive_short = short_progress.where(short_progress > 0)
    positive_excess = slow_excess.where(slow_excess > 0)
    core = np.sqrt(positive_short * positive_excess)

    ema20 = pd.DataFrame(p.ema20, index=close.index, columns=close.columns)
    persistence = (close > ema20).rolling(cfg.fast, min_periods=cfg.fast).mean()
    prior_high = close.shift(1).rolling(cfg.slow, min_periods=cfg.slow).max()
    follow_through = close.div(prior_high)

    strength = core * efficiency * persistence * follow_through
    valid = ready & np.isfinite(strength) & (strength > 0)
    return strength.where(valid).to_numpy(dtype=float)


class Owner:
    def __init__(self, market: Market, parameters: Parameters):
        if type(parameters) is not Parameters:
            raise ValueError('offensive absolute leadership requires registered parameters')
        self.market = market
        self.parameters = parameters
        self.base = CampaignOwner(market, CampaignParameters(parameters.enabled))
        self.strength = absolute_leadership_strength(market) if parameters.enabled else None
        if parameters.enabled:
            self.base.features = replace(self.base.features, score=self.strength)

    @property
    def trace(self):
        return self.base.trace

    def decide(self, observation):
        return self.base.decide(observation)

    def identity(self):
        root = Path(__file__).parent
        return {
            'name': 'offensive_absolute_leadership',
            'parameters': asdict(self.parameters),
            'implementation_sha256': file_hash(Path(__file__)),
            'contract_sha256': file_hash(root / 'offensive_absolute_leadership_contract.json'),
            'campaign_owner_sha256': file_hash(root / 'offensive_alpha_decay_displacement.py'),
            'trend_book_sha256': file_hash(root / 'trend_book.py'),
            'data_sha256': self.market.fingerprint(),
            'status': 'RESEARCH_NOT_ACCEPTED',
        }


__all__ = [
    'Owner', 'Parameters', 'absolute_leadership_strength', 'grid',
    'preserve_trace', 'verify_trace',
]

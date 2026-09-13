"""One global parameter set. No per-stock profiles or calendar-date alpha rules."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Config:
    # Signal/portfolio controls; these defaults are not claims of optimality.
    fast: int = 20
    slow: int = 60
    max_positions: int = 3
    rebalance: int = 10
    single_cap: float = .55
    sector_cap: float = .75
    trade_band: float = .06
    stop_atr: float = 4.5
    target_vol: float = .50
    shock_z: float = 2.5
    risk_drawdown: float = .12
    recovery: int = 3
    # Execution assumptions are separate from alpha and recorded in every run.
    initial_cash: float = 2_000_000.
    commission_bps: float = 2.5
    slippage_bps: float = 10.
    max_adv: float = .005

    def __post_init__(self) -> None:
        for name in ('fast', 'slow', 'max_positions', 'rebalance', 'recovery'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.fast < 5 or self.slow <= self.fast:
            raise ValueError('require 5 <= fast < slow')
        for name, value in asdict(self).items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{name} must be finite')
        for name in ('single_cap', 'sector_cap', 'target_vol', 'risk_drawdown', 'max_adv'):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f'{name} must lie in (0, 1]')
        if not 0 <= self.trade_band < self.single_cap:
            raise ValueError('require 0 <= trade_band < single_cap')
        if self.initial_cash <= 0 or self.stop_atr <= 0 or self.shock_z <= 0:
            raise ValueError('cash and risk multipliers must be positive')
        if min(self.commission_bps, self.slippage_bps) < 0:
            raise ValueError('costs must be nonnegative')

"""Funded ownership with persistent, actual-inventory protective intent.

A fresh breakout starts a small cash-funded position. Additions require profit on
observed acquisition cost, not a hypothetical signal or an unfilled order. The
research adapter uses the existing close-decision/next-open execution engine;
none of its stops claim execution at a historical peak or an intraday price.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import techquant.engine as engine
from techquant.config import Config
from techquant.data import file_hash


@dataclass(frozen=True)
class Parameters:
    entry_window: int = 40
    initial_fraction: float = .20
    trail_atr: float = 3.5
    distribution_guard: bool = True

    def __post_init__(self):
        if isinstance(self.entry_window, bool) or not isinstance(self.entry_window, int) or self.entry_window < 1:
            raise ValueError('entry_window must be a positive integer')
        for name in ('initial_fraction', 'trail_atr'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if self.initial_fraction > .60 or type(self.distribution_guard) is not bool:
            raise ValueError('invalid initial funding or distribution switch')


def regression_score(close: pd.DataFrame, window: int = 60) -> np.ndarray:
    """Trailing log-price slope times R-squared; no retrospective pivot fitting."""
    values = np.log(close.to_numpy())
    result = np.full(values.shape, -np.inf)
    x = np.arange(window, dtype=float)
    x -= x.mean()
    denominator = float(x @ x)
    # One row at a time also makes prefix equality independent of BLAS blocking.
    for i in range(window - 1, len(values)):
        y = values[i - window + 1:i + 1]
        centered = y - np.mean(y, axis=0)
        covariance = np.sum(x[:, None] * centered, axis=0)
        variance = np.sum(centered * centered, axis=0)
        r_squared = np.divide(covariance ** 2, denominator * variance,
                              out=np.zeros_like(variance), where=variance > 1e-20)
        result[i] = covariance / denominator * np.clip(r_squared, 0, 1)
    return np.nan_to_num(result, nan=-np.inf, neginf=-np.inf, posinf=-np.inf)


class Policy:
    def __init__(self, parameters: Parameters, cost_multiplier: float = 1.):
        self.parameters = parameters
        self.cost_multiplier = cost_multiplier
        self.nav = 0.
        self.pause_until = -1

    def build(self, market, _config):
        p = self.parameters
        quoted = market.panel('close')
        close = quoted.ffill()
        volume = market.panel('volume')
        active = quoted.notna() & volume.gt(0)
        ready = active & active.cumsum().ge(60)
        ready &= active.rolling(10, min_periods=10).mean().ge(.8)
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema60 = close.ewm(span=60, adjust=False).mean()
        ret5 = close / close.shift(5) - 1
        prior_high = close.shift().rolling(p.entry_window, min_periods=p.entry_window).max()
        fresh = close.ge(prior_high) & ready
        score = regression_score(close)
        entry = fresh & close.gt(ema20) & ema20.gt(ema60) & close.gt(close.shift(60))
        entry &= pd.DataFrame(score > 0, index=close.index, columns=close.columns)
        high, low = market.panel('high'), market.panel('low')
        previous = close.shift()
        tr = np.maximum(high - low, np.maximum(abs(high - previous), abs(low - previous)))
        atr = tr.rolling(20, min_periods=20).mean().shift()
        prior_volume = volume.rolling(20, min_periods=20).mean().shift()
        distribution = ((close / previous - 1).lt(-.03) & volume.gt(1.5 * prior_volume)
                        & (close - low).le(.3 * (high - low)) & close.gt(1.3 * ema60))
        n = len(market.symbols)
        self.units = np.zeros(n)
        self.basis = np.zeros(n)
        self.peak = np.zeros(n)
        self.ratchet = np.zeros(n)
        self.adds = np.zeros(n, dtype=int)
        self.last_add = np.full(n, -100, dtype=int)
        self.exit_latched = np.zeros(n, dtype=bool)
        self.warned = np.zeros(n, dtype=bool)
        self.reduction_goal = np.full(n, np.inf)
        return SimpleNamespace(
            close=close.to_numpy(), opening=market.panel('open').to_numpy(),
            active=active.to_numpy(), ready=ready.to_numpy(), entry=entry.to_numpy(),
            score=score, atr=atr.to_numpy(), distribution=distribution.to_numpy(),
            renewed=close.ge(close.shift().rolling(10, min_periods=10).max()).to_numpy(),
            trend_broken=(close.lt(.98 * ema60) & ret5.lt(0)).to_numpy(),
            breadth=(close.gt(ema20) & ready).sum(axis=1).div(ready.sum(axis=1).replace(0, np.nan)).fillna(0).to_numpy())

    def update(self, i, f, history, config):
        self.nav = float(history[-1])
        return 1., 'FUNDED_TREND_OWNERSHIP'

    def _observe_fills(self, i, f, current, config):
        """Infer inventory from actual marks, never from the previous target."""
        actual = np.divide(current * self.nav, f.close[i], out=np.zeros_like(current),
                           where=np.isfinite(f.close[i]) & (f.close[i] > 0))
        tolerance = np.maximum(1e-7, np.abs(self.units) * 1e-10)
        acquired = actual > self.units + tolerance
        for j in np.flatnonzero(acquired):
            price = f.opening[i, j] * (1 + config.slippage_bps / 10_000 * self.cost_multiplier)
            if not np.isfinite(price) or price <= 0:
                raise ValueError('observed unit acquisition lacks a valid known opening price')
            added = actual[j] - self.units[j]
            if self.units[j] <= tolerance[j]:
                self.basis[j] = price
                self.peak[j] = max(price, f.close[i, j])
                self.ratchet[j] = price * .92
                self.adds[j] = 0
                self.warned[j] = False
                self.reduction_goal[j] = np.inf
            else:
                self.basis[j] = (self.basis[j] * self.units[j] + added * price) / actual[j]
                self.adds[j] += 1
            self.last_add[j] = i
        empty = actual <= tolerance
        self.basis[empty] = self.peak[empty] = self.ratchet[empty] = 0.
        self.adds[empty] = 0
        self.exit_latched[empty] = False
        self.warned[empty] = False
        self.reduction_goal[empty] = np.inf
        self.units = actual

    def weights(self, i, f, current, config, **_kwargs):
        current = np.asarray(current, dtype=float)
        if self.nav <= 0 or not np.isfinite(current).all() or (current < 0).any() or current.sum() > 1 + 1e-8:
            raise ValueError('invalid observed funded portfolio')
        self._observe_fills(i, f, current, config)
        held = self.units > 1e-7
        self.peak[held] = np.maximum(self.peak[held], f.close[i, held])
        proposed_stop = np.maximum(self.basis * .92, self.peak - self.parameters.trail_atr * f.atr[i])
        self.ratchet[held] = np.maximum(self.ratchet[held], np.nan_to_num(proposed_stop[held], nan=0.))
        damaged = held & (~f.active[i] | (f.close[i] <= self.ratchet) | f.trend_broken[i])
        new_exit = damaged & ~self.exit_latched
        if new_exit.any():
            self.exit_latched |= new_exit
            self.pause_until = max(self.pause_until, i + 5)
        want = current.copy()
        want[self.exit_latched] = 0.
        reasons = []
        if self.exit_latched.any():
            reasons.append('OWNED_PRICE_DAMAGE_LATCHED_EXIT')
        # Distribution trims are absolute unit goals. A blocked/partial sell must
        # not silently cancel the order or repeatedly halve the surviving units.
        achieved = self.units <= self.reduction_goal + 1e-7
        self.warned[held & f.renewed[i] & achieved] = False
        self.reduction_goal[achieved] = np.inf
        warning = held & ~self.exit_latched & ~self.warned & f.distribution[i] & self.parameters.distribution_guard
        if warning.any():
            self.reduction_goal[warning] = self.units[warning] * .5
            self.warned[warning] = True
            self.pause_until = max(self.pause_until, i + 5)
            reasons.append('VOLUME_SUPPORTED_DISTRIBUTION_REDUCTION')
        pending_trim = held & ~self.exit_latched & np.isfinite(self.reduction_goal)
        want[pending_trim] = np.minimum(want[pending_trim], self.reduction_goal[pending_trim] * f.close[i, pending_trim] / self.nav)
        entry_cap, drift_cap = max(.60, 1 / len(current)), max(.80, 1 / len(current))
        concentration = want > drift_cap + 1e-10
        if concentration.any():
            want[concentration] = entry_cap
            reasons.append('WIDE_BAND_CONCENTRATION_REDUCTION')
        # Cash from a target sale is not yet owned cash. Only observed free cash
        # can pay for an admission; protection takes priority for five sessions.
        available = max(0., 1 - float(current.sum()))
        if i < self.pause_until or self.exit_latched.any():
            return want, reasons or ['PROTECTIVE_NEW_MONEY_PAUSE']
        addable = held & f.entry[i] & ~pending_trim & (self.adds < 2) & ((i - self.last_add) >= 5)
        addable &= f.close[i] >= self.basis * 1.08
        for j in sorted(np.flatnonzero(addable), key=lambda j: (-f.score[i, j], j)):
            add = min(.15, available, max(0., entry_cap - want[j]))
            if add >= .01:
                want[j] += add
                available -= add
                reasons.append('OBSERVED_PROFIT_FUNDED_ADD')
        capacity = max(0, 4 - int(held.sum()))
        candidates = sorted(np.flatnonzero(~held & f.entry[i]), key=lambda j: (-f.score[i, j], j))
        for j in candidates[:capacity]:
            add = min(self.parameters.initial_fraction, available, entry_cap)
            if add < .01:
                break
            want[j] = add
            available -= add
            reasons.append('FRESH_BREAKOUT_SMALL_FUNDED_ENTRY')
        return want, reasons or ['RETAIN_ACTUAL_ECONOMIC_UNITS']


def run(market, parameters=None, **kwargs):
    p = parameters or Parameters()
    policy = Policy(p, kwargs.get('cost_multiplier', 1.))
    cfg = replace(Config(), max_positions=4, rebalance=20, single_cap=.6, sector_cap=1.)
    with patch.multiple(engine, build_features=policy.build, RiskState=lambda: policy,
                        target_weights=policy.weights):
        result = engine.run(market, cfg, **kwargs)
    result.metadata['algorithm'] = {
        'name': 'funded_trend_ownership', 'parameters': asdict(p),
        'adapter_sha256': file_hash(Path(__file__)), 'status': 'RESEARCH_NOT_ACCEPTED'}
    return result

"""Vectorized causal features; no negative shifts, centered windows or full-sample ranks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .data import Market


@dataclass(frozen=True)
class Features:
    symbols: tuple[str, ...]
    sectors: tuple[str, ...]
    close: np.ndarray
    ready: np.ndarray
    score: np.ndarray
    entry: np.ndarray
    exit: np.ndarray
    vol: np.ndarray
    breadth: np.ndarray
    market_return: np.ndarray
    market_vol: np.ndarray
    market_dd: np.ndarray
    weak: np.ndarray
    shock_fraction: np.ndarray


def build_features(market: Market, config: Config) -> Features:
    quoted = market.panel('close')
    volume = market.panel('volume')
    active = quoted.notna() & volume.gt(0)
    close = quoted.ffill()  # marks over gaps, but readiness below still requires a fresh quote.
    returns = close.pct_change(fill_method=None)
    fast = close.rolling(config.fast, min_periods=config.fast).mean()
    # Grow the slow window from the first fast observations. This is not
    # backfilling pre-IPO history: each anchor only exists after its first quote.
    slow = close.rolling(config.slow, min_periods=config.fast).mean()
    count = active.cumsum()
    ready = active & (count >= config.fast) & (
        active.rolling(config.fast, min_periods=config.fast).mean() >= .8)
    r_fast = close / close.shift(config.fast) - 1
    anchor = quoted.where(active & (count == 1)).ffill()
    since_observed = close / anchor - 1
    r_slow = (close / close.shift(config.slow) - 1).fillna(since_observed)
    r_long = (close / close.shift(4 * config.slow) - 1).fillna(since_observed)
    vol = returns.rolling(config.fast, min_periods=config.fast).std(ddof=0).clip(lower=.008)
    tr = pd.DataFrame(np.maximum.reduce([
        (market.panel('high') - market.panel('low')).to_numpy(),
        (market.panel('high') - close.shift()).abs().to_numpy(),
        (market.panel('low') - close.shift()).abs().to_numpy()]),
        index=close.index, columns=close.columns)
    atr = tr.rolling(config.fast, min_periods=config.fast).mean()
    peak = close.rolling(config.slow, min_periods=1).max()
    breakout = close >= close.shift().rolling(config.fast).max()
    persistence = (close > fast).rolling(5).mean() >= .8
    entry = ready & (fast > slow) & (close > fast) & (r_fast > 0) & (breakout | persistence)
    # A trailing violation needs fast-trend damage; an abrupt stock shock does not
    # wait for a slow crossover. All decisions are still made after the close.
    exit_signal = ((close < slow) & (r_fast < 0)) | (
        (peak - close > config.stop_atr * atr) & (close < fast)) | (
        (returns < -np.maximum(.08, config.shock_z * vol.shift())) & (close < fast))
    score = (.2 * np.log1p(r_fast) + .3 * np.log1p(r_slow) +
             .5 * np.log1p(r_long)) / np.sqrt(vol)
    score = score.where(ready, -np.inf).fillna(-np.inf)
    # No fixed leader basket: removal tests remove a name from signals as well.
    breadth = ((close > fast) & ready).sum(axis=1).div(ready.sum(axis=1).replace(0, np.nan)).fillna(0)
    observed_returns = returns.where(active & active.shift(fill_value=False))
    market_return = observed_returns.mean(axis=1).fillna(0)
    index = (1 + market_return).cumprod()
    market_vol = market_return.rolling(config.fast, min_periods=5).std(ddof=0).shift().fillna(.02).clip(lower=.008)
    market_dd = 1 - index / index.rolling(config.slow, min_periods=1).max()
    weak = index < index.rolling(config.fast, min_periods=5).mean()
    shocks = (observed_returns < -config.shock_z * vol.shift()).sum(axis=1).div(
        observed_returns.notna().sum(axis=1).replace(0, np.nan)).fillna(0)
    return Features(market.symbols, tuple(market.sectors.get(s, 'unknown') for s in market.symbols),
                    close.to_numpy(), ready.to_numpy(), score.to_numpy(), entry.to_numpy(),
                    exit_signal.fillna(False).to_numpy(), vol.to_numpy(), breadth.to_numpy(),
                    market_return.to_numpy(), market_vol.to_numpy(), market_dd.to_numpy(),
                    weak.fillna(False).to_numpy(), shocks.to_numpy())

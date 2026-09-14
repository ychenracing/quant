"""Independent, matured-label expectation learning and funded trend ownership.

Research implementation of adaptive_expectation_contract.json. Predictors never
include identifiers, dates, sectors, outside-universe quotes or future-trained
coefficients. The normal engine remains responsible for all actual fills.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from techquant.config import Config
from techquant.data import Market, file_hash
from techquant.engine import Result, run
from techquant.policy import CloseDecision, CloseObservation


@dataclass(frozen=True)
class Parameters:
    horizon: int = 10
    shrinkage: float = 1.
    positions: int = 2

    def __post_init__(self) -> None:
        for name in ('horizon', 'positions'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not math.isfinite(self.shrinkage) or self.shrinkage <= 0:
            raise ValueError('shrinkage must be finite and positive')


@dataclass
class Prediction:
    symbols: tuple[str, ...]
    data_sha256: str
    horizon: int
    shrinkage: float
    expected_return: np.ndarray
    adverse: np.ndarray
    utility: np.ndarray
    ready: np.ndarray
    price: np.ndarray
    ema10: np.ndarray
    momentum5: np.ndarray
    fits: list[dict[str, Any]]

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for a in (self.expected_return, self.adverse, self.utility, self.ready):
            h.update(np.asarray(a, dtype='<f8').tobytes())
        return h.hexdigest()


def _features(market: Market) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    quoted = market.panel('close')
    volume = market.panel('volume')
    active = quoted.notna() & volume.gt(0)
    close = quoted.ffill()
    count = active.cumsum()
    anchor = quoted.where(active & count.eq(1)).ffill()
    observed = np.log(close / anchor)
    momenta = [(np.log(close / close.shift(k))).fillna(observed) for k in (1, 5, 20, 60, 120)]
    ret = close.pct_change(fill_method=None)
    ema10 = close.ewm(span=10, adjust=False, min_periods=5).mean()
    ema20 = close.ewm(span=20, adjust=False, min_periods=5).mean()
    ema60 = close.ewm(span=60, adjust=False, min_periods=5).mean()
    downside = np.sqrt(ret.clip(upper=0).pow(2).rolling(20, min_periods=5).mean())
    volatility5 = ret.rolling(5, min_periods=5).std(ddof=0)
    volatility20 = ret.rolling(20, min_periods=5).std(ddof=0).clip(lower=.001)
    prior_volume = volume.rolling(20, min_periods=5).mean().shift().clip(lower=1.)
    relative_volume = np.log(volume.clip(lower=1.) / prior_volume)
    span = market.panel('high') - market.panel('low')
    location = ((quoted - market.panel('low')) / span.where(span > 0)).fillna(.5)
    drawdown = 1. - close / close.rolling(20, min_periods=1).max()
    ready = active & count.ge(10)
    breadth = (close.gt(ema20) & ready).sum(axis=1).div(ready.sum(axis=1).replace(0, np.nan)).fillna(0)
    broad = np.broadcast_to(breadth.to_numpy()[:, None], close.shape)
    components = [*momenta, ema20 / ema60 - 1., downside,
                  volatility5 / volatility20, relative_volume, location, drawdown]
    x = np.stack([a.to_numpy() for a in components] + [broad,
                  momenta[-1].to_numpy() * momenta[0].clip(upper=0).to_numpy(),
                  momenta[2].pow(2).to_numpy()], axis=-1)
    valid = ready.to_numpy() & np.isfinite(x).all(axis=-1)
    # Explicit, uncalibrated warm-up prior, replaced by the first admissible fit.
    prior = np.stack([a.to_numpy() for a in momenta[1:]], axis=-1).mean(axis=-1)
    prior = np.nan_to_num(prior, nan=0.)
    return x, valid, close.to_numpy(), ema10.to_numpy(), momenta[1].to_numpy(), prior


def matured_labels(close: np.ndarray, open_: np.ndarray, *, horizon: int,
                   cutoff: int) -> np.ndarray:
    """Labels which have fully matured by cutoff; missing paths stay invalid.

    Crucially, this slices observable history BEFORE constructing windows. A
    caller cannot accidentally receive even an unmasked future outcome.
    """
    if horizon < 1 or cutoff < 0 or close.shape != open_.shape:
        raise ValueError('invalid label inputs')
    observed = np.asarray(close[:cutoff + 1], dtype=float)
    count = len(observed) - horizon
    if count <= 0:
        return np.empty((0, close.shape[1], 2))
    future = np.lib.stride_tricks.sliding_window_view(observed[1:], horizon, axis=0)
    lowest = future.min(axis=-1)  # NaN propagation rejects the complete missing path.
    entry = np.asarray(open_[1:count + 1], dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        growth = np.log(observed[horizon:] / entry)
        adverse = np.maximum(0., 1. - lowest / entry)
    result = np.stack((growth, adverse), axis=-1)
    valid = np.isfinite(future).all(axis=-1) & np.isfinite(entry) & (entry > 0)
    result[~valid] = np.nan
    return result


def forecast(market: Market, parameters: Parameters) -> Prediction:
    x, ready, price, ema10, momentum5, prior = _features(market)
    quoted, open_ = market.panel('close').to_numpy(), market.panel('open').to_numpy()
    n, width = ready.shape
    expected, adverse = prior.copy(), np.zeros((n, width))
    fit_rows: list[dict[str, Any]] = []
    model = None
    for i in range(n):
        if i % 20 == 0 and i >= parameters.horizon:
            labels = matured_labels(quoted, open_, horizon=parameters.horizon, cutoff=i)
            last = len(labels)
            first = max(0, last - 504)
            eligible = ready[first:last] & np.isfinite(labels[first:last]).all(axis=-1)
            date_count = eligible.sum(axis=1)
            distinct = int(np.count_nonzero(date_count))
            if distinct >= 20:
                # Each signal date receives the same aggregate weight before
                # recency weighting; IPOs do not grant a date extra influence.
                decay = np.exp2((np.arange(first, last) - (last - 1)) / 120.)
                weights = (np.broadcast_to((decay / np.maximum(date_count, 1))[:, None], eligible.shape))[eligible]
                weights /= weights.sum()
                xx, yy = x[first:last][eligible], labels[first:last][eligible]
                mean_x, mean_y = weights @ xx, weights @ yy
                scale = np.sqrt(weights @ (xx - mean_x) ** 2)
                scale = np.maximum(scale, 1e-8)
                z = (xx - mean_x) / scale
                regularized = z.T @ (weights[:, None] * z) + parameters.shrinkage * np.eye(z.shape[1])
                coefficients = np.linalg.solve(regularized, z.T @ (weights[:, None] * (yy - mean_y)))
                model = mean_x, scale, coefficients, mean_y
                used_dates = np.flatnonzero(date_count) + first
                fit_rows.append({'signal_session': i, 'signal_date': str(market.calendar[i].date()),
                    'first_feature_session': int(used_dates[0]),
                    'last_feature_session': int(used_dates[-1]),
                    'maximum_label_end_session': int(used_dates[-1] + parameters.horizon),
                    'distinct_label_dates': distinct, 'samples': int(eligible.sum()),
                    'coefficient_sha256': hashlib.sha256(coefficients.astype('<f8').tobytes()).hexdigest()})
        if model is not None:
            mean_x, scale, coefficients, mean_y = model
            good = ready[i]
            estimate = ((x[i, good] - mean_x) / scale) @ coefficients + mean_y
            # Bounds are economic-domain safety bounds, not acceptance thresholds.
            expected[i, good] = np.clip(estimate[:, 0], -math.log(4), math.log(4))
            adverse[i, good] = np.clip(estimate[:, 1], 0., .95)
    utility = np.where(ready, expected - adverse, -np.inf)
    return Prediction(market.symbols, market.fingerprint(), parameters.horizon,
                      parameters.shrinkage, expected, adverse, utility, ready,
                      price, ema10, momentum5, fit_rows)


class Owner:
    """Retain real funded units, with a latched protective exit and slow rotation."""
    def __init__(self, symbols: tuple[str, ...], prediction: Prediction, parameters: Parameters):
        if (symbols != prediction.symbols or parameters.horizon != prediction.horizon
                or parameters.shrinkage != prediction.shrinkage):
            raise ValueError('prediction identity differs from requested ownership policy')
        self.symbols, self.p, self.parameters = symbols, prediction, parameters
        self.peaks = np.zeros(len(symbols))
        self.was_owned = np.zeros(len(symbols), dtype=bool)
        self.pending_exit = np.zeros(len(symbols), dtype=bool)
        self.negative = np.zeros(len(symbols), dtype=int)
        self.pause_until = -1
        self.last_session = -1

    def decide(self, close: CloseObservation) -> CloseDecision:
        i, current = close.session, close.weights
        if i <= self.last_session:
            raise ValueError('ownership state cannot be reused or run backwards')
        self.last_session = i
        held = close.units > 1e-10
        price, score, ready = self.p.price[i], self.p.utility[i], self.p.ready[i]
        new = held & ~self.was_owned
        self.peaks[new] = price[new]
        self.peaks[held] = np.maximum(self.peaks[held], price[held])
        self.peaks[~held] = 0.
        self.pending_exit[~held] = False  # Only actual liquidation clears intent.
        self.negative = np.where(score < 0, self.negative + 1, 0)
        loss = np.zeros(len(held))
        np.divide(price, self.peaks, out=loss, where=held & (self.peaks > 0))
        breached = held & ((self.negative >= 2) | (1. - loss >= .12) | ~ready)
        triggered = breached & ~self.pending_exit
        why = []
        if np.any(triggered):
            self.pending_exit[triggered] = True
            self.pause_until = i + 5
            why.append('NEGATIVE_EXPECTATION_OR_OWNED_DRAWDOWN')
        retry = self.pending_exit & held
        want = current.copy()
        want[retry] = 0.
        if np.any(retry):
            why.append('PROTECTIVE_EXIT_RETRY')
        drift_cap = 1. if len(held) == 1 else .8
        if np.any(want > drift_cap):
            want = np.minimum(want, drift_cap)
            why.append('CONCENTRATION_REDUCTION')
        capacity = min(self.parameters.positions, len(held))
        positive = ready & np.isfinite(score) & (score > 0)
        ranked = sorted(np.flatnonzero(positive), key=lambda j: (-score[j], self.symbols[j]))
        entrants = [j for j in ranked if not held[j] and price[j] > self.p.ema10[i, j]
                    and self.p.momentum5[i, j] > 0]
        # Never fund an entry using an intended but unfilled sale. Deliberate
        # non-protective replacements are also sold before their cash is reused.
        if i <= self.pause_until or np.any(retry):
            why.append('PAUSE_NEW_MONEY')
        else:
            existing = list(np.flatnonzero(held))
            if i % 20 == 0 and len(existing) >= capacity and entrants:
                rank = {j: k + 1 for k, j in enumerate(ranked)}
                weakest = min(existing, key=lambda j: (score[j], self.symbols[j]))
                if (rank.get(weakest, len(held) + 1) > 2 * capacity
                        and score[entrants[0]] > 1.25 * max(score[weakest], .001)):
                    want[weakest] = 0.
                    why.append('MATERIAL_LEADER_REPLACEMENT')
            vacancies = max(0, capacity - len(existing))
            entrants = entrants[:vacancies]
            free_cash = min(close.cash / close.nav, max(0., 1. - current.sum()))
            if entrants and free_cash >= .01:
                entry_cap = 1. if len(held) == 1 else .6
                amount = min(entry_cap, free_cash / len(entrants))
                for j in entrants:
                    want[j] = amount
                why.append('FUNDED_POSITIVE_EXPECTATION_ENTRY')
        self.was_owned = held.copy()
        return CloseDecision(want, '|'.join(why) if why else 'RETAIN_FUNDED_UNITS')

    def identity(self) -> dict[str, Any]:
        return {'name': 'adaptive_return_downside_expectation', 'parameters': asdict(self.parameters),
                'data_sha256': self.p.data_sha256, 'universe': list(self.symbols),
                'implementation_sha256': file_hash(Path(__file__)),
                'forecast_sha256': self.p.fingerprint(), 'fit_log': self.p.fits,
                'status': 'RESEARCH_NOT_ACCEPTED'}


def replay(market: Market, parameters: Parameters, *, prediction: Prediction | None = None,
           config: Config | None = None, **kwargs: Any) -> Result:
    def factory(m: Market, cfg: Config) -> Owner:
        p = prediction or forecast(m, parameters)
        if p.data_sha256 != m.fingerprint():
            raise ValueError('cached forecast cannot be reused for a different dataset or subset')
        return Owner(m.symbols, p, parameters)
    return run(market, config, policy_factory=factory, **kwargs)

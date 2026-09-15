"""Causal relative-leadership regression; rankings are not execution returns.

Only completed next-open cohorts train the model. Each cohort has equal weight,
so an expanding universe does not implicitly give later market regimes more say.
There is one frozen ridge penalty and no model/parameter selection here.
"""
from __future__ import annotations
import numpy as np


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def cohort_features(close, ready, *, fast: int, slow: int):
    _positive_int(fast, 'fast'); _positive_int(slow, 'slow')
    if slow <= fast:
        raise ValueError('slow must exceed fast')
    prices = np.asarray(close, dtype=float)
    mask = np.asarray(ready)
    if (prices.ndim != 2 or not prices.size or mask.shape != prices.shape
            or mask.dtype != bool or np.isinf(prices).any()
            or np.any(np.isfinite(prices) & (prices <= 0))
            or np.any(mask & ~np.isfinite(prices))):
        raise ValueError('features need positive observed prices and a matching ready mask')
    length, names = prices.shape
    result = np.full((length, names, 3), np.nan)
    first = np.full(names, np.nan)
    marks = np.full(prices.shape, np.nan)
    previous = np.full(names, np.nan)
    # Forward marks are causal. They never make a stale quote ready.
    for i in range(length):
        fresh = np.isfinite(prices[i])
        first = np.where(np.isnan(first) & fresh, prices[i], first)
        previous = np.where(fresh, prices[i], previous)
        marks[i] = previous
        ids = np.flatnonzero(mask[i])
        if len(ids) < 2:
            continue
        columns = []
        for horizon in (fast, slow, 4*slow):
            lag = marks[i-horizon] if i >= horizon else first
            anchor = np.where(np.isfinite(lag), lag, first)
            columns.append(np.log(prices[i, ids]/anchor[ids]))
        values = np.column_stack(columns)
        centered = values-values.mean(axis=0)
        scale = centered.std(axis=0, ddof=0)
        scale = np.where(scale > 1e-12, scale, 1.)
        result[i, ids] = centered/scale
    return result


def mature_cohort(features, opened, active, origin: int, horizon: int, asof: int):
    """Return only observable endpoint pairs, centered over the same cohort."""
    end = origin+horizon+1
    if origin < 0 or end > asof or end >= len(opened):
        return None
    ids = np.flatnonzero(np.isfinite(features[origin]).all(axis=1)
        & active[origin+1] & active[end]
        & np.isfinite(opened[origin+1]) & np.isfinite(opened[end])
        & (opened[origin+1] > 0) & (opened[end] > 0))
    if len(ids) < 2:
        return None
    x = features[origin, ids]
    y = np.log(opened[end, ids]/opened[origin+1, ids])
    return x-x.mean(axis=0), y-y.mean()


def learn_relative_rank(close, opened, active, ready, *, fast: int, slow: int,
                        horizon: int, refit: int):
    """Issue ranks and fit provenance without consuming any immature labels.

    NaN denotes an explicit original-priority fallback, not a negative signal.
    The caller must not turn a relative rank into an admission or risk veto.
    """
    for value, name in ((fast,'fast'), (slow,'slow'), (horizon,'horizon'), (refit,'refit')):
        _positive_int(value, name)
    price, opening = np.asarray(close, dtype=float), np.asarray(opened, dtype=float)
    active, ready = np.asarray(active), np.asarray(ready)
    if (price.ndim != 2 or opening.shape != price.shape or active.shape != price.shape
            or ready.shape != price.shape or active.dtype != bool or ready.dtype != bool
            or np.any(ready & ~active) or np.isinf(opening).any()
            or np.any(np.isfinite(opening) & (opening <= 0))):
        raise ValueError('invalid or incompatible ranking input panels')
    features = cohort_features(price, ready, fast=fast, slow=slow)
    predictions = np.full(price.shape, np.nan)
    fits, beta = [], None
    for i in range(len(price)):
        if i % refit == 0:
            last = i-horizon-1
            origins = range(max(0, last-4*slow+1), last+1)
            gram, rhs = np.zeros((3,3)), np.zeros(3)
            count, samples, used = 0, 0, []
            for origin in origins:
                cohort = mature_cohort(features, opening, active, origin, horizon, i)
                if cohort is None:
                    continue
                x, y = cohort
                gram += x.T@x/len(y)
                rhs += x.T@y/len(y)
                count += 1; samples += len(y); used.append(origin)
            beta = None
            if count >= slow:
                coefficient = np.linalg.solve(gram/count+np.eye(3), rhs/count)
                if np.isfinite(coefficient).all() and np.any(coefficient != 0):
                    beta = coefficient
            fits.append({'session':i, 'usable_cohorts':count, 'sample_count':samples,
                'first_formation_session':used[0] if used else None,
                'last_formation_session':used[-1] if used else None,
                'latest_label_session':used[-1]+horizon+1 if used else -1,
                'coefficients':None if beta is None else beta.tolist()})
        if beta is not None:
            usable = np.isfinite(features[i]).all(axis=1)
            predictions[i, usable] = features[i, usable]@beta
    return predictions, fits


def rank_order(original, predicted, symbols, indices):
    """Order only supplied eligible indices; never compare mixed score scales."""
    old, new = np.asarray(original, dtype=float), np.asarray(predicted, dtype=float)
    ids = [int(j) for j in indices]
    if (old.ndim != 1 or new.shape != old.shape or len(symbols) != len(old)
            or len(set(ids)) != len(ids) or any(j < 0 or j >= len(old) for j in ids)
            or any(not np.isfinite(old[j]) for j in ids)):
        raise ValueError('invalid eligible rank inputs')
    if not all(np.isfinite(new[j]) for j in ids):
        return sorted(ids, key=lambda j: (-old[j], symbols[j]))
    return sorted(ids, key=lambda j: (-new[j], -old[j], symbols[j]))

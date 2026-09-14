"""Learn the ordering of adverse and favorable paths, not only their endpoint.

Barrier labels describe future closing prices. They are not executable stop
prices; the existing independent cash engine alone determines actual fills.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import itertools
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from techquant.data import Market, file_hash
from research.expectation import _features
from research.nonlinear import (MODEL, Prediction as BasePrediction, Owner as FundedOwner,
                               Parameters as OwnershipParameters, runtime, training, tree_hash)
from research.observed_trend import Owner as TrendOwner, Parameters as TrendParameters


@dataclass(frozen=True)
class Parameters:
    horizon: int = 20
    loss_barrier: float = .08

    def __post_init__(self):
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int) or self.horizon < 1:
            raise ValueError('horizon must be a positive integer')
        if isinstance(self.loss_barrier, bool) or not np.isfinite(self.loss_barrier) or not 0 < self.loss_barrier < .5:
            raise ValueError('loss barrier must be positive and below one half')


def grid():
    return [Parameters(*v) for v in itertools.product((20, 60), (.08, .12))]


def matured_outcomes(close, opening, *, horizon, cutoff, loss_barrier):
    """Only complete, observable paths enter even if a barrier was hit earlier."""
    Parameters(horizon, loss_barrier)
    close, opening = np.asarray(close, dtype=float), np.asarray(opening, dtype=float)
    if close.ndim != 2 or close.shape != opening.shape or not 0 <= cutoff < len(close):
        raise ValueError('invalid path input shape or cutoff')
    observed = close[:cutoff + 1]
    count = len(observed) - horizon
    if count <= 0:
        return np.empty((0, close.shape[1], 1))
    path = np.lib.stride_tricks.sliding_window_view(observed[1:], horizon, axis=0)
    entry = opening[1:count + 1]
    valid = np.isfinite(path).all(axis=-1) & (path > 0).all(axis=-1) & np.isfinite(entry) & (entry > 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        ratios = path / entry[:, :, None]
    higher, lower = ratios >= 1. + 2. * loss_barrier, ratios <= 1. - loss_barrier
    first_high = np.where(higher.any(axis=-1), higher.argmax(axis=-1), horizon)
    first_low = np.where(lower.any(axis=-1), lower.argmax(axis=-1), horizon)
    labels = np.where(first_high < first_low, 1., np.where(first_low < first_high, -1., 0.))
    labels[~valid] = np.nan
    return labels[:, :, None]


def expanded_probabilities(raw, classes, distinct_dates):
    """Smoothed fixed class order [-1, 0, +1], including absent training classes."""
    raw, classes = np.asarray(raw, dtype=float), np.asarray(classes)
    if (raw.ndim != 2 or classes.ndim != 1 or raw.shape[1] != len(classes)
            or len(np.unique(classes)) != len(classes) or not np.isin(classes, [-1, 0, 1]).all()
            or not np.isfinite(raw).all() or (raw < 0).any()
            or not np.allclose(raw.sum(axis=1), 1.) or distinct_dates < 1):
        raise ValueError('invalid outcome probabilities, classes or sample size')
    result = np.zeros((len(raw), 3))
    result[:, classes.astype(int) + 1] = raw
    return (result * distinct_dates + 1.) / (distinct_dates + 3.)


@dataclass
class Prediction(BasePrediction):
    outcome_probability: np.ndarray

    def fingerprint(self):
        digest = hashlib.sha256(super().fingerprint().encode())
        digest.update(np.asarray(self.outcome_probability, dtype='<f8').tobytes())
        return digest.hexdigest()


def forecast(market: Market, p: Parameters) -> Prediction:
    x, valid, price, ema10, momentum5, prior = _features(market)
    quote = market.panel('close'); closing = quote.to_numpy(); opening = market.panel('open').to_numpy()
    close = quote.ffill()
    ready = valid & ((quote.notna() & market.panel('volume').gt(0)).cumsum().to_numpy() >= 20)
    expected = prior.copy() * p.horizon / 60.; tail = np.full(prior.shape, .1)
    probabilities = np.full((*prior.shape, 3), np.nan)
    model = None; single = None; distinct = 0; fits = []
    for i in range(len(market.calendar)):
        if i % 10 == 0:
            labels = matured_outcomes(closing, opening, horizon=p.horizon, cutoff=i, loss_barrier=p.loss_barrier)
            sample = training(x, valid, labels, i, p.horizon)
            if sample is not None:
                xx, yy, weights, receipt = sample
                target = yy[:, 0].astype(int); classes = np.unique(target)
                distinct = receipt['distinct_dates']
                receipt.update(task='first_passage', date=str(market.calendar[i].date()),
                               classes=classes.tolist(), loss_barrier=p.loss_barrier)
                if len(classes) == 1:
                    model = None; single = int(classes[0])
                    receipt['single_class'] = single
                else:
                    with threadpool_limits(limits=1):
                        model = HistGradientBoostingClassifier(**MODEL).fit(xx, target, sample_weight=weights)
                    single = None; receipt['tree_sha256'] = tree_hash(model)
                fits.append(receipt)
        good = valid[i]
        if good.any() and distinct:
            if model is None:
                raw = np.ones((int(good.sum()), 1)); classes = np.array([single])
            else:
                with threadpool_limits(limits=1):
                    raw = model.predict_proba(x[i, good]); classes = model.classes_
            prob = expanded_probabilities(raw, classes, distinct)
            probabilities[i, good] = prob
            expected[i, good] = p.loss_barrier * (2. * prob[:, 2] - prob[:, 0])
            tail[i, good] = prob[:, 0]
    result = Prediction(market.symbols, market.fingerprint(), p.horizon, expected, tail, ready,
                        price, ema10, close.ewm(span=20, adjust=False).mean().to_numpy(),
                        close.ewm(span=60, adjust=False).mean().to_numpy(), momentum5,
                        close.pct_change(fill_method=None).to_numpy(), fits, probabilities)
    for field in ('expected','tail','ready','price','ema10','ema20','ema60','momentum5','ret1','outcome_probability'):
        getattr(result, field).setflags(write=False)
    return result


_CACHE = {}


def prepared(market, p):
    root = Path(__file__).parent
    key = (market.fingerprint(), p.horizon, p.loss_barrier,
           tuple(file_hash(root / name) for name in ('pathwise.py','expectation.py','nonlinear.py')),
           tuple(runtime().items()))
    if key not in _CACHE:
        _CACHE[key] = forecast(market, p)
    return _CACHE[key]


class Owner:
    def __init__(self, market: Market, p: Parameters):
        self.market, self.params = market, p
        self.f = prepared(market, p)
        learner = FundedOwner(market, OwnershipParameters(horizon=p.horizon, tail_threshold=.5, positions=4),
                              prediction=self.f)
        self.inner = TrendOwner(market, TrendParameters(trend_span=60, require_market_trend=True), learner=learner)

    def decide(self, observation):
        return self.inner.decide(observation)

    def identity(self):
        root = Path(__file__).parent
        return {'name': 'first_passage_expectation', 'parameters': asdict(self.params),
                'implementation_sha256': file_hash(Path(__file__)),
                'contract_sha256': file_hash(root / 'pathwise_contract.json'),
                'forecast_sha256': self.f.fingerprint(), 'data_sha256': self.f.data_sha256,
                'classes': [-1,0,1], 'fits': self.f.fits, 'learner_runtime': runtime(),
                'score_semantics': 'bounded path utility, not terminal or portfolio return',
                'ownership': self.inner.identity(), 'status': 'RESEARCH_NOT_ACCEPTED'}

"""Preregistered tail authority scored against its as-issued historical reference.

Only complete observed paths may score earlier predictions. Forecast skill is
not an executable loss bound or evidence of economic acceptance. The independent
price owner retains every inventory, cash and outstanding-protection obligation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from techquant.data import Market, file_hash
from techquant.evidence import source_identity
from techquant.policy import CloseDecision, CloseObservation
from research.coherent import SignalInputs
from research.expectation import _features
from research.issued_forecasts import SOURCE, RUN
from research.nonlinear import Owner as ForecastValidator, Parameters as ForecastParameters
from research.pathwise import Prediction, matured_outcomes
from research.trend_book import Owner as PriceOwner, Parameters as PriceParameters


@dataclass(frozen=True)
class Parameters:
    audit_tail: bool = True

    def __post_init__(self):
        if type(self.audit_tail) is not bool:
            raise ValueError('audit_tail must be one of the two registered booleans')


def grid():
    return [Parameters(False), Parameters(True)]


def historical_probability(outcomes, feature_valid, last_mature: int) -> np.ndarray:
    """Date-balanced Laplace reference using at most 504 mature feature sessions.

    Training validity is the original causal feature validity, not the later
    trading admission gate. Forty distinct valid dates are required. The sum of
    date-decayed observation weights is normalized to that distinct-date count,
    exactly as in the issuing learner's training convention.
    """
    outcomes, feature_valid = np.asarray(outcomes), np.asarray(feature_valid)
    if (outcomes.ndim != 2 or outcomes.shape != feature_valid.shape
            or type(last_mature) is not int or not 0 <= last_mature < len(outcomes)):
        raise ValueError('invalid mature outcome boundary or reference shape')
    first, end = max(0, last_mature + 1 - 504), last_mature + 1
    labels = outcomes[first:end]
    valid = feature_valid[first:end] & np.isfinite(labels)
    if not np.isin(labels[valid], [-1., 0., 1.]).all():
        raise ValueError('unrecognized outcome class')
    counts = valid.sum(axis=1)
    distinct = int(np.count_nonzero(counts))
    if distinct < 40:
        return np.full(3, np.nan)
    decay = np.exp2((np.arange(first, end) - last_mature) / 120.)
    weights = np.broadcast_to((decay / np.maximum(counts, 1))[:, None], valid.shape)[valid].copy()
    weights *= distinct / weights.sum()
    totals = np.bincount(labels[valid].astype(int) + 1, weights=weights, minlength=3)
    return (totals + 1.) / (distinct + 3.)


@dataclass(frozen=True)
class Audit:
    reference_probability: np.ndarray
    model_loss: np.ndarray
    reference_loss: np.ndarray
    scored_dates: np.ndarray
    skill: np.ndarray
    active: np.ndarray
    matured_outcome: np.ndarray

    def __post_init__(self):
        for field in fields(self):
            getattr(self, field.name).setflags(write=False)

    def arrays(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def fingerprint(self):
        digest = hashlib.sha256()
        for name, array in self.arrays().items():
            digest.update(name.encode())
            digest.update(str(array.shape).encode())
            digest.update(np.asarray(array, dtype='<f8').tobytes())
        return digest.hexdigest()


def audit_forecasts(market: Market, prediction: Prediction) -> Audit:
    """Issue references and score forecasts on a strictly advancing close clock.

    Loss arrays are indexed by *maturity day*, not by the earlier issuance day.
    No full-history label array is constructed and subsequently masked. Each
    complete 20-session path is formed only when its final close is observed.
    """
    ForecastValidator(market, ForecastParameters(20, .5, 4), prediction=prediction)
    shape = (len(market.calendar), len(market.symbols))
    probability = prediction.outcome_probability
    if probability.shape != (*shape, 3):
        raise ValueError('as-issued outcome probability shape mismatch')
    issued = np.isfinite(probability).all(axis=-1)
    missing = np.isnan(probability).all(axis=-1)
    values = probability[issued]
    if (not (issued | missing).all() or (values < 0).any() or (values > 1).any()
            or not np.allclose(values.sum(axis=-1), 1.)
            or not np.allclose(probability[:, :, 0][issued], prediction.tail[issued])):
        raise ValueError('invalid or inconsistent as-issued class probabilities')
    _, feature_valid, *_ = _features(market)
    closing, opening = market.panel('close').to_numpy(), market.panel('open').to_numpy()
    outcomes = np.full(shape, np.nan)
    matured = np.full(shape, np.nan)
    reference = np.full((shape[0], 3), np.nan)
    model_loss, reference_loss, skill = (np.full(shape[0], np.nan) for _ in range(3))
    scored_dates = np.zeros(shape[0], dtype=int)
    active = np.zeros(shape[0], dtype=bool)
    for i in range(shape[0]):
        j = i - 20
        if j >= 0:
            outcomes[j] = matured_outcomes(closing[j:i + 1], opening[j:i + 1],
                horizon=20, cutoff=20, loss_barrier=.12)[0, :, 0]
            matured[i] = outcomes[j]
            # A numeric warm-up tail prior without issued class probabilities
            # is not a scored model prediction. Missing paths stay unscored.
            good = prediction.ready[j] & issued[j] & np.isfinite(outcomes[j])
            if good.any() and np.isfinite(reference[j]).all():
                event = (outcomes[j, good] == -1.).astype(float)
                model_loss[i] = np.mean((prediction.tail[j, good] - event) ** 2)
                reference_loss[i] = np.mean((reference[j, 0] - event) ** 2)
            reference[i] = historical_probability(outcomes, feature_valid, j)
        start = max(0, i + 1 - 120)
        dates = np.arange(start, i + 1)
        dates = dates[np.isfinite(model_loss[dates]) & np.isfinite(reference_loss[dates])]
        scored_dates[i] = len(dates)
        if len(dates) >= 40:
            weights = np.exp2((dates - i) / 120.)
            baseline = float(weights @ reference_loss[dates])
            if baseline > 0:
                skill[i] = 1. - float(weights @ model_loss[dates]) / baseline
                active[i] = skill[i] > 0.
    return Audit(reference, model_loss, reference_loss, scored_dates, skill, active, matured)


class Owner(PriceOwner):
    def __init__(self, market: Market, p: Parameters, *, prediction: Prediction):
        # Validation and auditing have no fit path. The single price-account
        # owner remains responsible for all risk and actual-fill state.
        self.audit = audit_forecasts(market, prediction)
        self.reliability_parameters = p
        super().__init__(market, PriceParameters(2))
        self.f = prediction
        self.warning = np.zeros(len(market.symbols), dtype=bool)

    def _active(self, i: int) -> bool:
        return not self.reliability_parameters.audit_tail or bool(self.audit.active[i])

    def _signal_inputs(self, i: int) -> SignalInputs:
        signal = super()._signal_inputs(i)
        authority = self._active(i)
        self.warning = (signal.ready & (self.f.tail[i] >= .5)
                        & (self.price_signals.ret1[i] < 0)) if authority else np.zeros_like(signal.ready)
        broken = signal.broken | self.warning
        healthy = signal.healthy & (self.f.tail[i] < .25) if authority else signal.healthy
        # A current warning cannot purchase a previously empty slot or restore
        # units on the same close. This does not create a utility entry gate.
        return replace(signal, broken=broken, healthy=healthy, allowed=signal.allowed & ~broken)

    def decide(self, observation: CloseObservation) -> CloseDecision:
        decision = super().decide(observation)
        i = observation.session
        reason = 'TAIL_AUTHORITY:' + ('ON' if self._active(i) else 'OFF')
        if self.warning.any():
            reason += '|TAIL_EXIT_WARNING:' + ','.join(
                self.market.symbols[j] for j in np.flatnonzero(self.warning))
        return replace(decision, reason=decision.reason + '|' + reason)

    def preserve_audit(self, root: Path, *, require_existing: bool = False) -> Path:
        """Retain exact references, matured scores and prospective authority."""
        root = Path(root) / self.market.fingerprint()
        archive = root / (self.audit.fingerprint() + '.npz')
        receipt = archive.with_suffix('.json')
        expected = {'data_sha256': self.market.fingerprint(),
                    'forecast_sha256': self.f.fingerprint(), 'audit_sha256': self.audit.fingerprint(),
                    'source': source_identity(), 'symbols': list(self.market.symbols),
                    'dates': [str(d.date()) for d in self.market.calendar],
                    'loss_index': 'maturity_day; issuance is20sessions earlier',
                    'forecast_origin': f'{SOURCE}/{RUN}', 'model_refitted': False}
        if archive.exists() or receipt.exists() or require_existing:
            if not archive.is_file() or not receipt.is_file():
                raise ValueError('missing paired reliability audit evidence')
            saved = json.loads(receipt.read_text())
            if saved != dict(expected, archive_sha256=file_hash(archive)):
                raise ValueError('reliability audit identity or archive mismatch')
            with np.load(archive, allow_pickle=False) as payload:
                if set(payload.files) != set(self.audit.arrays()) or any(
                    not np.array_equal(payload[k], v, equal_nan=True) for k, v in self.audit.arrays().items()):
                    raise ValueError('reliability audit array mismatch')
        else:
            root.mkdir(parents=True, exist_ok=True)
            temporary = archive.with_suffix('.partial')
            with temporary.open('wb') as output:
                np.savez_compressed(output, **self.audit.arrays())
            temporary.replace(archive)
            receipt.write_text(json.dumps(dict(expected, archive_sha256=file_hash(archive)),
                                          indent=2, allow_nan=False) + '\n')
        return archive

    def identity(self):
        return {'name': 'price_selection_with_audited_tail_authority',
                'parameters': asdict(self.reliability_parameters),
                'implementation_sha256': file_hash(Path(__file__)),
                'contract_sha256': file_hash(Path(__file__).with_name('risk_reliability_contract.json')),
                'parent_policy': super().identity(),
                'data_sha256': self.market.fingerprint(),
                'forecast_sha256': self.f.fingerprint(), 'audit_sha256': self.audit.fingerprint(),
                'forecast_origin': f'{SOURCE}/{RUN}', 'model_refitted': False,
                'status': 'RESEARCH_NOT_ACCEPTED'}

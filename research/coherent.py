"""Admission-first ownership coordinated with the actual filled-account risk cap.

This implements the existing two-candidate contract. Forecasts are supplied as
immutable evidence; this owner never fits a model or mutates the cash ledger.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import numpy as np

from techquant.config import Config
from techquant.data import Market, file_hash
from techquant.features import build_features
from techquant.execution import round_quantity
from techquant.policy import CloseDecision, CloseObservation
from techquant.strategy import RiskState
from research.nonlinear import Owner as ForecastValidator, Parameters as ForecastParameters
from research.observed_trend import signals, Parameters as TrendParameters
from research.pathwise import Prediction


@dataclass(frozen=True)
class Parameters:
    ranking: str = 'forecast'

    def __post_init__(self):
        if self.ranking not in {'forecast', 'price'}:
            raise ValueError('ranking must be forecast or price')


def grid():
    return [Parameters('forecast'), Parameters('price')]


def equal_increment(indices, weights, budget, ceilings):
    """Share only available cash; redistribute slack without exceeding a cap."""
    result = np.zeros_like(weights)
    remaining = list(indices)
    for _ in range(len(remaining) + 1):
        if not remaining or budget <= 1e-12:
            break
        addition = np.minimum(budget / len(remaining),
                              np.maximum(0., ceilings[remaining] - weights[remaining] - result[remaining]))
        result[remaining] += addition
        budget -= float(addition.sum())
        remaining = [j for j in remaining if weights[j] + result[j] < ceilings[j] - 1e-12]
    return result


def executable_sale(symbol: str, raw_units: float) -> int:
    """Smallest declaration at least as large as the observed protective sale.

    Invert the unchanged execution primitive rather than duplicate board rules.
    This is a close-time request, not a guarantee of the next opening fill.
    """
    if not math.isfinite(raw_units) or raw_units <= 0:
        raise ValueError('protective sale must be finite and positive')
    tolerance = min(1e-9, raw_units * 1e-6)
    low = max(1, math.ceil(raw_units - tolerance))
    high = low
    while round_quantity(symbol, high) < raw_units - tolerance:
        high *= 2
    while low < high:
        mid = (low + high) // 2
        if round_quantity(symbol, mid) >= raw_units - tolerance:
            high = mid
        else:
            low = mid + 1
    return round_quantity(symbol, low)


class Owner:
    def __init__(self, market: Market, p: Parameters, *, prediction: Prediction):
        # Reuse the parent's exact-market validation, not its post-allocation
        # decision path. Passing a prediction prevents any model fit here.
        ForecastValidator(market, ForecastParameters(horizon=20, tail_threshold=.5, positions=4),
                          prediction=prediction)
        self.market, self.params, self.f = market, p, prediction
        self.config = Config()
        self.features = build_features(market, self.config)
        self.s = signals(market, TrendParameters(trend_span=60, require_market_trend=True))
        self.risk, self.history = RiskState(), []
        n = len(market.symbols)
        self.previous = np.zeros(n, dtype=bool)
        self.exit_pending = np.zeros(n, dtype=bool)
        self.readmit = np.zeros(n, dtype=bool)
        self.healthy = np.zeros(n, dtype=int)
        self.negative = np.zeros(n, dtype=int)
        self.reduction_ceiling = np.full(n, np.inf)
        self.restoration_pending = False
        self.restoration_targets = None
        self.restoration_origin_units = None
        # Only today's observed raw/adjusted conversion is consulted. Missing
        # raw quotes do not grant a licence to declare an intention completed.
        self.raw_per_unit = (market.panel('close') / market.panel('raw_close')).to_numpy()
        self.last_session = -1

    def decide(self, o: CloseObservation) -> CloseDecision:
        i, f = o.session, self.f
        if i <= self.last_session:
            raise ValueError('policy sessions must increase')
        self.last_session = i
        held = o.units > 1e-10
        sold = self.previous & ~held
        self.readmit[sold & self.exit_pending] = True
        self.healthy[sold] = 0
        self.exit_pending[~held] = False
        reached = o.units <= self.reduction_ceiling + 1e-10
        self.reduction_ceiling[reached] = np.inf
        good = f.ready[i] & (f.price[i] > f.ema10[i]) & (f.tail[i] < .25)
        self.healthy = np.where(good, self.healthy + 1, 0)
        self.readmit[self.healthy >= 3] = False
        self.negative = np.where(f.expected[i] < -.02, self.negative + 1, 0)
        broken = (self.s.exit[i] | ((f.tail[i] >= .5) & (f.ret1[i] < 0))
                  | ((self.negative >= 3) & (f.price[i] < f.ema60[i]))
                  | (f.ret1[i] <= -.08) | ~f.ready[i])
        self.exit_pending |= held & broken

        old_cap = self.risk.cap
        self.history.append(o.nav)
        cap, risk_reason = self.risk.update(i, self.features, self.history, self.config)
        cut = cap < old_cap - 1e-10
        increased = cap > old_cap + 1e-10
        if cut:
            self.restoration_pending, self.restoration_targets = False, None
            self.restoration_origin_units = None
        elif increased:
            self.restoration_pending = True
            # Already funded requests remain fixed; newly available cash can be
            # considered after those requests are observed, not spent twice.
        reasons = [risk_reason]
        marks = f.price[i]
        units = np.minimum(o.units, self.reduction_ceiling)
        units[self.exit_pending] = 0.
        values = np.zeros_like(units)
        np.multiply(units, marks, out=values, where=units > 0)
        weights = values / o.nav
        ceiling = np.full(len(units), 1. if len(units) == 1 else .8)
        over_name = weights > ceiling + 1e-10
        units[over_name] *= ceiling[over_name] / weights[over_name]
        weights[over_name] = ceiling[over_name]
        permitted = cap if cut or cap == 0 else min(1., cap + self.config.trade_band)
        if weights.sum() > permitted + 1e-10:
            units *= cap / weights.sum()
            weights *= cap / weights.sum()
            reasons.append('FIXED_UNIT_ACCOUNT_REDUCTION')
        # A floor-rounded partial sale can leave an unexecutable remainder.
        # Tighten its ceiling to a declaration the unchanged engine can attempt;
        # never forgive an unfilled protective sale or spend intended proceeds.
        for j in np.flatnonzero((units > 0) & (units < o.units - 1e-10)):
            conversion = self.raw_per_unit[i, j]
            if not np.isfinite(conversion) or conversion <= 0:
                continue
            raw_needed = (o.units[j] - units[j]) * conversion
            raw_sale = executable_sale(self.market.symbols[j], raw_needed)
            revised = max(0., o.units[j] - raw_sale / conversion)
            if revised < units[j] - 1e-10:
                units[j] = revised
                reasons.append('EXECUTABLE_PROTECTIVE_REDUCTION')
        values[:] = 0.
        np.multiply(units, marks, out=values, where=units > 0)
        weights = values / o.nav
        reducing = units < o.units - 1e-10
        self.reduction_ceiling[reducing] = np.minimum(self.reduction_ceiling[reducing], units[reducing])
        outstanding = bool(self.exit_pending.any() or reducing.any())
        if outstanding:
            reasons.append('PROTECTIVE_INVENTORY_RETRY')

        score = f.expected[i] if self.params.ranking == 'forecast' else self.features.score[i]
        allowed = (f.ready[i] & self.s.entry[i] & (f.price[i] > f.ema20[i])
                   & (f.momentum5[i] > 0) & (f.tail[i] < .5) & ~self.readmit
                   & np.isfinite(score))
        if not self.s.market[i]:
            allowed[:] = False
        # Rank only after all purchase predicates, before slots or cash. The
        # bounded forecast utility is deliberately not an absolute return gate.
        entrants = sorted(np.flatnonzero(allowed & ~held),
                          key=lambda j: (-score[j], self.market.symbols[j]))
        capacity = min(4, len(held))
        existing = list(np.flatnonzero(held))
        if not outstanding and i % 20 == 0 and len(existing) >= capacity and entrants:
            ranked = sorted(np.flatnonzero(f.ready[i] & np.isfinite(score)),
                            key=lambda j: (-score[j], self.market.symbols[j]))
            ranks = {j: k + 1 for k, j in enumerate(ranked)}
            weakest = min(existing, key=lambda j: (score[j], self.market.symbols[j]))
            if (ranks.get(weakest, len(held) + 1) > 2 * capacity
                    and score[entrants[0]] > 1.25 * max(score[weakest], .001)):
                units[weakest], weights[weakest] = 0., 0.
                self.exit_pending[weakest] = True
                outstanding = True
                reasons.append('LATCHED_LEADER_REPLACEMENT')

        if self.restoration_targets is not None:
            self.restoration_targets[self.exit_pending] = 0.
            if self.restoration_origin_units is not None:
                remaining = np.maximum(0., self.restoration_targets - o.units)
                progressed = o.units > self.restoration_origin_units + 1e-10
                for j in np.flatnonzero(progressed & (remaining > 1e-10)):
                    conversion = self.raw_per_unit[i, j]
                    if not np.isfinite(conversion) or conversion <= 0:
                        continue
                    notional = remaining[j] * marks[j]
                    why = ('MINIMUM_NOTIONAL' if notional < .01 * o.nav else
                           'MINIMUM_LOT' if round_quantity(self.market.symbols[j],
                                                          remaining[j] * conversion) == 0 else '')
                    if why:
                        # Cancellation is not a fill. Record symbol, remaining
                        # units and CNY amount in the immutable daily decision.
                        reasons.append('RESTORATION_RESIDUAL_CANCELLED:'
                            f'{self.market.symbols[j]}:{remaining[j]:.17g}:{notional:.17g}:{why}')
                        self.restoration_targets[j] = o.units[j]
            if np.all(o.units >= self.restoration_targets - 1e-10):
                self.restoration_targets = None
                self.restoration_origin_units = None
                self.restoration_pending = increased
        if not outstanding:
            newcomers = entrants[:max(0, capacity - len(existing))]
            receivers = ([j for j in existing if allowed[j]] + newcomers
                         if self.restoration_pending else newcomers)
            budget = min(max(0., o.cash / o.nav), max(0., cap - float(weights.sum())))
            ceilings = np.where(held, ceiling, 1. if len(held) == 1 else .6)
            if self.restoration_targets is None and receivers and budget >= .01:
                addition = equal_increment(receivers, weights, budget, ceilings)
                target = units + np.divide(addition * o.nav, marks,
                    out=np.zeros_like(units), where=np.isfinite(marks) & (marks > 0))
                if self.restoration_pending:
                    self.restoration_targets = target.copy()
                    self.restoration_origin_units = o.units.copy()
            elif self.restoration_targets is not None:
                target = np.maximum(units, self.restoration_targets)
                if increased and receivers:
                    reserved = np.zeros_like(units)
                    np.multiply(target, marks / o.nav, out=reserved, where=target > 0)
                    extra = max(0., budget - float((reserved - weights).sum()))
                    addition = equal_increment(receivers, reserved, extra, ceilings)
                    target += np.divide(addition * o.nav, marks, out=np.zeros_like(units),
                                        where=np.isfinite(marks) & (marks > 0))
                    self.restoration_targets = target.copy()
                    self.restoration_origin_units = o.units.copy()
            else:
                target = units.copy()
            # Eligibility can disappear while a buy is blocked. Preserve its
            # evidence/intention, but never execute new risk through a closed gate.
            requested = np.maximum(0., target - units)
            limits = np.divide(ceilings * o.nav, marks, out=np.zeros_like(units),
                               where=np.isfinite(marks) & (marks > 0))
            requested = np.minimum(requested, np.maximum(0., limits - units))
            requested[~allowed] = 0.
            cost = np.zeros_like(requested)
            np.multiply(requested, marks / o.nav, out=cost, where=requested > 0)
            if cost.sum() > budget and cost.sum() > 0:
                requested *= budget / cost.sum()
            if self.restoration_targets is not None and newcomers:
                # Unfilled but currently ineligible restoration is not a cash
                # reservation. Honor eligible pending buys first, then fund an
                # independently admitted vacancy only from the actual remainder.
                funded = np.zeros_like(requested)
                np.multiply(requested, marks / o.nav, out=funded, where=requested > 0)
                addition = equal_increment(newcomers, weights + funded,
                    max(0., budget - float(funded.sum())), ceilings)
                if addition.any():
                    requested += np.divide(addition * o.nav, marks,
                        out=np.zeros_like(units), where=np.isfinite(marks) & (marks > 0))
                    reasons.append('VACANCY_FROM_UNRESERVED_CASH')
            if requested.any():
                units += requested
                reasons.append('FUNDED_RESTORATION' if self.restoration_pending else 'ADMITTED_CASH_ENTRY')

        values[:] = 0.
        np.multiply(units, marks, out=values, where=units > 0)
        weights = values / o.nav
        self.previous = held.copy()
        # The recorded close decision includes permitted drift; risk.cap itself
        # is not changed to hide actual exposure or an unfilled protective order.
        decision_cap = min(1., max(cap, float(weights.sum())))
        return CloseDecision(weights, '|'.join(reasons), decision_cap, units)

    def identity(self):
        root = Path(__file__).parent
        return {'name': 'admission_first_filled_account_ownership', 'parameters': asdict(self.params),
                'implementation_sha256': file_hash(Path(__file__)),
                'contract_sha256': file_hash(root / 'coherent_contract.json'),
                'forecast_sha256': self.f.fingerprint(), 'data_sha256': self.f.data_sha256,
                'forecast_horizon': 20, 'loss_barrier': .12, 'fits': self.f.fits,
                'forecast_origin': '0b5626993598d6cbf8bf0af15345e01d18a730e7/34809876451',
                'forecast_mode': 'verified_issued_arrays_no_refit', 'status': 'RESEARCH_NOT_ACCEPTED'}

"""Admission-first ownership with one actual-account risk authority.

Research-only: the frozen predecessor is untouched. Stock membership, outstanding
protective fills and account recovery are resolved before allocating actual cash.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
from techquant.config import Config
from techquant.data import Market, file_hash
from techquant.features import build_features
from techquant.policy import CloseDecision, CloseObservation
from techquant.strategy import RiskState
from research.nonlinear import Owner as ForecastValidator, Parameters as ForecastParameters
from research.observed_trend import Parameters as TrendParameters, signals


@dataclass(frozen=True)
class Parameters:
    ranking: str = 'forecast'

    def __post_init__(self):
        if self.ranking not in ('forecast', 'price'):
            raise ValueError('ranking must be forecast or price')


def grid():
    return [Parameters('forecast'), Parameters('price')]


class Owner:
    def __init__(self, market: Market, p: Parameters, *, prediction=None, origin=None):
        if prediction is None:
            raise ValueError('this bounded study requires verified issued predictions; no silent refit')
        # Reuse the existing exact-universe/shape validation, not another model.
        self.f = ForecastValidator(market, ForecastParameters(20, .5, 4), prediction=prediction).f
        self.market, self.params = market, p
        self.origin = origin or {'kind': 'explicit_prediction', 'economic_status': 'UNVERIFIED'}
        self.config = Config()
        self.features = build_features(market, self.config)
        self.s = signals(market, TrendParameters(60, True))
        n = len(market.symbols)
        self.risk = RiskState()
        self.nav = []
        self.previous = np.zeros(n, dtype=bool)
        self.exit_pending = np.zeros(n, dtype=bool)
        self.readmit = np.zeros(n, dtype=bool)
        self.healthy = np.zeros(n, dtype=int)
        self.negative = np.zeros(n, dtype=int)
        self.ceiling = np.full(n, np.inf)
        self.restore = False
        self.restore_units = None
        self.last_session = -1

    def decide(self, o: CloseObservation) -> CloseDecision:
        i, f = o.session, self.f
        if i <= self.last_session or not 0 <= i < len(self.market.calendar):
            raise ValueError('policy sessions must increase within the market calendar')
        self.last_session = i
        held = o.units > 1e-10
        sold = self.previous & ~held
        self.readmit[sold & self.exit_pending] = True
        self.healthy[sold] = 0
        self.exit_pending[~held] = False
        healthy = f.ready[i] & (f.price[i] > f.ema10[i]) & (f.tail[i] < .25)
        self.healthy = np.where(healthy, self.healthy + 1, 0)
        self.readmit[self.healthy >= 3] = False
        self.negative = np.where(f.expected[i] < -.02, self.negative + 1, 0)
        broken = (self.s.exit[i] | ~f.ready[i] | (f.ret1[i] <= -.08)
                  | ((f.tail[i] >= .5) & (f.ret1[i] < 0))
                  | ((self.negative >= 3) & (f.price[i] < f.ema60[i])))
        self.exit_pending |= held & broken

        previous_cap = self.risk.cap
        self.nav.append(float(o.nav))
        cap, reason = self.risk.update(i, self.features, self.nav, self.config)
        why = [reason]
        cut = cap < previous_cap - 1e-12
        if cut:
            self.restore, self.restore_units = False, None
        elif cap > previous_cap + 1e-12:
            self.restore = True
        # Clear only quantities the actual opening ledger says have been sold.
        self.ceiling[o.units <= self.ceiling + 1e-8] = np.inf
        exposure = float(o.weights.sum())
        if exposure > cap + (0. if cut or cap <= 0 else self.config.trade_band) + 1e-12:
            proposed = o.units * (cap / exposure)
            self.ceiling = np.minimum(self.ceiling, proposed)
        pending = np.isfinite(self.ceiling) & (o.units > self.ceiling + 1e-8)
        targets = np.minimum(o.units, self.ceiling)
        targets[self.exit_pending] = 0.
        if pending.any():
            why.append('FIXED_RISK_REDUCTION_RETRY')
        if self.exit_pending.any():
            why.append('FULL_EXIT_RETRY')
        marks = f.price[i]
        name_cap = 1. if len(held) == 1 else .8
        name_ceiling = np.divide(name_cap * o.nav, marks, out=np.full(len(held), np.inf),
                                 where=np.isfinite(marks) & (marks > 0))
        targets = np.minimum(targets, name_ceiling)

        # All admission predicates precede ranking, membership and cash division.
        admitted = (f.ready[i] & self.s.entry[i] & self.s.market[i] & ~self.readmit
                    & (f.momentum5[i] > 0) & (marks > f.ema20[i]) & (f.tail[i] < .5))
        score = f.expected[i] if self.params.ranking == 'forecast' else self.features.score[i]
        ranked = sorted(np.flatnonzero(admitted & np.isfinite(score)),
                        key=lambda j: (-score[j], self.market.symbols[j]))
        capacity = min(4, len(held))
        entrants = [j for j in ranked if not held[j]]
        existing = list(np.flatnonzero(held))
        blocked = pending.any() or self.exit_pending.any()
        if not blocked and cap > 0:
            if i % 20 == 0 and len(existing) >= capacity and entrants:
                ranks = {j: k + 1 for k, j in enumerate(ranked)}
                worst = min(existing, key=lambda j: (score[j], self.market.symbols[j]))
                if (ranks.get(worst, len(held) + 1) > 2 * capacity
                        and score[entrants[0]] > 1.25 * max(score[worst], .001)):
                    targets[worst] = 0.
                    self.exit_pending[worst] = True
                    blocked = True
                    why.append('ADMITTED_LEADER_REPLACEMENT')
            if not blocked:
                if self.restore_units is not None:
                    # Completion requires an actual increase, not a requested buy.
                    if np.any(o.units > self.restore_units + 1e-8):
                        self.restore, self.restore_units = False, None
                fresh = entrants[:max(0, capacity - len(existing))]
                recipients = ([j for j in existing if admitted[j]] if self.restore else []) + fresh
                slack = min(max(0., o.cash / o.nav), max(0., cap - exposure))
                if recipients and slack >= .01:
                    want = np.nan_to_num(targets * marks, nan=0.) / o.nav
                    maximum = 1. if len(held) == 1 else .6
                    # Equal incremental *funded* cash, with existing caps respected.
                    free = recipients.copy()
                    for _ in range(len(free) + 1):
                        if not free or slack <= 1e-12:
                            break
                        increment = slack / len(free)
                        for j in free:
                            amount = min(increment, max(0., maximum - want[j]))
                            want[j] += amount
                            slack -= amount
                        free = [j for j in free if want[j] < maximum - 1e-12]
                    additions = np.divide(want * o.nav, marks, out=targets.copy(),
                                          where=np.isfinite(marks) & (marks > 0))
                    increased = additions > targets + 1e-8
                    targets[increased] = additions[increased]
                    if increased.any():
                        why.append('FUNDED_RISK_RESTORATION' if self.restore else 'ADMITTED_CASH_ENTRY')
                        if self.restore:
                            self.restore_units = o.units.copy()
        self.previous = held.copy()
        weights = np.nan_to_num(targets * marks, nan=0.) / o.nav
        # Report the enforceable cap including the already declared drift band,
        # not a permanent 100% placeholder in risk-timing evidence.
        effective_cap = min(1., max(cap, float(weights.sum())))
        decision = CloseDecision(weights, '|'.join(why), cap=effective_cap, unit_targets=targets)
        decision.validated_weights(len(held))
        decision.validated_unit_targets(marks, o.nav)
        return decision

    def identity(self):
        root = Path(__file__).parent
        return {'name': 'admission_first_filled_account_ownership', 'parameters': asdict(self.params),
                'implementation_sha256': file_hash(Path(__file__)),
                'contract_sha256': file_hash(root / 'coherent_contract.json'),
                'forecast_sha256': self.f.fingerprint(), 'data_sha256': self.f.data_sha256,
                'forecast_origin': self.origin, 'status': 'RESEARCH_NOT_ACCEPTED'}

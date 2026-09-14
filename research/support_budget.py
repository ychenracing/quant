"""Cash-funded ownership sized by causal, nondecreasing support risk.

This is a registered research policy, not the production default. A close-time
loss budget cannot guarantee a fill, realized loss bound, or maximum drawdown.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import numpy as np
import pandas as pd

from techquant.config import Config
from techquant.data import Market, file_hash
from techquant.execution import round_quantity
from techquant.features import build_features
from techquant.policy import CloseDecision, CloseObservation
from techquant.strategy import RiskState


@dataclass(frozen=True)
class Parameters:
    risk_budget: float = .06
    positions: int = 2

    def __post_init__(self):
        if (type(self.risk_budget) not in (float, int) or self.risk_budget not in (.06, .10)
                or type(self.positions) is not int or self.positions not in (2, 4)):
            raise ValueError('parameters must belong to the preregistered four-candidate grid')


def grid():
    return [Parameters(risk, count) for risk in (.06, .10) for count in (2, 4)]


class Owner:
    def __init__(self, market: Market, parameters: Parameters):
        self.market, self.params, self.config = market, parameters, Config()
        self.features = build_features(market, self.config)
        close = market.panel('close').ffill()
        high, low = market.panel('high'), market.panel('low')
        previous = close.shift()
        tr = pd.DataFrame(np.maximum.reduce([(high-low).to_numpy(),
            (high-previous).abs().to_numpy(), (low-previous).abs().to_numpy()]),
            index=close.index, columns=close.columns)
        self.atr = tr.rolling(20, min_periods=20).mean().to_numpy()
        self.support = close.shift().rolling(20, min_periods=20).min().to_numpy()
        self.breakout = close.gt(close.shift().rolling(10, min_periods=10).max()).to_numpy()
        active = market.panel('close').notna() & market.panel('volume').gt(0)
        self.ready = (active & active.cumsum().ge(20)).to_numpy() & np.isfinite(self.atr)
        self.raw_per_unit = (market.panel('close') / market.panel('raw_close')).to_numpy()
        self.risk, self.history = RiskState(), []
        n = len(market.symbols)
        self.stop = np.zeros(n)
        self.peak = np.zeros(n)
        self.pending_stop = np.zeros(n)
        self.previous_units = np.zeros(n)
        self.exit_pending = np.zeros(n, dtype=bool)
        self.readmit = np.zeros(n, dtype=bool)
        self.healthy = np.zeros(n, dtype=int)
        self.reduction_ceiling = np.full(n, np.inf)
        self.last_session = -1

    def admission_stop(self, i: int) -> np.ndarray:
        price = np.nan_to_num(self.features.close[i], nan=0.)
        raw = np.maximum(self.support[i], price - 3*self.atr[i])
        return np.nan_to_num(np.minimum(raw, .98*price), nan=0.)

    def identity(self) -> dict:
        root = Path(__file__).parent
        return {'name': 'support_risk_budget', 'parameters': asdict(self.params),
                'implementation_sha256': file_hash(Path(__file__)),
                'contract_sha256': file_hash(root/'support_budget_contract.json'),
                'data_sha256': self.market.fingerprint(), 'status': 'RESEARCH_NOT_ACCEPTED'}

    def _round_reductions(self, current, desired, i):
        """Invert the existing declaration rule; do not leave unsellable targets.

        Whole-position exits retain the engine's odd-lot disposal authority.
        At most one board-lot interval is searched for a partial declaration.
        """
        result = desired.copy()
        for j in np.flatnonzero((desired > 0) & (desired < current-1e-10)):
            factor = self.raw_per_unit[i, j]
            if not np.isfinite(factor) or factor <= 0:
                result[j] = 0.
                continue
            required = (current[j]-desired[j])*factor
            request = max(1, math.ceil(required-1e-9))
            declared = round_quantity(self.market.symbols[j], request)
            while declared < required-1e-9:
                request += 1
                declared = round_quantity(self.market.symbols[j], request)
            result[j] = max(0., current[j]-declared/factor)
        return result

    def decide(self, o: CloseObservation) -> CloseDecision:
        i = o.session
        if i <= self.last_session:
            raise ValueError('policy sessions must increase')
        self.last_session = i
        price = np.nan_to_num(self.features.close[i], nan=0.)
        held = o.units > 1e-10
        sold = (self.previous_units > 1e-10) & ~held
        self.readmit[sold] = True
        self.healthy[sold] = 0
        self.stop[sold] = self.peak[sold] = self.pending_stop[sold] = 0.
        self.exit_pending[~held] = False
        self.reduction_ceiling[o.units <= self.reduction_ceiling+1e-10] = np.inf
        initial = self.admission_stop(i)
        added = held & (o.units > self.previous_units+1e-10)
        self.stop[added] = np.maximum.reduce([self.stop[added], self.pending_stop[added], initial[added]])
        self.pending_stop[added] = 0.
        self.peak[held] = np.maximum(self.peak[held], price[held])
        ratchet = np.fmax(self.support[i], self.peak-3*self.atr[i])
        self.stop[held] = np.fmax(self.stop[held], ratchet[held])
        # Never lower a live campaign stop following a partial execution.
        broken = held & ((price <= self.stop) | self.features.exit[i] | ~self.ready[i])
        self.exit_pending |= broken
        allowed = (self.ready[i] & self.features.entry[i] & ~self.features.exit[i]
                   & np.isfinite(self.features.score[i]) & (self.features.score[i] > 0))
        self.healthy = np.where(allowed, self.healthy+1, 0)
        self.readmit &= self.healthy < self.config.recovery
        previous_cap = self.risk.cap
        self.history.append(o.nav)
        cap, reason = self.risk.update(i, self.features, self.history, self.config)
        cap_cut = cap < previous_cap-1e-12
        desired = np.minimum(o.units, self.reduction_ceiling)
        desired[self.exit_pending] = 0.
        weights = desired*price/o.nav
        admission_cap = max(self.config.single_cap, 1/len(held))
        over = weights > (1. if len(held) == 1 else .8)+1e-12
        desired[over] *= admission_cap/weights[over]
        if len(set(self.features.sectors)) > 1:
            for sector in sorted(set(self.features.sectors)):
                group = np.array([s == sector for s in self.features.sectors])
                exposure = float((desired*price/o.nav)[group].sum())
                if exposure > self.config.sector_cap+self.config.trade_band+1e-12:
                    desired[group] *= self.config.sector_cap/exposure
        exposure = float(desired@price/o.nav)
        ceiling = self._exposure_ceiling(cap, cap_cut)
        if exposure > ceiling+1e-12:
            desired *= cap/exposure
        # Completion is the economic goal, not the conservatively rounded
        # outbound declaration. Otherwise a partial opening fill can trigger an
        # extra board lot and ratchet the target down on every retry.
        completion = desired.copy()
        desired = self._round_reductions(o.units, desired, i)
        completion[desired <= 1e-10] = 0.
        cuts = desired < o.units-1e-10
        self.reduction_ceiling[cuts] = np.minimum(self.reduction_ceiling[cuts], completion[cuts])
        self.exit_pending |= held & (desired <= 1e-10)
        protected = bool(cuts.any()) or cap_cut
        if protected:
            reason += '|PROTECTIVE_INVENTORY_PENDING'
        elif cap > 0:
            desired = self._allocate(o, desired, price, initial, allowed & ~self.readmit, cap)
            if np.any(desired > o.units+1e-10):
                reason += '|CASH_FUNDED_SUPPORT_RISK'
            else:
                reason += '|RETAIN_ACTUAL_UNITS'
        self.previous_units = o.units.copy()
        weights = desired*price/o.nav
        decision = CloseDecision(weights, reason, max(cap, float(weights.sum())), desired)
        decision.validated_weights(len(held))
        decision.validated_unit_targets(price, o.nav)
        return decision

    def _exposure_ceiling(self, cap, cut):
        return cap if cut or cap == 0 else min(1., cap+self.config.trade_band)

    def _risk_limit(self, o, cap):
        return self.params.risk_budget*o.nav*cap

    def _allocate(self, o, desired, price, initial, allowed, cap):
        held = o.units > 1e-10
        capacity = min(self.params.positions, len(held))
        occupied = int(held.sum())
        stops = np.where(held, self.stop, np.maximum(initial, self.pending_stop))
        distance = np.maximum(price-stops, .02*price)
        risk_limit = self._risk_limit(o, cap)
        remaining_risk = max(0., risk_limit-float(o.units@distance))
        cash = max(0., min(.99*o.cash, cap*o.nav-float(o.units@price)))
        symbol_cap = max(self.config.single_cap, 1/len(held))
        sector_cap = self.config.sector_cap if len(set(self.features.sectors)) > 1 else 1.
        candidates = np.flatnonzero(allowed & (price > stops) & (~held | self.breakout[o.session]))
        candidates = sorted(candidates, key=lambda j: (-self.features.score[o.session,j], self.market.symbols[j]))
        for j in candidates:
            if not held[j] and occupied >= capacity:
                continue
            risk = remaining_risk if held[j] else min(remaining_risk, risk_limit/capacity)
            sector = np.array([s == self.features.sectors[j] for s in self.features.sectors])
            value = min(cash, risk*price[j]/distance[j],
                        max(0., symbol_cap*o.nav-desired[j]*price[j]),
                        max(0., sector_cap*o.nav-float((desired*price)[sector].sum())))
            material = self.config.trade_band if held[j] else .01
            if value < material*o.nav or value <= 1e-10:
                continue
            addition = value/price[j]
            desired[j] += addition
            self.pending_stop[j] = max(self.pending_stop[j], initial[j], self.stop[j])
            cash -= value
            remaining_risk = max(0., remaining_risk-addition*distance[j])
            if not held[j]:
                occupied += 1
        return desired

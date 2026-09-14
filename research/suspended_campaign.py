"""Restore only actually liquidated, individually intact risk suspensions.

Portfolio protection always completes first. A suspended campaign grants no
cash entitlement and cannot lower its former protective floor.
"""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import numpy as np
from techquant.data import Market, file_hash
from techquant.policy import CloseObservation
from research.observed_admission_completion import Owner as Parent, Parameters as ParentParameters
from research.quantity_obligation import preserve_trace, verify_trace


@dataclass(frozen=True)
class Parameters:
    pass


def grid():
    return [Parameters()]


class Owner(Parent):
    def __init__(self, market: Market, parameters: Parameters):
        if type(parameters) is not Parameters:
            raise ValueError('suspended campaign accepts only its registered singleton')
        super().__init__(market, ParentParameters())
        self.suspension_parameters = parameters
        self.fast = market.panel('close').ffill().rolling(
            self.inner.config.fast, min_periods=self.inner.config.fast).mean().to_numpy()
        self.suspended_units = np.zeros(len(market.symbols))
        self.stop_floor = np.zeros(len(market.symbols))
        self.flat_health = np.zeros(len(market.symbols), dtype=int)

    def decide(self, observation: CloseObservation):
        p, i = self.inner, observation.session
        if i <= p.last_session:
            raise ValueError('policy sessions must increase')
        held = observation.units > 1e-10
        price = np.nan_to_num(p.features.close[i], nan=0.)
        active = self.suspended_units > 1e-10
        opened = held & (p.previous_units <= 1e-10) & active
        invalid = active & ((price <= self.stop_floor) | p.features.exit[i] | ~p.ready[i])
        # The pending buy already carries the former stop. Consuming a real
        # acquisition clears only the flat-entry veto, never protection/history.
        p.readmit[opened] = False
        self.suspended_units[opened | invalid] = 0.
        self.stop_floor[opened | invalid] = 0.
        health = ~held & p.ready[i] & ~p.features.exit[i] & (price > self.fast[i])
        self.flat_health = np.where(health & (self.suspended_units > 0), self.flat_health + 1, 0)
        prior_exit, prior_cap = p.exit_pending.copy(), p.risk.cap
        decision = super().decide(observation)
        original = decision.unit_targets
        if original is None:
            raise AssertionError('the funded parent must declare economic units')
        capture = (held & (original <= 1e-10) & (p.risk.cap == 0)
                   & ~prior_exit & (price > p.stop) & ~p.features.exit[i] & p.ready[i])
        self.suspended_units[capture] = observation.units[capture]
        self.stop_floor[capture] = p.stop[capture]
        self.flat_health[capture] = 0
        protected = bool(np.any(original < observation.units - 1e-10)
            or np.any(p.exit_pending & held)
            or np.any(observation.units > p.reduction_ceiling + 1e-10)
            or p.risk.cap < prior_cap - 1e-12)
        desired = original.copy()
        restored = np.zeros(len(held), dtype=bool)
        if not protected and p.risk.cap > 0:
            initial = p.admission_stop(i)
            stops = np.where(held, p.stop, np.maximum(initial, p.pending_stop))
            eligible = ((self.suspended_units > 1e-10) & ~held & (original <= 1e-10)
                & (self.flat_health >= p.config.recovery) & p.ready[i] & ~p.features.exit[i]
                & np.isfinite(p.features.score[i]) & (p.features.score[i] > 0)
                & ~p.exit_pending & ~np.isfinite(p.reduction_ceiling))
            stops[eligible] = np.maximum(stops[eligible], self.stop_floor[eligible])
            distance = np.maximum(price-stops, .02*price)
            eligible &= price > stops
            # Reserve every existing parent purchase before considering a
            # suspended name. No sale, signal or target can create spendable cash.
            reserved = float(np.maximum(original-observation.units, 0.) @ price)
            cash = max(0., min(.99*observation.cash-reserved,
                p.risk.cap*observation.nav-float(original @ price)))
            risk_limit = p._risk_limit(observation, p.risk.cap)
            risk = max(0., risk_limit-float(original @ distance))
            capacity = min(p.params.positions, len(held))
            occupied = int(np.sum(original > 1e-10))
            symbol_cap = max(p.config.single_cap, 1/len(held))
            sector_cap = p.config.sector_cap if len(set(p.features.sectors)) > 1 else 1.
            for j in sorted(np.flatnonzero(eligible),
                            key=lambda k: (-p.features.score[i,k], self.market.symbols[k])):
                if occupied >= capacity:
                    break
                sector = np.array([s == p.features.sectors[j] for s in p.features.sectors])
                value = min(cash, min(risk, risk_limit/capacity)*price[j]/distance[j],
                    self.suspended_units[j]*price[j], symbol_cap*observation.nav,
                    max(0., sector_cap*observation.nav-float((desired*price)[sector].sum())))
                if value < .01*observation.nav or value <= 1e-10:
                    continue
                addition = value/price[j]
                desired[j] += addition
                p.pending_stop[j] = max(p.pending_stop[j], self.stop_floor[j], initial[j])
                cash = max(0., cash-value)
                risk = max(0., risk-addition*distance[j])
                occupied += 1
                restored[j] = True
        self.trace.append({'kind':'RISK_SUSPENDED_CAMPAIGN', 'date':observation.date,
            'session':i, 'actual_units':observation.units.tolist(), 'actual_cash':observation.cash,
            'captured':capture.tolist(), 'invalidated':invalid.tolist(), 'actual_open':opened.tolist(),
            'suspended_units':self.suspended_units.tolist(), 'stop_floor':self.stop_floor.tolist(),
            'flat_health':self.flat_health.tolist(), 'protected':protected, 'risk_cap':p.risk.cap,
            'original_units':original.tolist(), 'restored':restored.tolist(), 'desired_units':desired.tolist()})
        if not restored.any():
            return decision
        result = replace(decision, unit_targets=desired, weights=desired*price/observation.nav,
            reason=decision.reason+'|ACTUAL_CASH_SUSPENDED_CAMPAIGN')
        result.validated_weights(len(held))
        result.validated_unit_targets(price, observation.nav)
        return result

    def identity(self):
        return {'name':'actual_cash_suspended_campaign', 'parameters':asdict(self.suspension_parameters),
            'implementation_sha256':file_hash(Path(__file__)),
            'contract_sha256':file_hash(Path(__file__).with_name('suspended_campaign_contract.json')),
            'parent_definition':super().identity(), 'data_sha256':self.market.fingerprint(),
            'status':'RESEARCH_NOT_ACCEPTED'}

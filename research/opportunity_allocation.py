"""Fund distinct eligible opportunities from residual actual-account authority.

This changes a research capacity choice, not any qualification or protection.
The retained parent's orders are reserved before any extra admission is made.
"""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import numpy as np
from techquant.config import Config
from techquant.data import Market, file_hash
from techquant.policy import CloseDecision, CloseObservation
from research.observed_admission_completion import Owner as Parent, Parameters as ParentParameters
from research.quantity_obligation import preserve_trace, verify_trace


@dataclass(frozen=True)
class Parameters:
    fund_residual_opportunities: bool = False

    def __post_init__(self):
        if type(self.fund_residual_opportunities) is not bool:
            raise ValueError('only the registered boolean comparison is supported')


def grid():
    return [Parameters(False), Parameters(True)]


class Owner(Parent):
    def __init__(self, market: Market, parameters: Parameters, *, config: Config | None = None):
        if type(parameters) is not Parameters:
            raise ValueError('only registered opportunity-allocation parameters are supported')
        super().__init__(market, ParentParameters(), config=config)
        self.opportunity_parameters = parameters

    def decide(self, o: CloseObservation) -> CloseDecision:
        previous_cap = self.inner.risk.cap
        decision = super().decide(o)
        if (not self.opportunity_parameters.fund_residual_opportunities
                or self.inner.risk.cap < previous_cap-1e-12):
            return decision
        return self._fund_residual(o, decision)

    def _fund_residual(self, o: CloseObservation, decision: CloseDecision) -> CloseDecision:
        p, i = self.inner, o.session
        original = decision.unit_targets
        if original is None:
            raise AssertionError('the unchanged parent must declare economic units')
        held = o.units > 1e-10
        if (p.risk.cap <= 0 or int(held.sum()) < p.params.positions
                or np.any(original < o.units-1e-10)
                or np.any(p.exit_pending & held)
                or np.any(np.isfinite(p.reduction_ceiling) & held)):
            return decision
        price = np.nan_to_num(p.features.close[i], nan=0.)
        initial = p.admission_stop(i)
        stops = np.where(held, p.stop, np.maximum(initial, p.pending_stop))
        distance = np.maximum(price-stops, .02*price)
        eligible = (p.ready[i] & p.features.entry[i] & ~p.features.exit[i]
                    & np.isfinite(p.features.score[i]) & (p.features.score[i] > 0)
                    & ~p.readmit & ~held & (price > stops)
                    & (original <= o.units+1e-10))
        cash_authority = max(0., min(.99*o.cash, p.risk.cap*o.nav-float(o.units@price)))
        reserved = float(np.maximum(original-o.units, 0.)@price)
        cash = max(0., cash_authority-reserved)
        risk_authority = p._risk_limit(o, p.risk.cap)
        risk = max(0., risk_authority-float(original@distance))
        wanted = original.copy()
        name_cap = max(p.config.single_cap, 1/len(held))
        sector_cap = p.config.sector_cap if len(set(p.features.sectors)) > 1 else 1.
        candidates = sorted(np.flatnonzero(eligible),
                            key=lambda j: (-p.features.score[i, j], self.market.symbols[j]))
        for j in candidates:
            group = np.array([s == p.features.sectors[j] for s in p.features.sectors])
            value = min(cash, risk*price[j]/distance[j],
                        max(0., name_cap*o.nav-wanted[j]*price[j]),
                        max(0., sector_cap*o.nav-float((wanted*price)[group].sum())))
            if value < .01*o.nav or value <= 1e-10:
                continue
            addition = value/price[j]
            wanted[j] += addition
            p.pending_stop[j] = max(p.pending_stop[j], initial[j], p.stop[j])
            cash = max(0., cash-value)
            risk = max(0., risk-addition*distance[j])
        if np.array_equal(wanted, original):
            return decision
        self.trace.append(dict(kind='RESIDUAL_OPPORTUNITY_FUNDING', date=o.date, session=i,
            symbols=list(self.market.symbols), nav=o.nav, actual_cash=o.cash,
            actual_units=o.units.tolist(), parent_units=original.tolist(),
            declared_units=wanted.tolist(), eligible=eligible.tolist(), prices=price.tolist(),
            risk_distances=distance.tolist(), cash_authority=cash_authority,
            parent_cash_reserved=reserved, risk_authority=risk_authority,
            remaining_cash=cash, remaining_risk=risk, risk_cap=p.risk.cap))
        result = replace(decision, unit_targets=wanted, weights=wanted*price/o.nav,
                         reason=decision.reason+'|RESIDUAL_OPPORTUNITY_FUNDING')
        result.validated_weights(len(held))
        result.validated_unit_targets(price, o.nav)
        return result

    def identity(self):
        return dict(name='opportunity_allocation', parameters=asdict(self.opportunity_parameters),
            implementation_sha256=file_hash(Path(__file__)),
            contract_sha256=file_hash(Path(__file__).with_name('opportunity_allocation_contract.json')),
            parent_definition=super().identity(), data_sha256=self.market.fingerprint(),
            status='RESEARCH_NOT_ACCEPTED')

"""Complete only the selected parent's fresh orders from residual authority.

The frozen support parent owns eligibility, obligations, stops and risk history.
Completion creates neither a new admission nor cash from an unexecuted sale.
This is a registered research candidate, not the production default.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import numpy as np
from techquant.data import Market, file_hash
from techquant.policy import CloseObservation
from research.quantity_obligation import Owner as Parent, Parameters as ParentParameters
from research.quantity_obligation import preserve_trace, verify_trace


@dataclass(frozen=True)
class Parameters:
    pass


def grid():
    return [Parameters()]


class Owner(Parent):
    def __init__(self, market: Market, parameters: Parameters):
        if type(parameters) is not Parameters:
            raise ValueError('completion requires its registered singleton parameters')
        super().__init__(market, ParentParameters('support'))
        self.completion_parameters = parameters

    def decide(self, o: CloseObservation):
        decision = super().decide(o)
        original = decision.unit_targets
        if original is None:
            raise AssertionError('the support parent must provide unit intentions')
        held = o.units > 1e-10
        fresh = ~held & (original > o.units+1e-10)
        if not fresh.any():
            return decision
        # A parent-protected close cannot reach its positive allocation path.
        # Fail closed rather than enlarge an order if that invariant changes.
        if np.any(original < o.units-1e-10):
            raise AssertionError('parent mixed a fresh admission with a protective reduction')
        p, i = self.inner, o.session
        price = np.nan_to_num(p.features.close[i], nan=0.)
        stops = np.where(held, p.stop, np.maximum(p.admission_stop(i), p.pending_stop))
        distance = np.maximum(price-stops, .02*price)
        cash_authority = max(0., min(.99*o.cash, p.risk.cap*o.nav-float(o.units@price)))
        reserved = float(np.maximum(original-o.units, 0.)@price)
        cash = max(0., cash_authority-reserved)
        risk_authority = p._risk_limit(o, p.risk.cap)
        risk = max(0., risk_authority-float(original@distance))
        desired = original.copy()
        symbol_cap = max(p.config.single_cap, 1/len(held))
        sector_cap = p.config.sector_cap if len(set(p.features.sectors)) > 1 else 1.
        candidates = p._allocation_order(i, np.flatnonzero(fresh))
        for j in candidates:
            sector = np.array([s == p.features.sectors[j] for s in p.features.sectors])
            value = min(cash, risk*price[j]/distance[j],
                max(0., symbol_cap*o.nav-desired[j]*price[j]),
                max(0., sector_cap*o.nav-float((desired*price)[sector].sum())))
            if value <= 1e-10:
                continue
            addition = value/price[j]
            desired[j] += addition
            cash = max(0., cash-value)
            risk = max(0., risk-addition*distance[j])
        self.trace.append({'kind':'ADMISSION_BUDGET_COMPLETION', 'date':o.date,
            'session':i, 'symbols':list(self.market.symbols), 'nav':o.nav,
            'actual_cash':o.cash, 'actual_units':o.units.tolist(),
            'original_units':original.tolist(), 'completed_units':desired.tolist(),
            'prices':price.tolist(), 'risk_distances':distance.tolist(),
            'cash_authority':cash_authority, 'reserved_purchases':reserved,
            'risk_authority':risk_authority, 'original_book_risk':float(original@distance),
            'remaining_cash':cash, 'remaining_risk':risk, 'risk_cap':p.risk.cap,
            'symbol_cap':symbol_cap, 'sector_cap':sector_cap})
        if np.array_equal(desired, original):
            return decision
        result = replace(decision, unit_targets=desired, weights=desired*price/o.nav,
            reason=decision.reason+'|FRESH_ADMISSION_BUDGET_COMPLETION')
        result.validated_weights(len(held))
        result.validated_unit_targets(price, o.nav)
        return result

    def identity(self):
        root = Path(__file__).parent
        return {'name':'admission_budget_completion',
            'parameters':asdict(self.completion_parameters),
            'implementation_sha256':file_hash(Path(__file__)),
            'contract_sha256':file_hash(root/'admission_budget_completion_contract.json'),
            'parent_policy':super().identity(), 'data_sha256':self.market.fingerprint(),
            'status':'RESEARCH_NOT_ACCEPTED'}

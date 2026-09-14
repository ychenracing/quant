"""Jointly allocate unchanged signal utility within real cash and stop-risk.

This research comparison changes neither eligibility nor protective authority.
A maximum of two purchase legs makes exact polygon enumeration sufficient;
no optimizer dependency, forecast fit or hypothetical sale proceeds are used.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
import numpy as np
from techquant.config import Config
from techquant.data import Market, file_hash
from research.observed_admission_completion import Owner as Parent, Parameters as ParentParameters
from research.quantity_obligation import SupportIntent, preserve_trace, verify_trace


@dataclass(frozen=True)
class Parameters:
    joint: bool = True

    def __post_init__(self):
        if type(self.joint) is not bool:
            raise ValueError('joint must be a registered boolean candidate')


def grid():
    return [Parameters(False), Parameters(True)]


_INTERSECTIONS = np.array(list(combinations(range(6), 2)))


def _pair(scores, fractions, upper, minimum, cash, risk):
    """Return the utility-maximizing feasible vertex for two included legs."""
    a = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.],
                  [1., 1.], fractions])
    b = np.array([upper[0], -minimum[0], upper[1], -minimum[1], cash, risk])
    if (np.any(upper < minimum) or minimum.sum() > cash
            or float(minimum @ fractions) > risk):
        return None
    first, second = _INTERSECTIONS.T
    det = a[first, 0]*a[second, 1]-a[first, 1]*a[second, 0]
    valid = np.abs(det) > 1e-15
    first, second, det = first[valid], second[valid], det[valid]
    points = np.column_stack(((b[first]*a[second, 1]-a[first, 1]*b[second])/det,
                             (a[first, 0]*b[second]-b[first]*a[second, 0])/det))
    points = points[np.all(points @ a.T <= b+1e-12, axis=1)]
    if not len(points):
        return None
    point = points[np.argmax(points @ scores)]
    # Roundoff at an intersection cannot create additional resource authority.
    scale = min(1., cash/max(float(point.sum()), 1e-300),
                risk/max(float(point @ fractions), 1e-300),
                float(np.min(upper/np.maximum(point, 1e-300))))
    point = np.maximum(0., point*scale)
    return point if np.all(point >= minimum-1e-12) else None


def purchase_plan(*, scores, fractions, upper, minimum, sectors, sector_slack,
                  cash, risk, held, capacity, symbols):
    """Optimize incremental notionals as NAV fractions; never sell inventory."""
    scores, fractions, upper, minimum = (np.asarray(x, dtype=float)
                                         for x in (scores, fractions, upper, minimum))
    held = np.asarray(held, dtype=bool)
    n = len(symbols)
    if (not n or any(x.shape != (n,) for x in (scores, fractions, upper, minimum, held))
            or len(sectors) != n or len(set(symbols)) != n
            or type(capacity) is not int or not 1 <= capacity <= 2
            or not np.isfinite([cash, risk]).all() or min(cash, risk) < 0
            or not all(np.isfinite(x).all() for x in (scores, fractions, upper, minimum))
            or np.any(fractions <= 0) or np.any(upper < 0) or np.any(minimum <= 0)
            or any(s not in sector_slack or not np.isfinite(sector_slack[s])
                   or sector_slack[s] < 0 for s in sectors)):
        raise ValueError('invalid joint purchase resources')
    answer = np.zeros(n)
    free = max(0, capacity-int(held.sum()))
    upper = np.minimum(upper, np.array([sector_slack[s] for s in sectors]))
    available = sorted(np.flatnonzero((scores > 0) & (upper >= minimum)
                                      & (held | (free > 0))), key=lambda j: symbols[j])
    best, best_key = 0., ()
    for size in (1, 2):
        for group in combinations(available, size):
            ids = np.array(group)
            if np.count_nonzero(~held[ids]) > free:
                continue
            budget = min(cash, sector_slack[sectors[ids[0]]]) if size == 1 or (
                sectors[ids[0]] == sectors[ids[-1]]) else cash
            if size == 1:
                amount = min(upper[ids[0]], budget, risk/fractions[ids[0]])
                if amount < minimum[ids[0]]:
                    continue
                values = np.array([amount])
            else:
                values = _pair(scores[ids], fractions[ids], upper[ids], minimum[ids], budget, risk)
                if values is None:
                    continue
            objective = float(scores[ids] @ values)
            key = tuple(symbols[j] for j in ids)
            if objective > best or (objective == best and key < best_key):
                answer[:] = 0.
                answer[ids] = values
                best, best_key = objective, key
    return answer


class JointSupport(SupportIntent):
    def _allocate(self, o, desired, price, initial, allowed, cap):
        if np.any(desired < o.units-1e-10):
            raise AssertionError('a protective reduction cannot enter the allocation auction')
        held = o.units > 1e-10
        stops = np.where(held, self.stop, np.maximum(initial, self.pending_stop))
        distance = np.maximum(price-stops, .02*price)
        fractions = np.divide(distance, price, out=np.ones_like(price), where=price > 0)
        cash = max(0., min(.99*o.cash, cap*o.nav-float(o.units @ price)))/o.nav
        risk = max(0., self._risk_limit(o, cap)-float(o.units @ distance))/o.nav
        weights = o.units*price/o.nav
        name_cap = max(self.config.single_cap, 1/len(held))
        sector_cap = self.config.sector_cap if len(set(self.features.sectors)) > 1 else 1.
        slack = {s:max(0., sector_cap-float(weights[np.array(self.features.sectors)==s].sum()))
                 for s in self.features.sectors}
        eligible = allowed & (price > stops) & (~held | self.breakout[o.session])
        score = np.where(eligible & np.isfinite(self.features.score[o.session]),
                         self.features.score[o.session], 0.)
        upper = np.maximum(0., name_cap-weights)
        minimum = np.where(held, self.config.trade_band, .01)
        # With an explicit zero trade band, use the parent's positive-value floor.
        minimum = np.maximum(minimum, 1e-10/o.nav)
        bought = purchase_plan(scores=score, fractions=fractions, upper=upper, minimum=minimum,
            sectors=self.features.sectors, sector_slack=slack, cash=cash, risk=risk,
            held=held, capacity=min(self.params.positions,len(held)), symbols=self.market.symbols)
        result = desired+np.divide(bought*o.nav,price,out=np.zeros_like(price),where=price>0)
        selected = bought > 0
        self.pending_stop[selected] = np.maximum.reduce([
            self.pending_stop[selected], initial[selected], self.stop[selected]])
        self.auction_trace.append({'kind':'JOINT_FUNDED_ALLOCATION','date':o.date,
            'session':o.session,'symbols':list(self.market.symbols),'nav':o.nav,
            'actual_cash':o.cash,'actual_units':o.units.tolist(),'proposed_units':result.tolist(),
            'scores':score.tolist(),'risk_fractions':fractions.tolist(),
            'purchase_weights':bought.tolist(),'upper_bounds':upper.tolist(),
            'minimum_weights':minimum.tolist(),'sector_slack':slack,
            'cash_fraction':cash,'risk_fraction':risk,'risk_cap':cap,
            'actual_occupied_slots':int(held.sum()),'signal_utility':float(score@bought)})
        return result


class Owner(Parent):
    def __init__(self, market: Market, parameters: Parameters, *, config: Config | None = None):
        if type(parameters) is not Parameters:
            raise ValueError('only the registered allocation comparison is supported')
        super().__init__(market, ParentParameters(), config=config)
        self.auction_parameters = parameters
        if parameters.joint:
            old = self.inner
            inner = JointSupport(market, old.params, config=old.config)
            # Apply the already computed readiness estimates at construction only.
            # Every live state array is initialized normally, never replaced later.
            inner.features, inner.atr, inner.support, inner.ready = (
                old.features, old.atr, old.support, old.ready)
            inner.auction_trace = self.trace
            self.inner = inner

    def identity(self):
        return {'name':'allocation_auction','parameters':asdict(self.auction_parameters),
                'implementation_sha256':file_hash(Path(__file__)),
                'contract_sha256':file_hash(Path(__file__).with_name('allocation_auction_contract.json')),
                'parent_definition':super().identity(),'data_sha256':self.market.fingerprint(),
                'status':'RESEARCH_NOT_ACCEPTED'}

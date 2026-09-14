"""One scheduled economic reassignment, subordinate to the actual funded book."""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import numpy as np
from techquant.data import Market, file_hash
from techquant.policy import CloseObservation
from research.quantity_obligation import (
    Owner as Parent, Parameters as ParentParameters, preserve_trace, verify_trace,
)


@dataclass(frozen=True)
class Parameters:
    pass


def grid():
    return [Parameters()]


class Owner:
    def __init__(self, market: Market, parameters: Parameters):
        self.market, self.parameters = market, parameters
        self.parent = Parent(market, ParentParameters('support'))
        self.inner = self.parent.inner
        self.trace = self.parent.trace
        self.held_since = np.full(len(market.symbols), -1, dtype=int)
        self.pending_reassignment: int | None = None

    def decide(self, observation: CloseObservation):
        o, p = observation, self.inner
        held = o.units > 1e-10
        unresolved = bool(np.any(held & (p.exit_pending |
                              (o.units > p.reduction_ceiling+1e-10))))
        # Actual first observation, not a submitted order, establishes holding age.
        decision = self.parent.decide(o)
        self.held_since[~held] = -1
        self.held_since[held & (self.held_since < 0)] = o.session
        if self.pending_reassignment is not None:
            weak = self.pending_reassignment
            complete = bool(not held[weak])
            self.trace.append({'kind': 'OPPORTUNITY_EXIT_OBSERVED',
                'date': o.date, 'session': o.session,
                'incumbent': self.market.symbols[weak],
                'actual_units': float(o.units[weak]), 'fulfilled': complete,
                'authority': float(p.risk.cap)})
            if not complete:
                if (decision.unit_targets[weak] > 1e-10 or
                        np.any(decision.unit_targets > o.units+1e-10)):
                    raise AssertionError('unfulfilled reassignment cannot authorize purchases')
                return replace(decision, reason=decision.reason+'|OPPORTUNITY_EXIT_PENDING')
            self.pending_reassignment = None

        if (o.session % p.config.rebalance != 0 or p.risk.cap != 1. or
                int(held.sum()) != p.params.positions or unresolved or
                np.any(held & p.exit_pending) or
                not np.allclose(decision.unit_targets, o.units, rtol=0., atol=1e-10)):
            return decision
        i = o.session
        score = p.features.score[i]
        if not np.isfinite(score[held]).all():
            return decision
        weak = min(np.flatnonzero(held), key=lambda j: (score[j], self.market.symbols[j]))
        age = i-int(self.held_since[weak])
        if age < p.config.rebalance:
            return decision
        ranking = sorted(np.flatnonzero(p.ready[i] & np.isfinite(score)),
                         key=lambda j: (-score[j], self.market.symbols[j]))
        if weak not in ranking or ranking.index(weak) < p.params.positions:
            return decision
        price = np.nan_to_num(p.features.close[i], nan=0.)
        stops = np.maximum(p.admission_stop(i), p.pending_stop)
        eligible = (~held & p.ready[i] & p.features.entry[i] & ~p.features.exit[i]
                    & ~p.readmit & np.isfinite(score) & (score > 0) & (price > stops))
        choices = np.flatnonzero(eligible)
        if not len(choices):
            return decision
        best = min(choices, key=lambda j: (-score[j], self.market.symbols[j]))
        if score[best] <= 1.25*max(0., score[weak]):
            return decision
        desired = decision.unit_targets.copy()
        desired[weak] = 0.
        p.reduction_ceiling[weak] = 0.
        p.exit_pending[weak] = True
        self.pending_reassignment = int(weak)
        self.trace.append({'kind': 'OPPORTUNITY_REASSIGNMENT', 'date': o.date,
            'session': i, 'incumbent': self.market.symbols[weak],
            'challenger': self.market.symbols[best],
            'incumbent_score': float(score[weak]), 'challenger_score': float(score[best]),
            'held_sessions': age, 'held_rank': ranking.index(weak)+1,
            'authority': float(p.risk.cap), 'actual_units': o.units.tolist(),
            'economic_target': 0., 'classification': 'VOLUNTARY_ECONOMIC_EXIT_NOT_RISK_ALARM'})
        result = replace(decision, weights=desired*price/o.nav, unit_targets=desired,
                         reason=decision.reason+'|SCHEDULED_OPPORTUNITY_REASSIGNMENT')
        result.validated_weights(len(held))
        result.validated_unit_targets(price, o.nav)
        return result

    def identity(self):
        return {'name': 'scheduled_opportunity_reassignment', 'parameters': asdict(self.parameters),
            'implementation_sha256': file_hash(Path(__file__)),
            'contract_sha256': file_hash(Path(__file__).with_name('opportunity_reassignment_contract.json')),
            'parent_policy': self.parent.identity(), 'data_sha256': self.market.fingerprint(),
            'status': 'RESEARCH_NOT_ACCEPTED'}

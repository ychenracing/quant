"""One preregistered, fill-aware economic handover above the unchanged parent.

A confirmed score difference may terminate an intact holding; it neither
reserves a replacement nor authorizes spending unobserved sale proceeds.
"""
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
        self.confirmed_pair: tuple[int, int] | None = None
        self.confirmation = 0
        self.pending_handover: int | None = None
        self.last_handover = -self.inner.config.rebalance

    def _reset_confirmation(self):
        self.confirmed_pair, self.confirmation = None, 0

    def decide(self, observation: CloseObservation):
        o, p = observation, self.inner
        held = o.units > 1e-10
        prior_session, prior_cap = p.last_session, p.risk.cap
        unresolved = bool(np.any(held & (p.exit_pending |
                              (o.units > p.reduction_ceiling + 1e-10))))
        # Always advance actual-account risk, stops and obligations exactly once.
        decision = self.parent.decide(o)
        if self.pending_handover is not None:
            weak = self.pending_handover
            fulfilled = bool(not held[weak])
            self.trace.append({'kind': 'LEADERSHIP_EXIT_OBSERVED',
                'date': o.date, 'session': o.session,
                'incumbent': self.market.symbols[weak],
                'actual_units': float(o.units[weak]), 'fulfilled': fulfilled,
                'authority': float(p.risk.cap)})
            if not fulfilled:
                # The same zero ceiling is already enforced by the parent.
                # Fail closed if a future parent change breaks that contract.
                if (decision.unit_targets[weak] > 1e-10 or
                        np.any(decision.unit_targets > o.units + 1e-10)):
                    raise AssertionError('unfulfilled handover cannot authorize purchases')
                self._reset_confirmation()
                return replace(decision, reason=decision.reason+'|LEADERSHIP_EXIT_PENDING')
            self.pending_handover = None

        intact = np.allclose(decision.unit_targets, o.units, rtol=0., atol=1e-10)
        if (int(held.sum()) != p.params.positions or p.risk.cap <= 0 or
                p.risk.cap < prior_cap-1e-12 or unresolved or not intact or
                np.any(held & p.exit_pending)):
            self._reset_confirmation()
            return decision
        i = o.session
        price = np.nan_to_num(p.features.close[i], nan=0.)
        score = p.features.score[i]
        # This is the parent's original unheld admission gate, not a new ranker.
        stops = np.maximum(p.admission_stop(i), p.pending_stop)
        eligible = (~held & p.ready[i] & p.features.entry[i] & ~p.features.exit[i]
                    & ~p.readmit & np.isfinite(score) & (score > 0) & (price > stops))
        challengers = np.flatnonzero(eligible)
        if not len(challengers) or not np.isfinite(score[held]).all():
            self._reset_confirmation()
            return decision
        best = min(challengers, key=lambda j: (-score[j], self.market.symbols[j]))
        weak = min(np.flatnonzero(held), key=lambda j: (score[j], self.market.symbols[j]))
        if score[best] <= 2*max(0., score[weak]):
            self._reset_confirmation()
            return decision
        pair = (int(best), int(weak))
        self.confirmation = (self.confirmation+1 if self.confirmed_pair == pair
                             and i == prior_session+1 else 1)
        self.confirmed_pair = pair
        if (self.confirmation < p.config.recovery or
                i-self.last_handover < p.config.rebalance):
            return decision

        desired = decision.unit_targets.copy()
        desired[weak] = 0.
        # Keep the economic zero obligation until the ledger reports zero.
        # The other holding, risk state, campaign stop and cash stay untouched.
        p.reduction_ceiling[weak] = 0.
        p.exit_pending[weak] = True
        self.pending_handover, self.last_handover = int(weak), i
        self.trace.append({'kind': 'LEADERSHIP_HANDOVER', 'date': o.date,
            'session': i, 'incumbent': self.market.symbols[weak],
            'challenger': self.market.symbols[best],
            'incumbent_score': float(score[weak]), 'challenger_score': float(score[best]),
            'confirmation_count': self.confirmation, 'authority': float(p.risk.cap),
            'actual_units': o.units.tolist(), 'economic_target': 0.,
            'classification': 'VOLUNTARY_ECONOMIC_EXIT_NOT_RISK_ALARM'})
        self._reset_confirmation()
        result = replace(decision, weights=desired*price/o.nav, unit_targets=desired,
                         reason=decision.reason+'|CONFIRMED_LEADERSHIP_HANDOVER')
        result.validated_weights(len(held))
        result.validated_unit_targets(price, o.nav)
        return result

    def identity(self):
        return {'name': 'confirmed_leadership_handover', 'parameters': asdict(self.parameters),
            'implementation_sha256': file_hash(Path(__file__)),
            'contract_sha256': file_hash(Path(__file__).with_name('leadership_handover_contract.json')),
            'parent_policy': self.parent.identity(), 'data_sha256': self.market.fingerprint(),
            'status': 'RESEARCH_NOT_ACCEPTED'}

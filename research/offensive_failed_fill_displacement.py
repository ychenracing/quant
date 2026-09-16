"""Settlement-aware reference rearm plus execution-confirmed failed-fill displacement.

A nominal vacancy only becomes evidence of capital starvation after the base owner
has requested a fill and the next observed inventory proves no new position was
created. The rule then sells first and never spends forecast sale proceeds.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import numpy as np

from techquant.data import Market, file_hash
from techquant.policy import CloseDecision, CloseObservation
from research.offensive_alpha_decay_displacement import Owner as BaseOwner, Parameters as BaseParameters
from research.quantity_obligation import preserve_trace, verify_trace


@dataclass(frozen=True)
class Parameters:
    enabled: bool

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('enabled must be boolean')


def grid():
    return [Parameters(False), Parameters(True)]


class Owner:
    def __init__(self, market: Market, parameters: Parameters):
        if type(parameters) is not Parameters:
            raise ValueError('failed-fill displacement requires registered parameters')
        self.market = market
        self.parameters = parameters
        self.base = BaseOwner(market, BaseParameters(parameters.enabled))
        n = len(market.symbols)
        self.invalidated = np.zeros(n, dtype=bool)
        self.saw_nonentry = np.zeros(n, dtype=bool)
        self.invalidated_since = np.full(n, -1, dtype=int)
        self.failed_reference = np.full(n, np.nan)
        self.failed_fill_attempt_session = -1
        self.failed_fill_attempt_held_count = -1

    @property
    def trace(self):
        return self.base.trace

    def _event(self, o: CloseObservation, action: str, mask: np.ndarray | None = None, **extra) -> None:
        if mask is not None and not np.any(mask):
            return
        event = {
            'kind': 'FAILED_FILL_DISPLACEMENT_EVENT',
            'date': o.date,
            'session': int(o.session),
            'action': action,
        }
        if mask is not None:
            event['symbols'] = [self.market.symbols[j] for j in np.flatnonzero(mask)]
        event.update(extra)
        self.base.trace.append(event)

    def _clear_campaign(self, o: CloseObservation, action: str, mask: np.ndarray) -> None:
        if not np.any(mask):
            return
        self._event(o, action, mask)
        self.invalidated[mask] = False
        self.saw_nonentry[mask] = False
        self.invalidated_since[mask] = -1
        self.failed_reference[mask] = np.nan

    def _settled_reference_state(self, o: CloseObservation, held: np.ndarray, score: np.ndarray):
        i = o.session
        p = self.base.price_signals
        acute = p.ret1[i] <= -.08
        existing = held | self.base.retired
        newly = acute & existing & ~self.invalidated
        if np.any(newly):
            refs = self.base.owned_alpha_reference.copy()
            pending_held = held & ~np.isfinite(refs) & np.isfinite(self.base.pending_alpha_reference)
            refs[pending_held] = self.base.pending_alpha_reference[pending_held]
            finite = newly & np.isfinite(refs)
            self.failed_reference[finite] = refs[finite]
            self.invalidated[newly] = True
            self.saw_nonentry[newly] = False
            self.invalidated_since[newly] = i
            self._event(o, 'ACUTE_REFERENCE_INVALIDATION', newly)

        later_flat = self.invalidated & ~held & (i > self.invalidated_since)
        became_nonentry = later_flat & ~self.base.trend.entry[i]
        newly_nonentry = became_nonentry & ~self.saw_nonentry
        self.saw_nonentry[became_nonentry] = True
        self._event(o, 'FRESH_EPOCH_ARMED', newly_nonentry)

        broken = self.base.trend.exit[i] | (p.ret1[i] <= -.08) | ~p.ready[i]
        ordinary = (
            p.ready[i] & self.base.trend.entry[i] & (p.price[i] > p.ema20[i]) &
            (p.momentum5[i] > 0) & np.isfinite(score) & (score > 0) &
            ~held & ~broken & ~self.base.retired
        )
        recovered = later_flat & ordinary & np.isfinite(self.failed_reference) & (score > self.failed_reference)
        settlement_blocked = recovered & self.base.was_held
        self._event(o, 'SETTLEMENT_REARM_BLOCKED', settlement_blocked)
        self._clear_campaign(o, 'REFERENCE_ALPHA_REARM', recovered & ~self.base.was_held)
        edge = (
            self.invalidated & ~held & (i > self.invalidated_since) & self.saw_nonentry &
            self.base.trend.entry[i] & ~self.base.was_held
        )
        self._clear_campaign(o, 'TREND_EDGE_REARM', edge)
        return broken, ordinary

    def decide(self, o: CloseObservation):
        if not self.parameters.enabled:
            return self.base.decide(o)

        i = o.session
        held = o.units > 1e-10
        held_count = int(held.sum())
        p = self.base.price_signals
        score = self.base.features.score[i]
        confirmed_failed_fill = (
            self.failed_fill_attempt_session == i - 1 and
            held_count <= self.failed_fill_attempt_held_count and
            o.nav > 0 and o.cash < o.nav * .01
        )
        prior_attempt_session = self.failed_fill_attempt_session
        self.failed_fill_attempt_session = -1
        self.failed_fill_attempt_held_count = -1

        _, ordinary = self._settled_reference_state(o, held, score)
        retry = self.invalidated & held & (i > self.invalidated_since)
        if np.any(retry):
            if i <= self.base.last_session:
                raise ValueError('policy sessions must increase')
            self.base.last_session = i
            self.base._observe_inventory(held)
            units = o.units.copy(); units[retry] = 0.
            marks = np.nan_to_num(p.price[i], nan=0.)
            weights = np.divide(units * marks, o.nav, out=np.zeros_like(units), where=o.nav > 0)
            self._event(o, 'INVALIDATED_EXIT_RETRY', retry)
            return CloseDecision(weights, 'FAILED_FILL|INVALIDATED_EXIT_RETRY', 1., units)

        vacancy = max(0, self.base.config.max_positions - held_count)
        if confirmed_failed_fill and vacancy > 0 and self.base.trend.market[i]:
            refs = self.base.owned_alpha_reference.copy()
            pending_new = held & ~self.base.was_held & ~np.isfinite(refs) & np.isfinite(self.base.pending_alpha_reference)
            refs[pending_new] = self.base.pending_alpha_reference[pending_new]
            decay = score - refs
            decayed = held & np.isfinite(decay) & (decay < 0)
            challengers = np.flatnonzero(ordinary & ~self.invalidated)
            if decayed.any() and len(challengers):
                incumbent = min(np.flatnonzero(decayed), key=lambda j: (decay[j], self.market.symbols[j]))
                challenger = min(challengers, key=lambda j: (-score[j], self.market.symbols[j]))
                if score[challenger] > refs[incumbent]:
                    if i <= self.base.last_session:
                        raise ValueError('policy sessions must increase')
                    self.base.last_session = i
                    self.base._observe_inventory(held)
                    units = o.units.copy(); units[incumbent] = 0.
                    marks = np.nan_to_num(p.price[i], nan=0.)
                    weights = np.divide(units * marks, o.nav, out=np.zeros_like(units), where=o.nav > 0)
                    self.base.retired[incumbent] = True
                    self.base.retired_since[incumbent] = int(i)
                    self.base.trace.append({
                        'kind': 'FAILED_FILL_DISPLACEMENT_EVENT', 'date': o.date, 'session': int(i),
                        'action': 'FAILED_FILL_DISPLACEMENT', 'symbol': self.market.symbols[incumbent],
                        'challenger': self.market.symbols[challenger],
                        'prior_attempt_session': int(prior_attempt_session),
                        'observed_cash': float(o.cash), 'observed_nav': float(o.nav),
                        'owned_alpha_reference': float(refs[incumbent]),
                        'incumbent_score': float(score[incumbent]), 'challenger_score': float(score[challenger]),
                    })
                    return CloseDecision(weights, 'FAILED_FILL_DISPLACEMENT', 1., units)
            self._event(o, 'FAILED_FILL_CONFIRMED_NO_DISPLACEMENT', None,
                        prior_attempt_session=int(prior_attempt_session), observed_cash=float(o.cash),
                        observed_nav=float(o.nav), held_count=held_count)

        original = self.base.features
        if np.any(self.invalidated):
            masked = original.score.copy(); masked[i, self.invalidated] = np.nan
            self.base.features = replace(original, score=masked)
        trace_before = len(self.base.trace)
        try:
            decision = self.base.decide(o)
        finally:
            self.base.features = original
        vacancy_fill_requested = any(row.get('action') == 'VACANCY_FILL' for row in self.base.trace[trace_before:])
        if vacancy > 0 and o.nav > 0 and o.cash < o.nav * .01 and vacancy_fill_requested:
            self.failed_fill_attempt_session = i
            self.failed_fill_attempt_held_count = held_count
            self._event(o, 'NONDEPLOYABLE_FILL_ATTEMPT', None, held_count=held_count,
                        observed_cash=float(o.cash), observed_nav=float(o.nav))
        return decision

    def identity(self):
        root = Path(__file__).parent
        return {
            'name': 'offensive_failed_fill_displacement', 'parameters': asdict(self.parameters),
            'implementation_sha256': file_hash(Path(__file__)),
            'contract_sha256': file_hash(root/'offensive_failed_fill_displacement_contract.json'),
            'base_sha256': file_hash(root/'offensive_alpha_decay_displacement.py'),
            'data_sha256': self.market.fingerprint(), 'status': 'RESEARCH_NOT_ACCEPTED',
        }


__all__=['Owner','Parameters','grid','preserve_trace','verify_trace']

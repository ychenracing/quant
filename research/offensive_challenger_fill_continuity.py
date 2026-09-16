"""Execution-confirmed displacement bound to the exact failed challenger.

The base owner may try to fill a nominal vacancy. A failed attempt can release
an incumbent only if that exact requested challenger remains the strongest
eligible opportunity on the next observed close. No forecast proceeds are used.
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
            raise ValueError('challenger fill continuity requires registered parameters')
        self.market = market; self.parameters = parameters
        self.base = BaseOwner(market, BaseParameters(parameters.enabled))
        n = len(market.symbols)
        self.invalidated = np.zeros(n, dtype=bool); self.saw_nonentry = np.zeros(n, dtype=bool)
        self.invalidated_since = np.full(n, -1, dtype=int); self.failed_reference = np.full(n, np.nan)
        self.failed_fill_attempt_session = -1; self.failed_fill_attempt_held_count = -1; self.failed_fill_attempt_challenger = -1

    @property
    def trace(self): return self.base.trace

    def _event(self, o: CloseObservation, action: str, mask: np.ndarray | None = None, **extra) -> None:
        if mask is not None and not np.any(mask): return
        event = {'kind':'CHALLENGER_FILL_CONTINUITY_EVENT','date':o.date,'session':int(o.session),'action':action}
        if mask is not None: event['symbols']=[self.market.symbols[j] for j in np.flatnonzero(mask)]
        event.update(extra); self.base.trace.append(event)

    def _clear_campaign(self, o, action, mask):
        if not np.any(mask): return
        self._event(o,action,mask); self.invalidated[mask]=False; self.saw_nonentry[mask]=False
        self.invalidated_since[mask]=-1; self.failed_reference[mask]=np.nan

    def _settled_reference_state(self, o, held, score):
        i=o.session; p=self.base.price_signals
        acute=p.ret1[i] <= -.08; existing=held | self.base.retired; newly=acute & existing & ~self.invalidated
        if np.any(newly):
            refs=self.base.owned_alpha_reference.copy(); pending=held & ~np.isfinite(refs) & np.isfinite(self.base.pending_alpha_reference)
            refs[pending]=self.base.pending_alpha_reference[pending]; finite=newly & np.isfinite(refs); self.failed_reference[finite]=refs[finite]
            self.invalidated[newly]=True; self.saw_nonentry[newly]=False; self.invalidated_since[newly]=i; self._event(o,'ACUTE_REFERENCE_INVALIDATION',newly)
        later=self.invalidated & ~held & (i>self.invalidated_since); nonentry=later & ~self.base.trend.entry[i]; newly_nonentry=nonentry & ~self.saw_nonentry
        self.saw_nonentry[nonentry]=True; self._event(o,'FRESH_EPOCH_ARMED',newly_nonentry)
        broken=self.base.trend.exit[i] | (p.ret1[i] <= -.08) | ~p.ready[i]
        ordinary=(p.ready[i] & self.base.trend.entry[i] & (p.price[i]>p.ema20[i]) & (p.momentum5[i]>0) & np.isfinite(score) & (score>0) & ~held & ~broken & ~self.base.retired)
        recovered=later & ordinary & np.isfinite(self.failed_reference) & (score>self.failed_reference)
        self._event(o,'SETTLEMENT_REARM_BLOCKED',recovered & self.base.was_held); self._clear_campaign(o,'REFERENCE_ALPHA_REARM',recovered & ~self.base.was_held)
        edge=(self.invalidated & ~held & (i>self.invalidated_since) & self.saw_nonentry & self.base.trend.entry[i] & ~self.base.was_held)
        self._clear_campaign(o,'TREND_EDGE_REARM',edge); return ordinary

    def decide(self, o: CloseObservation):
        if not self.parameters.enabled: return self.base.decide(o)
        i=o.session; held=o.units>1e-10; held_count=int(held.sum()); p=self.base.price_signals; score=self.base.features.score[i]
        prior_session=self.failed_fill_attempt_session; prior_count=self.failed_fill_attempt_held_count; prior_challenger=self.failed_fill_attempt_challenger
        confirmed=(prior_session==i-1 and prior_challenger>=0 and held_count<=prior_count and not held[prior_challenger] and o.nav>0 and o.cash<o.nav*.01)
        self.failed_fill_attempt_session=-1; self.failed_fill_attempt_held_count=-1; self.failed_fill_attempt_challenger=-1
        ordinary=self._settled_reference_state(o,held,score)
        retry=self.invalidated & held & (i>self.invalidated_since)
        if np.any(retry):
            if i<=self.base.last_session: raise ValueError('policy sessions must increase')
            self.base.last_session=i; self.base._observe_inventory(held); units=o.units.copy(); units[retry]=0.; marks=np.nan_to_num(p.price[i],nan=0.)
            weights=np.divide(units*marks,o.nav,out=np.zeros_like(units),where=o.nav>0); self._event(o,'INVALIDATED_EXIT_RETRY',retry)
            return CloseDecision(weights,'CHALLENGER_FILL_CONTINUITY|INVALIDATED_EXIT_RETRY',1.,units)
        vacancy=max(0,self.base.config.max_positions-held_count)
        if confirmed and vacancy>0 and self.base.trend.market[i]:
            choices=np.flatnonzero(ordinary & ~self.invalidated); best=(min(choices,key=lambda j:(-score[j],self.market.symbols[j])) if len(choices) else -1)
            if best != prior_challenger:
                self._event(o,'CHALLENGER_IDENTITY_CHANGED',None,prior_challenger=self.market.symbols[prior_challenger],current_best=(self.market.symbols[best] if best>=0 else None),prior_attempt_session=int(prior_session))
            else:
                refs=self.base.owned_alpha_reference.copy(); pending=held & ~self.base.was_held & ~np.isfinite(refs) & np.isfinite(self.base.pending_alpha_reference); refs[pending]=self.base.pending_alpha_reference[pending]
                decay=score-refs; decayed=held & np.isfinite(decay) & (decay<0)
                if decayed.any():
                    incumbent=min(np.flatnonzero(decayed),key=lambda j:(decay[j],self.market.symbols[j]))
                    if score[best] > refs[incumbent]:
                        if i<=self.base.last_session: raise ValueError('policy sessions must increase')
                        self.base.last_session=i; self.base._observe_inventory(held); units=o.units.copy(); units[incumbent]=0.; marks=np.nan_to_num(p.price[i],nan=0.)
                        weights=np.divide(units*marks,o.nav,out=np.zeros_like(units),where=o.nav>0); self.base.retired[incumbent]=True; self.base.retired_since[incumbent]=int(i)
                        self.base.trace.append({'kind':'CHALLENGER_FILL_CONTINUITY_EVENT','date':o.date,'session':int(i),'action':'IDENTITY_CONTINUOUS_DISPLACEMENT','symbol':self.market.symbols[incumbent],'challenger':self.market.symbols[best],'prior_attempt_session':int(prior_session),'observed_cash':float(o.cash),'observed_nav':float(o.nav),'owned_alpha_reference':float(refs[incumbent]),'incumbent_score':float(score[incumbent]),'challenger_score':float(score[best])})
                        return CloseDecision(weights,'CHALLENGER_FILL_CONTINUITY|DISPLACEMENT',1.,units)
                self._event(o,'IDENTITY_CONTINUITY_NO_DISPLACEMENT',None,challenger=self.market.symbols[prior_challenger],prior_attempt_session=int(prior_session))
        original=self.base.features
        if np.any(self.invalidated):
            masked=original.score.copy(); masked[i,self.invalidated]=np.nan; self.base.features=replace(original,score=masked)
        before=len(self.base.trace)
        try: decision=self.base.decide(o)
        finally: self.base.features=original
        fills=[r for r in self.base.trace[before:] if r.get('action')=='VACANCY_FILL']
        if vacancy>0 and o.nav>0 and o.cash<o.nav*.01 and fills:
            requested=fills[-1].get('symbols',[])
            if len(requested)==1:
                j=self.market.symbols.index(requested[0]); self.failed_fill_attempt_session=i; self.failed_fill_attempt_held_count=held_count; self.failed_fill_attempt_challenger=j
                self._event(o,'NONDEPLOYABLE_CHALLENGER_ATTEMPT',None,challenger=requested[0],held_count=held_count,observed_cash=float(o.cash),observed_nav=float(o.nav))
        return decision

    def identity(self):
        root=Path(__file__).parent
        return {'name':'offensive_challenger_fill_continuity','parameters':asdict(self.parameters),'implementation_sha256':file_hash(Path(__file__)),'contract_sha256':file_hash(root/'offensive_challenger_fill_continuity_contract.json'),'base_sha256':file_hash(root/'offensive_alpha_decay_displacement.py'),'data_sha256':self.market.fingerprint(),'status':'RESEARCH_NOT_ACCEPTED'}


__all__=['Owner','Parameters','grid','preserve_trace','verify_trace']

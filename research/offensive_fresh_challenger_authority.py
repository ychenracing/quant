"""Failed-fill challenger continuity with campaign-provenance authority.

A reference-rearmed failed campaign remains ordinarily admissible, but until it
forms actual inventory or a fresh trend epoch it cannot force liquidation of a
funded incumbent. This changes displacement authority, not entry eligibility.
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
        if type(self.enabled) is not bool: raise ValueError('enabled must be boolean')

def grid(): return [Parameters(False), Parameters(True)]

class Owner:
    def __init__(self, market: Market, parameters: Parameters):
        if type(parameters) is not Parameters: raise ValueError('fresh challenger authority requires registered parameters')
        self.market=market; self.parameters=parameters; self.base=BaseOwner(market,BaseParameters(parameters.enabled)); n=len(market.symbols)
        self.invalidated=np.zeros(n,dtype=bool); self.saw_nonentry=np.zeros(n,dtype=bool); self.invalidated_since=np.full(n,-1,dtype=int); self.failed_reference=np.full(n,np.nan)
        self.reference_rearmed_without_fresh_epoch=np.zeros(n,dtype=bool); self.recovered_saw_nonentry=np.zeros(n,dtype=bool)
        self.failed_fill_attempt_session=-1; self.failed_fill_attempt_held_count=-1; self.failed_fill_attempt_challenger=-1
    @property
    def trace(self): return self.base.trace
    def _event(self,o,action,mask=None,**extra):
        if mask is not None and not np.any(mask): return
        e={'kind':'FRESH_CHALLENGER_AUTHORITY_EVENT','date':o.date,'session':int(o.session),'action':action}
        if mask is not None: e['symbols']=[self.market.symbols[j] for j in np.flatnonzero(mask)]
        e.update(extra); self.base.trace.append(e)
    def _clear_campaign(self,o,action,mask):
        if not np.any(mask): return
        self._event(o,action,mask)
        if action=='REFERENCE_ALPHA_REARM': self.reference_rearmed_without_fresh_epoch[mask]=True; self.recovered_saw_nonentry[mask]=False
        elif action=='TREND_EDGE_REARM': self.reference_rearmed_without_fresh_epoch[mask]=False; self.recovered_saw_nonentry[mask]=False
        self.invalidated[mask]=False; self.saw_nonentry[mask]=False; self.invalidated_since[mask]=-1; self.failed_reference[mask]=np.nan
    def _update_recovered_provenance(self,o,held):
        i=o.session; actual=self.reference_rearmed_without_fresh_epoch & held
        if np.any(actual): self._event(o,'RECOVERED_CAMPAIGN_ACTUAL_ESTABLISHMENT',actual); self.reference_rearmed_without_fresh_epoch[actual]=False; self.recovered_saw_nonentry[actual]=False
        marked=self.reference_rearmed_without_fresh_epoch.copy(); observed_false=marked & ~held & ~self.base.trend.entry[i]; self.recovered_saw_nonentry[observed_false]=True
        fresh=marked & ~held & self.recovered_saw_nonentry & self.base.trend.entry[i]
        if np.any(fresh): self._event(o,'RECOVERED_CAMPAIGN_FRESH_EPOCH',fresh); self.reference_rearmed_without_fresh_epoch[fresh]=False; self.recovered_saw_nonentry[fresh]=False
    def _settled_reference_state(self,o,held,score):
        i=o.session; p=self.base.price_signals; acute=p.ret1[i] <= -.08; existing=held | self.base.retired; newly=acute & existing & ~self.invalidated
        if np.any(newly):
            refs=self.base.owned_alpha_reference.copy(); pending=held & ~np.isfinite(refs) & np.isfinite(self.base.pending_alpha_reference); refs[pending]=self.base.pending_alpha_reference[pending]
            finite=newly & np.isfinite(refs); self.failed_reference[finite]=refs[finite]; self.invalidated[newly]=True; self.saw_nonentry[newly]=False; self.invalidated_since[newly]=i; self._event(o,'ACUTE_REFERENCE_INVALIDATION',newly)
        later=self.invalidated & ~held & (i>self.invalidated_since); nonentry=later & ~self.base.trend.entry[i]; newly_nonentry=nonentry & ~self.saw_nonentry; self.saw_nonentry[nonentry]=True; self._event(o,'FRESH_EPOCH_ARMED',newly_nonentry)
        broken=self.base.trend.exit[i] | (p.ret1[i] <= -.08) | ~p.ready[i]
        ordinary=(p.ready[i] & self.base.trend.entry[i] & (p.price[i]>p.ema20[i]) & (p.momentum5[i]>0) & np.isfinite(score) & (score>0) & ~held & ~broken & ~self.base.retired)
        recovered=later & ordinary & np.isfinite(self.failed_reference) & (score>self.failed_reference); self._event(o,'SETTLEMENT_REARM_BLOCKED',recovered & self.base.was_held); self._clear_campaign(o,'REFERENCE_ALPHA_REARM',recovered & ~self.base.was_held)
        edge=(self.invalidated & ~held & (i>self.invalidated_since) & self.saw_nonentry & self.base.trend.entry[i] & ~self.base.was_held); self._clear_campaign(o,'TREND_EDGE_REARM',edge); return ordinary
    def decide(self,o:CloseObservation):
        if not self.parameters.enabled: return self.base.decide(o)
        i=o.session; held=o.units>1e-10; held_count=int(held.sum()); p=self.base.price_signals; score=self.base.features.score[i]; self._update_recovered_provenance(o,held)
        ps=self.failed_fill_attempt_session; pc=self.failed_fill_attempt_held_count; pj=self.failed_fill_attempt_challenger
        confirmed=(ps==i-1 and pj>=0 and held_count<=pc and not held[pj] and o.nav>0 and o.cash<o.nav*.01)
        self.failed_fill_attempt_session=-1; self.failed_fill_attempt_held_count=-1; self.failed_fill_attempt_challenger=-1
        ordinary=self._settled_reference_state(o,held,score); retry=self.invalidated & held & (i>self.invalidated_since)
        if np.any(retry):
            if i<=self.base.last_session: raise ValueError('policy sessions must increase')
            self.base.last_session=i; self.base._observe_inventory(held); units=o.units.copy(); units[retry]=0.; marks=np.nan_to_num(p.price[i],nan=0.); weights=np.divide(units*marks,o.nav,out=np.zeros_like(units),where=o.nav>0); self._event(o,'INVALIDATED_EXIT_RETRY',retry); return CloseDecision(weights,'FRESH_CHALLENGER_AUTHORITY|INVALIDATED_EXIT_RETRY',1.,units)
        vacancy=max(0,self.base.config.max_positions-held_count)
        if confirmed and vacancy>0 and self.base.trend.market[i]:
            choices=np.flatnonzero(ordinary & ~self.invalidated); best=min(choices,key=lambda j:(-score[j],self.market.symbols[j])) if len(choices) else -1
            if best!=pj: self._event(o,'CHALLENGER_IDENTITY_CHANGED',None,prior_challenger=self.market.symbols[pj],current_best=(self.market.symbols[best] if best>=0 else None),prior_attempt_session=int(ps))
            elif self.reference_rearmed_without_fresh_epoch[best]: self._event(o,'RECOVERED_CHALLENGER_AUTHORITY_BLOCK',None,challenger=self.market.symbols[best],prior_attempt_session=int(ps))
            else:
                refs=self.base.owned_alpha_reference.copy(); pending=held & ~self.base.was_held & ~np.isfinite(refs) & np.isfinite(self.base.pending_alpha_reference); refs[pending]=self.base.pending_alpha_reference[pending]; decay=score-refs; decayed=held & np.isfinite(decay) & (decay<0)
                if decayed.any():
                    incumbent=min(np.flatnonzero(decayed),key=lambda j:(decay[j],self.market.symbols[j]))
                    if score[best]>refs[incumbent]:
                        if i<=self.base.last_session: raise ValueError('policy sessions must increase')
                        self.base.last_session=i; self.base._observe_inventory(held); units=o.units.copy(); units[incumbent]=0.; marks=np.nan_to_num(p.price[i],nan=0.); weights=np.divide(units*marks,o.nav,out=np.zeros_like(units),where=o.nav>0); self.base.retired[incumbent]=True; self.base.retired_since[incumbent]=int(i)
                        self.base.trace.append({'kind':'FRESH_CHALLENGER_AUTHORITY_EVENT','date':o.date,'session':int(i),'action':'FRESH_AUTHORITY_DISPLACEMENT','symbol':self.market.symbols[incumbent],'challenger':self.market.symbols[best],'prior_attempt_session':int(ps),'observed_cash':float(o.cash),'observed_nav':float(o.nav),'owned_alpha_reference':float(refs[incumbent]),'incumbent_score':float(score[incumbent]),'challenger_score':float(score[best])})
                        return CloseDecision(weights,'FRESH_CHALLENGER_AUTHORITY|DISPLACEMENT',1.,units)
                self._event(o,'IDENTITY_CONTINUITY_NO_DISPLACEMENT',None,challenger=self.market.symbols[pj],prior_attempt_session=int(ps))
        original=self.base.features
        if np.any(self.invalidated): masked=original.score.copy(); masked[i,self.invalidated]=np.nan; self.base.features=replace(original,score=masked)
        before=len(self.base.trace)
        try: decision=self.base.decide(o)
        finally: self.base.features=original
        fills=[r for r in self.base.trace[before:] if r.get('action')=='VACANCY_FILL']
        if vacancy>0 and o.nav>0 and o.cash<o.nav*.01 and fills:
            requested=fills[-1].get('symbols',[])
            if len(requested)==1:
                j=self.market.symbols.index(requested[0]); self.failed_fill_attempt_session=i; self.failed_fill_attempt_held_count=held_count; self.failed_fill_attempt_challenger=j; self._event(o,'NONDEPLOYABLE_CHALLENGER_ATTEMPT',None,challenger=requested[0],held_count=held_count,observed_cash=float(o.cash),observed_nav=float(o.nav))
        return decision
    def identity(self):
        root=Path(__file__).parent
        return {'name':'offensive_fresh_challenger_authority','parameters':asdict(self.parameters),'implementation_sha256':file_hash(Path(__file__)),'contract_sha256':file_hash(root/'offensive_fresh_challenger_authority_contract.json'),'base_sha256':file_hash(root/'offensive_alpha_decay_displacement.py'),'data_sha256':self.market.fingerprint(),'status':'RESEARCH_NOT_ACCEPTED'}

__all__=['Owner','Parameters','grid','preserve_trace','verify_trace']

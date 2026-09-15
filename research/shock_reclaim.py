"""Provisionally accelerate reentry of a top-ranked shock-exited campaign."""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import numpy as np
from techquant.config import Config
from techquant.data import Market, file_hash
from techquant.policy import CloseObservation
from research.observed_admission_completion import Owner as Parent, Parameters as ParentParameters
from research.quantity_obligation import preserve_trace, verify_trace

@dataclass(frozen=True)
class Parameters:
    enabled: bool
    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('enabled must be boolean')

def grid():
    return [Parameters(False), Parameters(True)]

class Owner(Parent):
    def __init__(self, market: Market, parameters: Parameters, *, config: Config | None = None):
        if type(parameters) is not Parameters:
            raise ValueError('shock reclaim requires its registered parameters')
        super().__init__(market, ParentParameters(), config=config)
        self.reclaim_parameters = parameters
        n=len(market.symbols)
        self.reclaim_reference=np.full(n,np.nan)
        self.reclaim_pending=np.zeros(n,dtype=bool)
        self.reclaim_flat=np.zeros(n,dtype=bool)

    def _observe_reclaim_inventory(self, units: np.ndarray):
        units=np.asarray(units,dtype=float)
        filled=self.reclaim_flat & (units>1e-10)
        if np.any(filled):
            self.reclaim_flat[filled]=False
            self.reclaim_pending[filled]=False
            self.reclaim_reference[filled]=np.nan
            self.inner.readmit[filled]=False
        completed=self.reclaim_pending & (units<=1e-10)
        self.reclaim_pending[completed]=False
        self.reclaim_flat[completed]=True
        return filled,completed

    def _remember_shock_exit(self, session: int, actual: np.ndarray, targets: np.ndarray,
                             reason: str, account_drawdown: bool, other_protection: bool,
                             invalid: np.ndarray | None=None):
        if reason not in {'CROSS_SECTION_SHOCK','ACCUMULATED_MARKET_SHOCK'} or account_drawdown or other_protection:
            return np.zeros(len(self.market.symbols),dtype=bool)
        actual=np.asarray(actual,dtype=float);targets=np.asarray(targets,dtype=float)
        mask=(actual>1e-10)&(targets<=1e-10)
        if invalid is not None: mask &= ~np.asarray(invalid,dtype=bool)
        price=np.nan_to_num(self.inner.features.close[session],nan=0.)
        mask &= price>0
        self.reclaim_reference[mask]=price[mask]
        self.reclaim_pending[mask]=True
        self.reclaim_flat[mask]=False
        return mask

    def _independent_risk(self,o: CloseObservation):
        p,i=self.inner,o.session;f,c=p.features,p.config
        peak=max(p.risk.episode_peak,o.nav)
        dd=1-o.nav/peak if peak>0 else 0.
        loss=o.nav/p.history[-1]-1 if p.history else 0.
        account=dd>=c.risk_drawdown and loss<-.01
        broad=(f.market_dd[i]>=c.risk_drawdown and f.weak[i] and f.breadth[i]<.2)
        warning=(dd>=c.risk_drawdown*2/3 and loss<-max(.02,1.5*f.market_vol[i]))
        return bool(account),bool(broad or warning)

    def decide(self,o: CloseObservation):
        if not self.reclaim_parameters.enabled:
            return super().decide(o)
        p,i=self.inner,o.session
        if i<=p.last_session:
            raise ValueError('policy sessions must increase')
        price=np.nan_to_num(p.features.close[i],nan=0.)
        previous_exit=p.exit_pending.copy(); previous_ceiling=p.reduction_ceiling.copy()
        current_broken=(o.units>1e-10)&((price<=p.stop)|p.features.exit[i]|~p.ready[i])
        cancel=self.reclaim_pending & current_broken
        self.reclaim_pending[cancel]=False; self.reclaim_reference[cancel]=np.nan
        acquired,_=self._observe_reclaim_inventory(o.units)
        naturally_free=self.reclaim_flat & ~p.readmit
        self.reclaim_flat[naturally_free]=False;self.reclaim_reference[naturally_free]=np.nan
        before_readmit=p.readmit.copy()
        allowed=(p.ready[i]&p.features.entry[i]&~p.features.exit[i]
                 &np.isfinite(p.features.score[i])&(p.features.score[i]>0))
        ranked=p._allocation_order(i,np.flatnonzero(allowed))
        top=np.zeros(len(self.market.symbols),dtype=bool)
        if ranked: top[ranked[0]]=True
        release=(self.reclaim_flat & before_readmit & (o.units<=1e-10) & allowed & top
                 &np.isfinite(self.reclaim_reference) & (price>self.reclaim_reference))
        p.readmit[release]=False
        account_drawdown,other_protection=self._independent_risk(o)
        decision=super().decide(o)
        # Permission is provisional unless the ordinary parent itself has now matured.
        natural_now=release & (p.healthy>=p.config.recovery)
        p.readmit[release & ~natural_now]=before_readmit[release & ~natural_now]
        self.reclaim_flat[natural_now]=False;self.reclaim_reference[natural_now]=np.nan
        requested=release & (decision.unit_targets>o.units+1e-10)
        if np.any(release):
            self.trace.append({'kind':'SHOCK_RECLAIM_PERMISSION','date':o.date,'session':i,
                'released':[self.market.symbols[j] for j in np.flatnonzero(release)],
                'requested':[self.market.symbols[j] for j in np.flatnonzero(requested)],
                'reference_close':{self.market.symbols[j]:float(self.reclaim_reference[j]) for j in np.flatnonzero(release) if np.isfinite(self.reclaim_reference[j])},
                'score':{self.market.symbols[j]:float(p.features.score[i,j]) for j in np.flatnonzero(release)}})
            if np.any(requested): decision=replace(decision,reason=decision.reason+'|SHOCK_RECLAIM')
        base_reason=decision.reason.split('|',1)[0]
        invalid=previous_exit | (np.isfinite(previous_ceiling)&(previous_ceiling<o.units-1e-10)) | current_broken
        self._remember_shock_exit(i,o.units,decision.unit_targets,base_reason,account_drawdown,other_protection,invalid)
        return decision

    def identity(self):
        return {'name':'top_ranked_market_shock_reclaim','parameters':asdict(self.reclaim_parameters),
            'implementation_sha256':file_hash(Path(__file__)),
            'contract_sha256':file_hash(Path(__file__).with_name('shock_reclaim_contract.json')),
            'parent_policy':super().identity(),'data_sha256':self.market.fingerprint(),
            'status':'RESEARCH_NOT_ACCEPTED'}

__all__=['Owner','Parameters','grid','preserve_trace','verify_trace']
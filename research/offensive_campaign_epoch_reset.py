"""Campaign-epoch invalidation after an authoritative acute security break."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import numpy as np

from techquant.data import Market, file_hash
from techquant.policy import CloseDecision
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
            raise ValueError('campaign epoch reset requires registered parameters')
        self.market=market; self.parameters=parameters
        self.base=BaseOwner(market,BaseParameters(parameters.enabled))
        n=len(market.symbols)
        self.invalidated=np.zeros(n,dtype=bool)
        self.saw_nonentry=np.zeros(n,dtype=bool)
        self.invalidated_since=np.full(n,-1,dtype=int)

    @property
    def trace(self): return self.base.trace

    def _event(self,o,action,mask):
        if np.any(mask):
            self.base.trace.append({'kind':'CAMPAIGN_EPOCH_RESET_EVENT','date':o.date,
                'session':int(o.session),'action':action,
                'symbols':[self.market.symbols[j] for j in np.flatnonzero(mask)]})

    def decide(self,o):
        if not self.parameters.enabled:
            return self.base.decide(o)
        i=o.session; held=o.units>1e-10
        acute=self.base.price_signals.ret1[i] <= -.08
        existing=held | self.base.retired
        newly=acute & existing
        if np.any(newly):
            self.invalidated[newly]=True; self.saw_nonentry[newly]=False; self.invalidated_since[newly]=i
            self._event(o,'ACUTE_CAMPAIGN_INVALIDATION',newly)

        flat=~held
        later=self.invalidated & flat & (i>self.invalidated_since)
        became_nonentry=later & ~self.base.trend.entry[i]
        newly_nonentry=became_nonentry & ~self.saw_nonentry
        self.saw_nonentry[became_nonentry]=True
        self._event(o,'FRESH_EPOCH_ARMED',newly_nonentry)
        rearm=later & self.saw_nonentry & self.base.trend.entry[i]
        if np.any(rearm):
            self.invalidated[rearm]=False; self.saw_nonentry[rearm]=False; self.invalidated_since[rearm]=-1
            self._event(o,'FRESH_EPOCH_REARM',rearm)

        retry=self.invalidated & held & (i>self.invalidated_since)
        if np.any(retry):
            if i <= self.base.last_session:
                raise ValueError('policy sessions must increase')
            self.base.last_session=i
            self.base._observe_inventory(held)
            units=o.units.copy(); units[retry]=0.
            marks=np.nan_to_num(self.base.price_signals.price[i],nan=0.)
            weights=np.divide(units*marks,o.nav,out=np.zeros_like(units),where=o.nav>0)
            self._event(o,'INVALIDATED_EXIT_RETRY',retry)
            return CloseDecision(weights,'CAMPAIGN_EPOCH_RESET|INVALIDATED_EXIT_RETRY',1.,units)

        original=self.base.features
        if np.any(self.invalidated):
            score=original.score.copy(); score[i,self.invalidated]=np.nan
            self.base.features=replace(original,score=score)
        try:
            return self.base.decide(o)
        finally:
            self.base.features=original

    def identity(self):
        root=Path(__file__).parent
        return {'name':'offensive_campaign_epoch_reset','parameters':asdict(self.parameters),
            'implementation_sha256':file_hash(Path(__file__)),
            'contract_sha256':file_hash(root/'offensive_campaign_epoch_reset_contract.json'),
            'base_sha256':file_hash(root/'offensive_alpha_decay_displacement.py'),
            'data_sha256':self.market.fingerprint(),'status':'RESEARCH_NOT_ACCEPTED'}


__all__=['Owner','Parameters','grid','preserve_trace','verify_trace']

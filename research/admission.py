"""Independent reinvestment quarantine, not a veto on intact winning inventory."""
from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import itertools
import numpy as np
from techquant.data import Market,file_hash
from techquant.policy import CloseDecision,CloseObservation
from research.leadership import Owner as Leader,Parameters as LeaderParameters


@dataclass(frozen=True)
class Parameters:
    leader_rule:str='responsive'
    market_span:int=20
    minimum_breadth:float=.6

    def __post_init__(self):
        if self.leader_rule not in ('responsive','persistent'):
            raise ValueError('unknown leadership rule')
        if isinstance(self.market_span,bool) or not isinstance(self.market_span,int) or self.market_span<1:
            raise ValueError('market span must be positive')
        if not np.isfinite(self.minimum_breadth) or not 0<self.minimum_breadth<=1:
            raise ValueError('invalid breadth')

    def leader(self):
        return (LeaderParameters(60,20,2,False) if self.leader_rule=='responsive'
                else LeaderParameters(252,60,2,False))


def grid():
    return [Parameters(*v) for v in itertools.product(('responsive','persistent'),(10,20,40),(.4,.6))]


class Owner:
    def __init__(self,market:Market,p:Parameters):
        self.market,self.params=market,p
        self.inner=Leader(market,p.leader())
        quoted=market.panel('close');active=quoted.notna() & market.panel('volume').gt(0)
        close=quoted.ffill()
        changes=close.pct_change(fill_method=None).where(active & active.shift(fill_value=False))
        index=(1+changes.mean(axis=1).fillna(0.)).cumprod()
        ready=active & active.cumsum().ge(20)
        stock_trend=close.ewm(span=20,adjust=False).mean()
        breadth=((close>stock_trend)&ready).sum(axis=1).div(ready.sum(axis=1).replace(0,np.nan)).fillna(0.)
        good=index.gt(index.ewm(span=p.market_span,adjust=False).mean()) & breadth.ge(p.minimum_breadth)
        self.admit=good.rolling(3,min_periods=3).sum().eq(3).to_numpy()

    def decide(self,o:CloseObservation)->CloseDecision:
        request=self.inner.decide(o)
        if self.admit[o.session]:return request
        # Crucially, the inner state was informed of real, unscaled inventory.
        return CloseDecision(np.minimum(request.weights,o.weights),
                             request.reason+'|WAIT_FOR_MARKET_ADMISSION',request.cap)

    def identity(self):
        return {'name':'confirmed_market_admission','parameters':asdict(self.params),
                'implementation_sha256':file_hash(Path(__file__)),
                'ownership_sha256':file_hash(Path(__file__).with_name('leadership.py')),
                'data_sha256':self.market.fingerprint(),'status':'RESEARCH_NOT_ACCEPTED'}

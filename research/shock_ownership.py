"""Independent shock-latched ownership and continuously observed market recovery.

Ordinary winner volatility does not cause automatic trimming. Actual account
shocks or broad price shocks request liquidation for the next session. Recovery
uses market prices which continue to evolve while the account holds cash.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import itertools
import numpy as np
from techquant.data import Market,file_hash
from techquant.policy import CloseDecision,CloseObservation
from research.leadership import signals,Parameters as SignalParameters


@dataclass(frozen=True)
class Parameters:
    leadership_window:int=60
    positions:int=2
    shock_loss:float=.04
    recovery_window:int=10

    def __post_init__(self):
        for value in (self.leadership_window,self.positions,self.recovery_window):
            if isinstance(value,bool) or not isinstance(value,int) or value<1:
                raise ValueError('lookbacks and positions must be positive integers')
        if not np.isfinite(self.shock_loss) or not 0<self.shock_loss<1:
            raise ValueError('shock loss must be finite and between zero and one')


def grid():
    return [Parameters(*v) for v in itertools.product((60,120),(2,4),(.04,.06),(10,20))]


class Owner:
    def __init__(self,market:Market,p:Parameters):
        self.market,self.params=market,p
        self.s=signals(market,SignalParameters(p.leadership_window,60,p.positions,False))
        quote=market.panel('close');active=quote.notna()&market.panel('volume').gt(0)
        price=quote.ffill()
        returns=price.pct_change(fill_method=None).where(active&active.shift(fill_value=False))
        index=(1.+returns.mean(axis=1).fillna(0.)).cumprod()
        self.market_shock=(returns.median(axis=1).le(-.75*p.shock_loss)&
                           returns.lt(0).sum(axis=1).div(returns.notna().sum(axis=1).replace(0,np.nan)).ge(.6)).to_numpy()
        ema10=index.ewm(span=10,adjust=False).mean()
        ema20=index.ewm(span=20,adjust=False).mean()
        stock20=price.ewm(span=20,adjust=False).mean()
        ready=active&active.cumsum().ge(20)
        breadth=(ready&price.gt(stock20)).sum(axis=1).div(ready.sum(axis=1).replace(0,np.nan)).fillna(0.)
        self.healthy_market=(index.ge(index.shift().rolling(p.recovery_window,min_periods=p.recovery_window).max())&
                             ema10.gt(ema20)&breadth.ge(.6)).to_numpy()
        self.negative20=price.pct_change(20,fill_method=None).lt(0).to_numpy()
        self.exit_pending=np.zeros(len(market.symbols),dtype=bool)
        self.below=np.zeros(len(market.symbols),dtype=int)
        self.closed=False
        self.healthy=0
        self.nav=[]
        self.last_session=-1

    def decide(self,o:CloseObservation)->CloseDecision:
        i=o.session
        if i<=self.last_session:raise ValueError('policy sessions must increase')
        self.last_session=i
        p,s=self.params,self.s
        held=o.units>1e-10
        self.exit_pending[~held]=False
        day_loss=o.nav/self.nav[-1]-1 if self.nav else 0.
        loss3=o.nav/self.nav[-3]-1 if len(self.nav)>=3 else 0.
        self.nav.append(o.nav)
        shock=(day_loss<=-p.shock_loss or loss3<=-1.5*p.shock_loss or self.market_shock[i])
        why=[]
        if shock:
            self.closed=True;self.healthy=0;self.exit_pending|=held
            why.append('OBSERVED_ACCOUNT_OR_MARKET_SHOCK')
        elif self.closed:
            self.healthy=self.healthy+1 if self.healthy_market[i] else 0
            if self.healthy>=3 and not self.exit_pending.any():
                self.closed=False;self.healthy=0
                why.append('PRICE_CONFIRMED_MARKET_RECOVERY')
        self.below=np.where(s.price[i]<.97*s.ema[i],self.below+1,0)
        broken=held&((self.below>=2)&self.negative20[i]|~s.ready[i])
        self.exit_pending|=broken
        if self.closed:self.exit_pending|=held
        want=o.weights.copy();want[self.exit_pending]=0.
        if self.exit_pending.any():why.append('PROTECTIVE_EXIT_RETRY')
        want=np.minimum(want,1. if len(held)==1 else .8)
        if not self.closed and not self.exit_pending.any():
            score=s.score[i]
            ranked=sorted(np.flatnonzero(s.ready[i]&(score>0)),key=lambda j:(-score[j],self.market.symbols[j]))
            eligible=s.ready[i]&(score>0)&(s.price[i]>s.ema10[i])&(s.price[i]>s.ema[i])&(s.momentum5[i]>0)
            entrants=[j for j in ranked if eligible[j] and not held[j]]
            existing=list(np.flatnonzero(held));capacity=min(p.positions,len(held))
            if i%20==0 and len(existing)>=capacity and entrants:
                ranks={j:k+1 for k,j in enumerate(ranked)}
                worst=min(existing,key=lambda j:(score[j],self.market.symbols[j]))
                if ranks.get(worst,len(held)+1)>2*capacity and score[entrants[0]]>1.25*max(score[worst],.001):
                    want[worst]=0.;why.append('MATERIAL_LEADER_REPLACEMENT')
            entrants=entrants[:max(0,capacity-len(existing))]
            cash=min(o.cash/o.nav,max(0.,1.-o.weights.sum()))
            if entrants and cash>=.01:
                want[entrants]=min(cash/len(entrants),1. if len(held)==1 else .6)
                why.append('CASH_FUNDED_TREND_ENTRY')
        return CloseDecision(want,'|'.join(why) if why else
                             ('WAIT_FOR_PRICE_RECOVERY' if self.closed else 'RETAIN_INTACT_TRENDS'))

    def identity(self):
        return {'name':'shock_latched_ownership','parameters':asdict(self.params),
                'implementation_sha256':file_hash(Path(__file__)),
                'signals_sha256':file_hash(Path(__file__).with_name('leadership.py')),
                'data_sha256':self.market.fingerprint(),'status':'RESEARCH_NOT_ACCEPTED'}

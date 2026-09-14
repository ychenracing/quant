"""Independent persistent ownership / local-downside research policy.

This is a research adapter over the unchanged execution engine, not a native
reference strategy. Hooks are scoped to one process and always restored. Every
result records the adapter bytes and numerical parameters separately from the
production engine. Do not run simultaneous adapters in threads.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from techquant.config import Config
from techquant.data import file_hash
import techquant.engine as engine

@dataclass(frozen=True)
class Parameters:
    lookback: int = 120
    trend: int = 80
    positions: int = 2
    protection: str = 'downside'

    def __post_init__(self):
        if self.lookback < 10 or self.trend < 10 or self.positions < 1:
            raise ValueError('invalid ownership horizon or position count')
        if self.protection not in ('structural','downside'):
            raise ValueError('unknown protection policy')


def allocation(current, intact, eligible, score, positions, scheduled):
    """Hold units, remove broken holdings, then water-fill only available cash."""
    current=np.asarray(current,dtype=float)
    intact=np.asarray(intact,dtype=bool);eligible=np.asarray(eligible,dtype=bool)
    score=np.asarray(score,dtype=float)
    if (not np.isfinite(current).all() or (current<0).any() or current.sum()>1+1e-8
        or current.ndim!=1 or any(a.shape!=current.shape for a in (intact,eligible,score))):
        raise ValueError('invalid portfolio inputs')
    want=current.copy();want[~intact]=0.
    n=len(current);k=min(positions,n)
    order=sorted(np.flatnonzero(intact & eligible & (score>0)),key=lambda j:(-score[j],j))
    held=list(np.flatnonzero(want>1e-10))
    if scheduled and len(held)>=k and order:
        rank={j:p for p,j in enumerate(order)}
        outside=[j for j in order if j not in held]
        for j in sorted(held,key=lambda j:score[j]):
            if not outside:break
            challenger=outside[0]
            if rank.get(j,len(order))>=2*k and score[challenger]>1.25*max(score[j],.001):
                want[j]=0.;held.remove(j);outside.pop(0)
    entry_cap=max(.6,1/n);drift_cap=max(.8,1/n)
    # A hard concentration intervention has a wide band; entry sizing is not a
    # continuous equal-weight rebalance of successful positions.
    want[want>drift_cap+1e-10]=entry_cap
    held=list(np.flatnonzero(want>1e-10))
    chosen=held+[j for j in order if j not in held][:max(0,k-len(held))]
    budget=1-want.sum()
    if budget<.05 or not chosen:return want
    recipients=[j for j in chosen if eligible[j] and want[j]<entry_cap-1e-10]
    for _ in range(k+1):
        if not recipients or budget<=1e-10:break
        amount=budget/len(recipients)
        for j in recipients:
            add=min(amount,entry_cap-want[j]);want[j]+=add;budget-=add
        recipients=[j for j in recipients if want[j]<entry_cap-1e-10]
    return np.maximum(want,0.)


class Policy:
    def __init__(self,params):self.params=params

    def build(self,market,_config):
        p=self.params
        q=market.panel('close');active=q.notna() & market.panel('volume').gt(0)
        close=q.ffill();ret=close.pct_change(fill_method=None)
        fast=close.rolling(10,min_periods=10).mean()
        slow=close.rolling(p.trend,min_periods=10).mean()
        count=active.cumsum()
        ready=active & count.ge(10) & active.rolling(10,min_periods=10).mean().ge(.8)
        first=q.where(active & count.eq(1)).ffill()
        since=close/first
        horizons=[max(10,p.lookback//2),p.lookback,p.lookback*2]
        momentum=sum(np.log((close/close.shift(h)).fillna(since)) for h in horizons)/3
        short=close/close.shift(10)-1
        entry=ready & close.gt(fast) & close.gt(slow) & short.gt(0) & momentum.gt(0)
        structural=(close.lt(slow) & short.lt(0)) | close.le(close.shift().rolling(20,min_periods=10).min())
        prior_vol=ret.rolling(20,min_periods=10).std(ddof=0).shift().clip(lower=.008)
        fall5=close/close.shift(5)-1
        shock=((ret.lt(-2.5*prior_vol)) | fall5.lt(-np.maximum(.06,2.5*prior_vol.shift(4)*np.sqrt(5)))) & close.lt(fast)
        guarded=np.zeros(len(market.symbols),bool);healthy=np.zeros(len(market.symbols),int)
        intact=np.ones(ready.shape,bool);entries=entry.to_numpy().copy()
        for i in range(len(close)):
            bad=structural.iloc[i].to_numpy() | (~ready.iloc[i].to_numpy())
            if p.protection=='downside':bad|=shock.iloc[i].fillna(False).to_numpy()
            guarded|=bad
            good=entries[i] & ~bad
            healthy=np.where(good,healthy+1,0)
            guarded=np.where(healthy>=2,False,guarded)
            intact[i]=~guarded & ~bad
            entries[i]&=intact[i]
        breadth=(close.gt(fast)&ready).sum(axis=1)/ready.sum(axis=1).replace(0,np.nan)
        return SimpleNamespace(symbols=market.symbols,sectors=tuple(market.sectors.get(s,'unknown') for s in market.symbols),
            ready=ready.to_numpy(),breadth=breadth.fillna(0).to_numpy(),entry=entries,
            intact=intact,score=momentum.where(ready,-np.inf).fillna(-np.inf).to_numpy())

    def update(self,i,f,history,config):
        return (1.,'LOCAL_DOWNSIDE_OWNERSHIP') if f.ready[i].any() else (0.,'WARMUP_OR_NO_FRESH_QUOTES')

    def weights(self,i,f,current,config,*,cap,rebalance,**kwargs):
        want=allocation(current,f.intact[i],f.entry[i],f.score[i],self.params.positions,rebalance)
        why=[]
        if np.any((current>0)&(~f.intact[i])):why.append('LOCAL_TREND_OR_DOWNSIDE_EXIT')
        if np.any(want>current+1e-10):why.append('REINVEST_OBSERVED_LEADERS')
        if not why:why.append('RETAIN_ECONOMIC_UNITS')
        return want,why


def run(market,params=None,**kwargs):
    params=params or Parameters();policy=Policy(params)
    cfg=replace(Config(),rebalance=20)
    with patch.multiple(engine,build_features=policy.build,RiskState=lambda:policy,target_weights=policy.weights):
        result=engine.run(market,cfg,**kwargs)
    result.metadata['algorithm']={'name':'persistent_ownership_local_downside','parameters':asdict(params),
        'adapter_sha256':file_hash(Path(__file__)),'status':'RESEARCH_NOT_ACCEPTED'}
    return result

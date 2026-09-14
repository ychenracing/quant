"""Independent causal directional-change research; unchanged execution engine.

Observed high-water exits and low-water reversals do not predict a turning point.
The adapter is research-only, process-scoped and hash-bound to each replay.
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
    exit_drawdown: float = .12
    reentry_rebound: float = .04
    positions: int = 2
    lookback: int = 120

    def __post_init__(self):
        for name in ('exit_drawdown','reentry_rebound'):
            v=getattr(self,name)
            if isinstance(v,bool) or not np.isfinite(v) or not 0<v<1:
                raise ValueError('directional thresholds must be finite in (0,1)')
        for name in ('positions','lookback'):
            v=getattr(self,name)
            if isinstance(v,bool) or not isinstance(v,int) or v<1:
                raise ValueError('positions and lookback must be positive integers')
        if self.lookback<10:raise ValueError('lookback must be at least 10')


def directional_state(close, ready, initial, confirmation, exit_drawdown, rebound):
    """A single forward pass. An exit cannot reenter on that same close."""
    close=np.asarray(close,dtype=float)
    if close.ndim!=2 or any(np.shape(x)!=close.shape for x in (ready,initial,confirmation)):
        raise ValueError('state inputs must have the same two-dimensional shape')
    opened=np.zeros(close.shape[1],bool);seen=np.zeros(close.shape[1],bool)
    peak=np.zeros(close.shape[1]);trough=np.full(close.shape[1],np.inf)
    output=np.zeros(close.shape,bool)
    for i,price in enumerate(close):
        valid=np.asarray(ready[i],bool)&np.isfinite(price)&(price>0)
        was_open=opened.copy()
        peak=np.where(valid & was_open,np.maximum(peak,price),peak)
        exiting=was_open & (~valid | (price<=peak*(1-exit_drawdown)))
        opened[exiting]=False
        trough=np.where(exiting & valid,price,trough)
        closed=valid & ~was_open
        trough=np.where(closed,np.minimum(trough,price),trough)
        entering=closed & ((~seen & initial[i]) | (seen & confirmation[i] & (price>=trough*(1+rebound))))
        opened[entering]=True;seen[entering]=True
        peak=np.where(entering,price,peak)
        output[i]=opened & valid
    return output


def allocate(current, intact, eligible, score, positions, scheduled):
    """Retain units and water-fill material cash; no continuous winner trimming."""
    current=np.asarray(current,dtype=float)
    intact=np.asarray(intact,dtype=bool);eligible=np.asarray(eligible,dtype=bool)
    score=np.asarray(score,dtype=float)
    if (not np.isfinite(current).all() or (current<0).any() or current.sum()>1+1e-8
        or current.ndim!=1 or any(a.shape!=current.shape for a in (intact,eligible,score))):
        raise ValueError('invalid portfolio inputs')
    want=current.copy();want[~intact]=0.
    n=len(current);k=min(positions,n)
    if k<1:raise ValueError('empty position capacity')
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
    def __init__(self,parameters):self.parameters=parameters
    def build(self,market,_config):
        p=self.parameters
        q=market.panel('close');active=q.notna() & market.panel('volume').gt(0)
        close=q.ffill();count=active.cumsum()
        ready=active & count.ge(10) & active.rolling(10,min_periods=10).mean().ge(.8)
        first=q.where(active & count.eq(1)).ffill();since=close/first
        horizons=[max(10,p.lookback//2),p.lookback,p.lookback*2]
        momentum=sum(np.log((close/close.shift(h)).fillna(since)) for h in horizons)/3
        short=close/close.shift(10)-1;ret5=close/close.shift(5)-1
        intact=directional_state(close.to_numpy(),ready.to_numpy(),short.gt(0).to_numpy(),
            ret5.gt(0).to_numpy(),p.exit_drawdown,p.reentry_rebound)
        score=momentum.where(ready,-np.inf).fillna(-np.inf).to_numpy()
        entry=ready.to_numpy() & intact & (score>0)
        breadth=(close.gt(close.rolling(10,min_periods=10).mean())&ready).sum(axis=1)/ready.sum(axis=1).replace(0,np.nan)
        return SimpleNamespace(symbols=market.symbols,ready=ready.to_numpy(),
            breadth=breadth.fillna(0).to_numpy(),entry=entry,intact=intact,score=score)
    def update(self,i,f,history,config):
        return (1.,'DIRECTIONAL_OWNERSHIP') if f.ready[i].any() else (0.,'WARMUP_OR_NO_FRESH_QUOTES')
    def weights(self,i,f,current,config,*,cap,rebalance,**kwargs):
        want=allocate(current,f.intact[i],f.entry[i],f.score[i],self.parameters.positions,rebalance)
        why=[]
        if np.any((current>0)&~f.intact[i]):why.append('OBSERVED_PEAK_REVERSAL_EXIT')
        if np.any(want>current+1e-10):why.append('OBSERVED_TROUGH_RECOVERY_OR_LEADERSHIP')
        if not why:why.append('RETAIN_ECONOMIC_UNITS')
        return want,why


def run(market,parameters=None,**kwargs):
    p=parameters or Parameters();policy=Policy(p);cfg=replace(Config(),rebalance=20)
    with patch.multiple(engine,build_features=policy.build,RiskState=lambda:policy,target_weights=policy.weights):
        result=engine.run(market,cfg,**kwargs)
    result.metadata['algorithm']={'name':'causal_directional_change','parameters':asdict(p),
        'adapter_sha256':file_hash(Path(__file__)),'status':'RESEARCH_NOT_ACCEPTED'}
    return result

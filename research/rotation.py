"""Independent momentum leadership with persistent holdings and cash-preserving exits."""
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
    lookback:int=80
    trend:int=40
    positions:int=3
    review:int=20
    def __post_init__(self):
        for name in ('lookback','trend','positions','review'):
            v=getattr(self,name)
            if isinstance(v,bool) or not isinstance(v,int) or v<1:raise ValueError('invalid integer parameter')

class Policy:
    def __init__(self,parameters):self.parameters=parameters
    def build(self,market,_config):
        p=self.parameters;q=market.panel('close');active=q.notna()&market.panel('volume').gt(0)
        c=q.ffill();count=active.cumsum();ready=active&count.ge(10)&active.rolling(10,min_periods=10).mean().ge(.8)
        first=q.where(active&count.eq(1)).ffill();score=np.log((c/c.shift(p.lookback)).fillna(c/first))
        slow=c.rolling(p.trend,min_periods=10).mean();weak=c.lt(slow);broken=weak&weak.shift(fill_value=False)
        entry=ready&c.gt(slow)&c.gt(c.shift(5))&score.gt(0)
        return SimpleNamespace(symbols=market.symbols,ready=ready.to_numpy(),intact=(~broken&ready).to_numpy(),
            entry=entry.to_numpy(),score=score.where(ready,-np.inf).fillna(-np.inf).to_numpy(),
            breadth=(c.gt(slow)&ready).sum(axis=1).div(ready.sum(axis=1).replace(0,np.nan)).fillna(0).to_numpy())
    def update(self,i,f,history,config):return (1.,'PERSISTENT_MOMENTUM') if f.ready[i].any() else (0.,'WARMUP')
    def weights(self,i,f,current,config,*,rebalance,**kwargs):
        p=self.parameters;want=current.copy();forced=(current>1e-10)&~f.intact[i];want[forced]=0.
        if forced.any():return want,['SLOW_TREND_EXIT_HOLD_CASH']
        held=set(np.flatnonzero(current>1e-10));capacity=min(p.positions,len(current))
        eligible=f.entry[i]|((current>1e-10)&f.intact[i]&(f.score[i]>0))
        order=sorted(np.flatnonzero(eligible),key=lambda j:(-f.score[i,j],j))
        chosen=order[:capacity]
        # Vacant risk capacity is considered daily; a full intact book only rotates on review.
        if not rebalance and len(held)>=capacity:chosen=list(held)
        if set(chosen)==held:
            if 1-current.sum()<.05:return current.copy(),['RETAIN_UNITS']
            recipients=[j for j in chosen if f.entry[i,j]]
            want=current.copy()
        else:
            want=np.zeros_like(current);recipients=chosen
        cap=max(.6,1/len(current));budget=1-want.sum()
        for _ in range(capacity+1):
            recipients=[j for j in recipients if want[j]<cap-1e-10]
            if not recipients or budget<1e-10:break
            amount=budget/len(recipients)
            for j in recipients:
                add=min(amount,cap-want[j]);want[j]+=add;budget-=add
        if np.max(want)>max(.8,1/len(current)):
            j=int(np.argmax(want));want[j]=cap
        return want,['LEADERSHIP_ROTATION_OR_OBSERVED_ADMISSION']

def run(market,parameters=None,**kwargs):
    p=parameters or Parameters();policy=Policy(p)
    with patch.multiple(engine,build_features=policy.build,RiskState=lambda:policy,target_weights=policy.weights):
        result=engine.run(market,replace(Config(),rebalance=p.review),**kwargs)
    result.metadata['algorithm']={'name':'persistent_momentum_rotation','parameters':asdict(p),
        'adapter_sha256':file_hash(Path(__file__)),'status':'RESEARCH_NOT_ACCEPTED'}
    return result

"""Fresh-entry, cash-preserving trend research; not an accepted production policy."""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from techquant.config import Config
from techquant.data import file_hash
import techquant.engine as engine
from research.directional import allocate

@dataclass(frozen=True)
class Parameters:
    slow:int=40
    stop:float=.12
    positions:int=2
    entry_window:int=20
    def __post_init__(self):
        for name in ('slow','positions','entry_window'):
            v=getattr(self,name)
            if isinstance(v,bool) or not isinstance(v,int) or v<1:raise ValueError('invalid integral parameter')
        if self.slow<10 or self.entry_window<10 or not np.isfinite(self.stop) or not 0<self.stop<1:
            raise ValueError('invalid trend horizon or stop')

def fresh_weights(current,intact,entry,score,positions):
    current=np.asarray(current,float);intact=np.asarray(intact,bool)
    if np.any((current>1e-10)&~intact):
        want=current.copy();want[~intact]=0.
        return want
    return allocate(current,intact,entry,score,positions,False)


def signal_state(close,ready,entry,broken,stop):
    opened=np.zeros(close.shape[1],bool);peak=np.zeros(close.shape[1]);state=np.zeros(close.shape,bool)
    for i,c in enumerate(close):
        valid=ready[i]&np.isfinite(c)&(c>0)
        peak=np.where(opened&valid,np.maximum(peak,c),peak)
        exit_mask=opened&(~valid|broken[i]|(c<=peak*(1-stop)))
        opened[exit_mask]=False
        enter_mask=valid&entry[i]&~opened&~exit_mask
        peak=np.where(enter_mask,c,peak);opened[enter_mask]=True
        state[i]=opened&valid
    return state


class Policy:
    def __init__(self,parameters):self.parameters=parameters
    def build(self,market,_config):
        p=self.parameters;q=market.panel('close');active=q.notna()&market.panel('volume').gt(0)
        c=q.ffill();count=active.cumsum();ready=active&count.ge(10)&active.rolling(10,min_periods=10).mean().ge(.8)
        first=q.where(active&count.eq(1)).ffill();since=c/first
        score=sum(np.log((c/c.shift(h)).fillna(since)) for h in (30,60,120))/3
        slow=c.rolling(p.slow,min_periods=10).mean();weak=c.lt(slow)
        broken=(weak&weak.shift(fill_value=False)).to_numpy()
        fresh=c.ge(c.shift().rolling(p.entry_window,min_periods=9).max())&c.gt(slow)&score.gt(0)&ready
        intact=signal_state(c.to_numpy(),ready.to_numpy(),fresh.to_numpy(),broken,p.stop)
        return SimpleNamespace(symbols=market.symbols,ready=ready.to_numpy(),intact=intact,
            entry=fresh.to_numpy()&intact,score=score.where(ready,-np.inf).fillna(-np.inf).to_numpy(),
            breadth=(c.gt(slow)&ready).sum(axis=1).div(ready.sum(axis=1).replace(0,np.nan)).fillna(0).to_numpy())
    def update(self,i,f,history,config):return (1.,'CASH_PRESERVING_TREND') if f.ready[i].any() else (0.,'WARMUP')
    def weights(self,i,f,current,config,**kwargs):
        want=fresh_weights(current,f.intact[i],f.entry[i],f.score[i],self.parameters.positions)
        why='PROTECTIVE_EXIT_HOLD_CASH' if np.any((current>1e-10)&~f.intact[i]) else 'FRESH_ENTRY_OR_RETAIN_UNITS'
        return want,[why]


def run(market,parameters=None,**kwargs):
    p=parameters or Parameters();policy=Policy(p)
    with patch.multiple(engine,build_features=policy.build,RiskState=lambda:policy,target_weights=policy.weights):
        result=engine.run(market,replace(Config(),rebalance=20),**kwargs)
    result.metadata['algorithm']={'name':'fresh_breakout_cash_preservation','parameters':asdict(p),
        'adapter_sha256':file_hash(Path(__file__)),'allocation_sha256':file_hash(Path('research/directional.py')),
        'status':'RESEARCH_NOT_ACCEPTED'}
    return result

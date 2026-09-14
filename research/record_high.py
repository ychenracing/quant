"""Record-high admission to distinguish a renewed trend from a bear-market bounce."""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from techquant.config import Config
from techquant.data import file_hash
import techquant.engine as engine
from research.breakout import signal_state, fresh_weights

@dataclass(frozen=True)
class Parameters:
    stop:float=.12
    positions:int=3
    minimum_observations:int=40
    def __post_init__(self):
        if not 0<self.stop<1:raise ValueError('invalid stop')
        for name in ('positions','minimum_observations'):
            v=getattr(self,name)
            if isinstance(v,bool) or not isinstance(v,int) or v<1:raise ValueError('invalid integer')

class Policy:
    def __init__(self,parameters):self.parameters=parameters
    def build(self,market,_config):
        p=self.parameters;q=market.panel('close');active=q.notna()&market.panel('volume').gt(0)
        c=q.ffill();count=active.cumsum();ready=active&count.ge(p.minimum_observations)&active.rolling(10,min_periods=10).mean().ge(.8)
        first=q.where(active&count.eq(1)).ffill();since=c/first
        score=sum(np.log((c/c.shift(h)).fillna(since)) for h in (30,60,120))/3
        fresh=c.ge(c.cummax().shift())&score.gt(0)&ready
        intact=signal_state(c.to_numpy(),ready.to_numpy(),fresh.to_numpy(),np.zeros(c.shape,bool),p.stop)
        return SimpleNamespace(symbols=market.symbols,ready=ready.to_numpy(),intact=intact,
            entry=fresh.to_numpy()&intact,score=score.where(ready,-np.inf).fillna(-np.inf).to_numpy(),
            breadth=np.mean(intact,axis=1))
    def update(self,i,f,history,config):return (1.,'RECORD_HIGH_OWNERSHIP') if f.ready[i].any() else (0.,'WARMUP')
    def weights(self,i,f,current,config,**kwargs):
        want=fresh_weights(current,f.intact[i],f.entry[i],f.score[i],self.parameters.positions)
        why='PEAK_DAMAGE_HOLD_CASH' if np.any((current>1e-10)&~f.intact[i]) else 'FRESH_RECORD_OR_RETAIN_UNITS'
        return want,[why]

def run(market,parameters=None,**kwargs):
    p=parameters or Parameters();policy=Policy(p)
    with patch.multiple(engine,build_features=policy.build,RiskState=lambda:policy,target_weights=policy.weights):
        result=engine.run(market,replace(Config(),rebalance=20),**kwargs)
    result.metadata['algorithm']={'name':'record_high_ownership','parameters':asdict(p),
        'adapter_sha256':file_hash(Path(__file__)),'dependencies':{
            path:file_hash(Path(path)) for path in ('research/breakout.py','research/directional.py')},
        'status':'RESEARCH_NOT_ACCEPTED'}
    return result

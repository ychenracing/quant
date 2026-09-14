"""Broad confirmation governs new cash, not mechanical sales of owned winners."""
from dataclasses import dataclass,asdict
from pathlib import Path
from unittest.mock import patch
import numpy as np
from techquant.config import Config
from techquant.data import file_hash
from research import breakout
import techquant.engine as engine

@dataclass(frozen=True)
class Parameters:
    market_window:int=120
    recovery_breadth:float=.75
    stop_atr:float=3.5
    def __post_init__(self):
        if self.market_window<10 or not 0<self.recovery_breadth<=1 or self.stop_atr<=0:
            raise ValueError('invalid broad-entry parameters')

def median_index(quotes,active):
    fresh_pair=active & active.shift(1,fill_value=False)
    returns=quotes.pct_change(fill_method=None).where(fresh_pair)
    return (1+returns.median(axis=1).fillna(0.)).cumprod()

class Policy(breakout.Policy):
    def __init__(self,params):
        super().__init__(breakout.Parameters(40,params.stop_atr,.7,'trailing'));self.regime=params
    def build(self,market,config):
        f=super().build(market,config);p=self.regime
        q=market.panel('close');active=q.notna()&market.panel('volume').gt(0);close=q.ffill()
        index=median_index(q,active)
        broad=index>index.rolling(p.market_window,min_periods=10).mean()
        ready=active & active.cumsum().ge(11)
        breadth=((close>close.rolling(20,min_periods=10).mean())&ready).sum(axis=1)/ready.sum(axis=1).replace(0,np.nan)
        recovery=(index/index.shift(20)>1)&(breadth>=p.recovery_breadth)
        f.entry &= (broad|recovery).to_numpy()[:,None]
        return f

def run(market,params=None,**kwargs):
    params=params or Parameters();policy=Policy(params)
    with patch.multiple(engine,build_features=policy.build,RiskState=lambda:policy,target_weights=policy.weights):
        result=engine.run(market,Config(),**kwargs)
    result.metadata['algorithm']={'name':'broad_confirmation_entry','parameters':asdict(params),
        'adapter_sha256':file_hash(Path(__file__)),'ownership_sha256':file_hash(Path(breakout.__file__)),
        'status':'RESEARCH_NOT_ACCEPTED'}
    return result

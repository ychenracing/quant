"""Independent cash-sleeve signal research, without cross-name capital recycling.

Single-security screening uses each sleeve's own order materiality. Aggregate
screening is therefore diagnostic, not whole-account execution acceptance.
"""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from techquant.config import Config
from techquant.data import file_hash
import techquant.engine as engine

@dataclass(frozen=True)
class Parameters:
    stop: float = .10
    reentry_window: int = 40
    def __post_init__(self):
        if isinstance(self.stop,bool) or not np.isfinite(self.stop) or not 0<self.stop<1:
            raise ValueError('stop must be a finite fraction in (0,1)')
        if isinstance(self.reentry_window,bool) or not isinstance(self.reentry_window,int) or self.reentry_window<1:
            raise ValueError('reentry_window must be a positive integer')


def state_path(close, active, stop, reentry_window):
    c=np.asarray(close,float);active=np.asarray(active,bool)
    if c.ndim!=2 or c.shape!=active.shape:raise ValueError('matching two-dimensional inputs required')
    prior=pd.DataFrame(c).rolling(reentry_window,min_periods=1).max().shift().to_numpy()
    seen=np.zeros(c.shape[1],bool);opened=seen.copy();peak=np.zeros(c.shape[1]);states=np.zeros(c.shape,bool)
    for i,price in enumerate(c):
        valid=active[i]&np.isfinite(price)&(price>0)
        peak=np.where(opened&valid,np.maximum(peak,price),peak)
        exited=opened&(~valid|(price<=peak*(1-stop)))
        opened[exited]=False
        fresh=valid&~opened&~exited&((~seen)|(price>=prior[i]))
        peak=np.where(fresh,price,peak);opened[fresh]=True;seen|=valid
        states[i]=opened&valid
    return states


class Policy:
    def __init__(self,parameters):self.parameters=parameters
    def build(self,market,_config):
        p=self.parameters;c=market.panel('close');active=c.notna()&market.panel('volume').gt(0)
        states=state_path(c.to_numpy(),active.to_numpy(),p.stop,p.reentry_window)
        return SimpleNamespace(states=states,breadth=states.mean(axis=1))
    def update(self,i,f,history,config):return (1.,'INDEPENDENT_CASH_SLEEVE')
    def weights(self,i,f,current,config,**kwargs):
        if len(current)!=1:raise ValueError('diagnostic adapter requires exactly one security')
        if not f.states[i,0]:return np.zeros_like(current),['OBSERVED_PEAK_DAMAGE_HOLD_OWN_CASH']
        if current[0]>0:return current.copy(),['RETAIN_OWN_ECONOMIC_UNITS']
        return np.ones_like(current),['INITIAL_OWNERSHIP_OR_FRESH_BREAKOUT']


def run_single(market,parameters,capital,**kwargs):
    if len(market.symbols)!=1:raise ValueError('one sleeve per replay')
    policy=Policy(parameters);cfg=replace(Config(),initial_cash=capital)
    with patch.multiple(engine,build_features=policy.build,RiskState=lambda:policy,target_weights=policy.weights):
        result=engine.run(market,cfg,**kwargs)
    result.metadata['algorithm']={'name':'independent_cash_sleeves_screening','parameters':asdict(parameters),
        'adapter_sha256':file_hash(Path(__file__)),'status':'DIAGNOSTIC_PER_SLEEVE_MATERIALITY_NOT_ACCOUNT_ACCEPTANCE'}
    return result

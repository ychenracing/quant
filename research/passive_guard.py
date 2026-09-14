"""Isolate timing value with a passive reference that remains observable in cash."""
from dataclasses import asdict, dataclass
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
    span:int=20
    confirmation:int=2
    hysteresis:float=.02
    floor:float=0.
    def __post_init__(self):
        for name in ('span','confirmation'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<1:
                raise ValueError('positive integer required')
        if not 0<=self.hysteresis<1 or not 0<=self.floor<1:
            raise ValueError('invalid risk fraction')

def cap_path(nav,p):
    nav=np.asarray(nav,float)
    if nav.ndim!=1 or not len(nav) or not np.isfinite(nav).all() or (nav<=0).any():
        raise ValueError('invalid reference NAV')
    mean=pd.Series(nav).ewm(span=p.span,adjust=False).mean().to_numpy()
    active=True;weak=strong=0;caps=np.ones(len(nav))
    for i,value in enumerate(nav):
        weak=weak+1 if value<mean[i]*(1-p.hysteresis) else 0
        strong=strong+1 if value>mean[i]*(1+p.hysteresis) else 0
        if active and weak>=p.confirmation:active=False
        elif not active and strong>=p.confirmation:active=True
        caps[i]=1. if active else p.floor
    return caps

def observed_actions(market,reference):
    """Only already-observed unit changes or as-yet-unfunded initial admissions."""
    shape=reference.targets.shape;actions=np.zeros(shape,bool);units=np.zeros(shape[1])
    lookup={s:j for j,s in enumerate(market.symbols)};by_date={}
    for order in reference.orders:
        if order['status']=='FILLED':by_date.setdefault(order['date'],[]).append(order)
    targets=reference.targets.to_numpy()
    for i,date in enumerate(market.calendar):
        for order in by_date.get(str(date.date()),[]):
            j=lookup[order['symbol']]
            units[j]+=order['units']*(1 if order['side']=='BUY' else -1)
            actions[i,j]=True
        actions[i]|=(np.abs(units)<1e-10)&(targets[i]>1e-10)
    return actions

class Policy:
    def __init__(self,market,reference,p):
        self.reference=reference;self.caps=cap_path(reference.equity.nav.to_numpy(),p)
        self.targets=reference.targets.to_numpy()
        self.actions=observed_actions(market,reference)
    def build(self,market,config):return SimpleNamespace(breadth=self.reference.equity.breadth.to_numpy())
    def update(self,i,f,history,config):
        return self.caps[i],('PASSIVE_REFERENCE_OPEN' if self.caps[i]==1 else 'PASSIVE_REFERENCE_RISK_REDUCTION')
    def weights(self,i,f,current,config,**kwargs):
        cap=self.caps[i]
        if i==0 or self.caps[i]!=self.caps[i-1]:
            return self.targets[i]*cap,['RISK_TRANSITION_WITH_OBSERVED_PASSIVE_PROPORTIONS']
        want=current.copy()
        # New listings and incomplete initial fills may change reference positions;
        # absent such a change the actual economic units are not rebalanced.
        want[self.actions[i]]=self.targets[i,self.actions[i]]*cap
        if want.sum()>1:want/=want.sum()
        return want,['RETAIN_UNITS_AND_OBSERVED_REFERENCE_ADMISSIONS']

def run(market,parameters=None,*,shadow=None,**kwargs):
    p=parameters or Parameters()
    reference=shadow or engine.run(market,benchmark='buy_hold')
    if reference.metadata['data_sha256']!=market.fingerprint() or not reference.targets.index.equals(market.calendar):
        raise ValueError('reference identity mismatch')
    if reference.metadata['benchmark']!='buy_hold':raise ValueError('passive reference required')
    policy=Policy(market,reference,p)
    with patch.multiple(engine,build_features=policy.build,RiskState=lambda:policy,target_weights=policy.weights):
        result=engine.run(market,Config(**reference.metadata['config']),**kwargs)
    result.metadata['algorithm']={'name':'passive_reference_guard','parameters':asdict(p),
        'adapter_sha256':file_hash(Path(__file__)),
        'status':'RESEARCH_NOT_ACCEPTED'}
    return result

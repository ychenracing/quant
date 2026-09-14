"""Causal shadow-book protection with recovery observable while real capital is cash."""
from dataclasses import asdict,dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from techquant.config import Config
from techquant.data import file_hash
import techquant.engine as engine
from research.breakout import run as unrestricted, Parameters as TrendParameters

@dataclass(frozen=True)
class Parameters:
    span:int=20
    loss:float=.10
    rebound:float=.03
    def __post_init__(self):
        if isinstance(self.span,bool) or not isinstance(self.span,int) or self.span<2:raise ValueError('invalid span')
        if not 0<self.loss<1 or not 0<self.rebound<1:raise ValueError('invalid risk threshold')


def budget_path(nav,p):
    nav=np.asarray(nav,float)
    if nav.ndim!=1 or not len(nav) or not np.isfinite(nav).all() or (nav<=0).any():raise ValueError('invalid shadow NAV')
    mean=pd.Series(nav).ewm(span=p.span,adjust=False).mean().to_numpy()
    active=True;peak=nav[0];trough=nav[0];caps=np.ones(len(nav));reasons=[]
    for i,value in enumerate(nav):
        ret=value/nav[i-1]-1 if i else 0.
        if active:
            peak=max(peak,value)
            if value<=peak*(1-p.loss) or ret<=-.04:
                active=False;trough=value;reason='SHADOW_DRAWDOWN_OR_DAILY_LOSS'
            else:reason='SHADOW_TREND_OPEN'
        else:
            trough=min(trough,value)
            if value>mean[i] and value>=trough*(1+p.rebound) and ret>0:
                active=True;peak=value;reason='OBSERVED_SHADOW_RECOVERY'
            else:reason='SHADOW_RECOVERY_WAIT'
        caps[i]=float(active);reasons.append(reason)
    return caps,reasons


def held_weights(market,shadow):
    units=np.zeros(len(market.symbols));lookup={s:j for j,s in enumerate(market.symbols)}
    by_date={}
    for order in shadow.orders:
        if order['status']=='FILLED':by_date.setdefault(order['date'],[]).append(order)
    weights=np.zeros(shadow.targets.shape);close=market.panel('close').ffill().to_numpy()
    for i,date in enumerate(market.calendar):
        for order in by_date.get(str(date.date()),[]):
            units[lookup[order['symbol']]]+=order['units']*(1 if order['side']=='BUY' else -1)
        weights[i]=np.nan_to_num(units*close[i],nan=0.)/float(shadow.equity.nav.iloc[i])
    return weights


class Policy:
    def __init__(self,market,shadow,p):
        self.shadow=shadow;self.caps,self.reasons=budget_path(shadow.equity.nav.to_numpy(),p)
        self.targets=shadow.targets.to_numpy()
        self.actions=np.abs(self.targets-held_weights(market,shadow))>1e-8
    def build(self,market,config):return SimpleNamespace(breadth=self.shadow.equity.breadth.to_numpy())
    def update(self,i,f,history,config):return self.caps[i],self.reasons[i]
    def weights(self,i,f,current,config,**kwargs):
        cap=self.caps[i]
        if not cap:return np.zeros_like(current),['ACTUAL_CAPITAL_IN_CASH']
        if i==0 or self.caps[i-1]==0:return self.targets[i].copy(),['RESTORE_OBSERVED_SHADOW_HOLDINGS']
        want=current.copy();want[self.actions[i]]=self.targets[i,self.actions[i]]
        if want.sum()>1:want/=want.sum()
        return want,['FOLLOW_FRESH_TREND_ACTIONS_RETAIN_OTHER_UNITS']


def run(market,parameters=None,*,shadow=None,**kwargs):
    p=parameters or Parameters()
    base_parameters=TrendParameters(slow=20,stop=.12,positions=2,entry_window=10)
    if shadow is None:shadow=unrestricted(market,base_parameters)
    if shadow.metadata['data_sha256']!=market.fingerprint() or not shadow.targets.index.equals(market.calendar):
        raise ValueError('shadow snapshot identity mismatch')
    if shadow.metadata['algorithm']['parameters']!=asdict(base_parameters):raise ValueError('shadow policy identity mismatch')
    policy=Policy(market,shadow,p)
    with patch.multiple(engine,build_features=policy.build,RiskState=lambda:policy,target_weights=policy.weights):
        result=engine.run(market,Config(**shadow.metadata['config']),**kwargs)
    result.metadata['algorithm']={'name':'shadow_book_drawdown_and_recovery','parameters':asdict(p),
        'adapter_sha256':file_hash(Path(__file__)),'shadow_policy':shadow.metadata['algorithm'],
        'status':'RESEARCH_NOT_ACCEPTED'}
    return result

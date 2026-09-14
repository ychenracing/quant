"""Downside-only shock scaling, with an independently observable regime reference."""
from dataclasses import asdict,dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from techquant.config import Config
from techquant.data import file_hash
import techquant.engine as engine
from research.rotation import run as alpha_run, Parameters as AlphaParameters
from research.shadow_guard import held_weights

@dataclass(frozen=True)
class Parameters:
    shock_z:float=2.5
    regime_span:int=80
    recovery:int=2
    bear_cap:float=.25
    def __post_init__(self):
        if not np.isfinite(self.shock_z) or self.shock_z<=0 or not 0<=self.bear_cap<1:
            raise ValueError('invalid risk scale')
        for name in ('regime_span','recovery'):
            x=getattr(self,name)
            if isinstance(x,bool) or not isinstance(x,int) or x<1:raise ValueError('positive integer required')

def budget_path(alpha,passive,p):
    alpha=np.asarray(alpha,float);passive=np.asarray(passive,float)
    if alpha.ndim!=1 or alpha.shape!=passive.shape or not len(alpha) or not np.isfinite(alpha).all() or not np.isfinite(passive).all() or (alpha<=0).any() or (passive<=0).any():
        raise ValueError('invalid reference paths')
    ar=pd.Series(alpha).pct_change(fill_method=None).fillna(0)
    pr=pd.Series(passive).pct_change(fill_method=None).fillna(0).to_numpy()
    downside=np.sqrt(ar.clip(upper=0).pow(2).rolling(20,min_periods=5).mean().shift()).fillna(.008).clip(lower=.008).to_numpy()
    slow=pd.Series(passive).ewm(span=p.regime_span,adjust=False).mean()
    fast=pd.Series(passive).ewm(span=10,adjust=False).mean().to_numpy()
    bear=((passive<slow)&(slow<slow.shift(10))).to_numpy()
    state=1.;healthy=0;caps=np.ones(len(alpha));reasons=[]
    for i,value in enumerate(alpha):
        if ar.iloc[i]<-max(.025,p.shock_z*downside[i]):
            state=0.;healthy=0;reason='DOWNSIDE_SCALED_SHOCK'
        else:
            desired=p.bear_cap if bear[i] else 1.
            if desired<state:
                state=desired;healthy=0;reason='SLOW_TECHNOLOGY_BEAR'
            elif desired>state:
                healthy=healthy+1 if passive[i]>fast[i] and pr[i]>=0 else 0
                if healthy>=p.recovery:
                    state=desired;healthy=0;reason='INDEPENDENT_REFERENCE_RECOVERY'
                else:reason='REFERENCE_RECOVERY_WAIT'
            else:reason='REGIME_RISK_BUDGET'
        caps[i]=state;reasons.append(reason)
    return caps,reasons

class Policy:
    def __init__(self,market,alpha,passive,p):
        self.alpha=alpha;self.caps,self.reasons=budget_path(alpha.equity.nav.to_numpy(),passive.equity.nav.to_numpy(),p)
        self.targets=alpha.targets.to_numpy();self.actions=np.abs(self.targets-held_weights(market,alpha))>1e-8
    def build(self,market,config):return SimpleNamespace(breadth=self.alpha.equity.breadth.to_numpy())
    def update(self,i,f,history,config):return self.caps[i],self.reasons[i]
    def weights(self,i,f,current,config,**kwargs):
        cap=self.caps[i]
        if cap==0:return np.zeros_like(current),['CASH_REDUCTION_RETRY_UNTIL_FILLED']
        if i==0 or self.caps[i]!=self.caps[i-1]:
            return self.targets[i]*cap,['RISK_TRANSITION_RETAINS_ALPHA_MEMBERSHIP']
        want=current.copy();want[self.actions[i]]=self.targets[i,self.actions[i]]*cap
        if want.sum()>1:want/=want.sum()
        return want,['PERSISTENT_ALPHA_OWNERSHIP']

def run(market,parameters=None,*,alpha=None,passive=None,**kwargs):
    p=parameters or Parameters();base=AlphaParameters(lookback=40,trend=80,positions=2,review=20)
    alpha=alpha if alpha is not None else alpha_run(market,base)
    passive=passive if passive is not None else engine.run(market,benchmark='buy_hold')
    for ref in (alpha,passive):
        if ref.metadata['data_sha256']!=market.fingerprint() or not ref.targets.index.equals(market.calendar):
            raise ValueError('reference identity mismatch')
    if alpha.metadata['algorithm']['parameters']!=asdict(base) or passive.metadata['benchmark']!='buy_hold':
        raise ValueError('unexpected reference policy')
    policy=Policy(market,alpha,passive,p)
    with patch.multiple(engine,build_features=policy.build,RiskState=lambda:policy,target_weights=policy.weights):
        result=engine.run(market,Config(**alpha.metadata['config']),**kwargs)
    result.metadata['algorithm']={'name':'downside_scaled_regime_guard','parameters':asdict(p),
        'adapter_sha256':file_hash(Path(__file__)),'alpha_policy':alpha.metadata['algorithm'],
        'dependencies':{'research/shadow_guard.py':file_hash(Path('research/shadow_guard.py'))},
        'status':'RESEARCH_NOT_ACCEPTED'}
    return result

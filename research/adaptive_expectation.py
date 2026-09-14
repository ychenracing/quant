"""Online, matured-label return/downside learning for supplied technology names.

Every refit standardizes and learns only from fully observed historical labels.
The policy never refits an entire backtest and then applies that model backwards.
Predictions are research estimates, not calibrated probabilities or guarantees.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import techquant.engine as engine
from techquant.config import Config
from techquant.data import file_hash
from research.directional import allocate


@dataclass(frozen=True)
class Parameters:
    horizon: int = 20
    shrinkage: float = 1.
    positions: int = 2

    def __post_init__(self):
        for name in ('horizon','positions'):
            v=getattr(self,name)
            if isinstance(v,bool) or not isinstance(v,int) or v<1:
                raise ValueError('positive integral horizon and capacity required')
        if isinstance(self.shrinkage,bool) or not np.isfinite(self.shrinkage) or self.shrinkage<=0:
            raise ValueError('positive finite shrinkage required')


def features(market):
    q=market.panel('close');c=q.ffill();volume=market.panel('volume')
    active=q.notna()&volume.gt(0);count=active.cumsum()
    ready=active&count.ge(10)&active.rolling(10,min_periods=10).mean().ge(.8)
    first=q.where(active&count.eq(1)).ffill();since=c/first
    momenta=[np.log((c/c.shift(h)).fillna(since)) for h in (5,20,60,120)]
    ret=c.pct_change(fill_method=None);ema10=c.ewm(span=10,adjust=False).mean()
    ema20=c.ewm(span=20,adjust=False).mean();ema60=c.ewm(span=60,adjust=False).mean()
    downside=ret.clip(upper=0).pow(2).rolling(20,min_periods=5).mean().pow(.5)
    vol20=ret.rolling(20,min_periods=5).std(ddof=0).clip(lower=.003)
    ratio=ret.rolling(5,min_periods=3).std(ddof=0)/vol20
    vratio=np.log(volume/volume.rolling(20,min_periods=5).mean().shift())
    high,low=market.panel('high'),market.panel('low')
    location=(c-low)/(high-low).replace(0,np.nan)
    dd=c/c.rolling(20,min_periods=5).max()-1
    breadth=(c.gt(ema20)&ready).sum(axis=1).div(ready.sum(axis=1).replace(0,np.nan)).fillna(0)
    blocks=[ret,*momenta,np.log(ema20/ema60),downside,ratio,vratio,location,dd,
            pd.DataFrame(np.broadcast_to(breadth.to_numpy()[:,None],c.shape),index=c.index,columns=c.columns),
            momenta[2]*ret.clip(upper=0),momenta[1].pow(2)]
    x=np.stack([b.to_numpy() for b in blocks],axis=-1)
    return x,ready.to_numpy(),c.to_numpy(),q.to_numpy(),market.panel('open').to_numpy(),breadth.to_numpy(),c.gt(ema10).to_numpy(),momenta


def matured_labels(quoted,opening,horizon):
    n,k=quoted.shape;y=np.full((n,k,2),np.nan)
    for t in range(n-horizon):
        future=quoted[t+1:t+horizon+1];start=opening[t+1]
        valid=np.isfinite(future).all(axis=0)&np.isfinite(start)&(start>0)
        j=np.flatnonzero(valid)
        y[t,j,0]=np.log(quoted[t+horizon,j]/start[j])
        y[t,j,1]=np.maximum(0.,1-np.min(future[:,j],axis=0)/start[j])
    return y


def fit_ridge(x,y,valid,signal,horizon,shrinkage):
    """The maturity bound is enforced here, not left to a caller's slicing."""
    last=signal-horizon
    begin=max(0,last-503)
    dates=np.arange(begin,last+1)
    mask=valid[begin:last+1]&np.isfinite(y[begin:last+1]).all(axis=2)
    counts=mask.sum(axis=1)
    if np.count_nonzero(counts)<20:return None
    w=np.exp2((dates-signal)/120)[:,None]/np.maximum(counts[:,None],1)
    w=np.broadcast_to(w,mask.shape)[mask];w/=w.sum()
    design=x[begin:last+1][mask];response=y[begin:last+1][mask]
    center=np.sum(w[:,None]*design,axis=0)
    scale=np.sqrt(np.sum(w[:,None]*(design-center)**2,axis=0)).clip(min=1e-6)
    z=(design-center)/scale
    target_mean=np.sum(w[:,None]*response,axis=0)
    gram=z.T@(w[:,None]*z)+shrinkage*np.eye(z.shape[1])
    coefficients=np.linalg.solve(gram,z.T@(w[:,None]*(response-target_mean)))
    identity={'signal_index':signal,'last_label_start':int(dates[counts>0][-1]),
              'distinct_label_dates':int(np.count_nonzero(counts)),'sample_count':int(mask.sum()),
              'fit_sha256':hashlib.sha256(np.concatenate((center,scale,target_mean,coefficients.ravel())).tobytes()).hexdigest()}
    return center,scale,target_mean,coefficients,identity


class Policy:
    def __init__(self,parameters):
        self.parameters=parameters;self.fits=[];self.pause_until=-1

    def build(self,market,_config):
        p=self.parameters
        x,ready,close,quoted,opening,breadth,above_fast,momenta=features(market)
        valid=ready&np.isfinite(x).all(axis=2)
        labels=matured_labels(quoted,opening,p.horizon)
        prior=sum(m.to_numpy() for m in momenta)/len(momenta)
        score=np.full(close.shape,-np.inf);model=None
        for i in range(len(close)):
            if i%20==0 and i>=p.horizon:
                fitted=fit_ridge(x,labels,valid,i,p.horizon,p.shrinkage)
                if fitted is not None:
                    model=fitted;self.fits.append(fitted[-1])
            eligible=valid[i]
            if model is None:score[i,eligible]=prior[i,eligible]
            else:
                center,scale,mean,coefficients,_=model
                pred=(x[i,eligible]-center)/scale@coefficients+mean
                # Finite bounds prevent an extrapolated regression from declaring
                # unbounded returns or negative loss. They are not acceptance gates.
                score[i,eligible]=np.clip(pred[:,0],-.7,.7)-np.clip(pred[:,1],0,.5)
        negative=score<0
        broken=negative&np.vstack((np.zeros((1,len(market.symbols)),bool),negative[:-1]))
        self.owned=np.zeros(len(market.symbols),bool)
        self.peak=np.zeros(len(market.symbols))
        self.exit_latched=np.zeros(len(market.symbols),bool)
        return SimpleNamespace(close=close,ready=ready,score=score,breadth=breadth,
            entry=valid&above_fast&(momenta[0].to_numpy()>0)&(score>0),broken=broken)

    def update(self,i,f,history,config):return 1.,'CAUSALLY_MATURED_RETURN_DOWNSIDE'

    def weights(self,i,f,current,config,*,rebalance=False,**kwargs):
        held=current>1e-10
        new=held&~self.owned
        self.peak[new]=f.close[i,new]
        self.peak[held]=np.maximum(self.peak[held],f.close[i,held])
        self.peak[~held]=0.;self.exit_latched[~held]=False
        damaged=held&(~f.ready[i]|f.broken[i]|(f.close[i]<=self.peak*.88))
        new_exit=damaged&~self.exit_latched
        if new_exit.any():
            self.exit_latched|=new_exit;self.pause_until=i+5
        self.owned=held
        want=current.copy();want[self.exit_latched]=0.
        if i<self.pause_until or self.exit_latched.any():
            return want,['PROTECTIVE_EXIT_LATCH_AND_NEW_MONEY_PAUSE']
        intact=f.ready[i]&~self.exit_latched
        want=allocate(current,intact,f.entry[i],f.score[i],self.parameters.positions,rebalance)
        return want,['MATURED_EXPECTATION_ADMISSION_OR_PERSISTENT_UNITS']


def run(market,parameters=None,**kwargs):
    p=parameters or Parameters();policy=Policy(p)
    cfg=replace(Config(),max_positions=p.positions,rebalance=20,single_cap=.6,sector_cap=1.)
    with patch.multiple(engine,build_features=policy.build,RiskState=lambda:policy,target_weights=policy.weights):
        result=engine.run(market,cfg,**kwargs)
    result.metadata['algorithm']={'name':'adaptive_return_downside_expectation','parameters':asdict(p),
        'adapter_sha256':file_hash(Path(__file__)),'allocation_sha256':file_hash(Path('research/directional.py')),
        'fit_history':policy.fits,'status':'RESEARCH_NOT_ACCEPTED'}
    return result

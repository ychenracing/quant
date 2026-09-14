"""Separate causal return ranking and matured near-term downside probabilities.

Small fixed-complexity tree ensembles are research dependencies, not references
or a changed execution engine. Neither model sees future validation partitions,
security identifiers, outside-universe quotes or labels which have not matured.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import hashlib
from importlib.metadata import version
import itertools
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from techquant.data import Market,file_hash
from techquant.policy import CloseDecision,CloseObservation
from research.expectation import _features,matured_labels

MODEL=dict(learning_rate=.08,max_iter=64,max_leaf_nodes=7,min_samples_leaf=50,
           l2_regularization=1.,early_stopping=False,random_state=0)


@dataclass(frozen=True)
class Parameters:
    horizon:int=20
    tail_threshold:float=.25
    positions:int=2

    def __post_init__(self):
        if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in (self.horizon,self.positions)):
            raise ValueError('horizon and positions must be positive integers')
        if not np.isfinite(self.tail_threshold) or not 0<self.tail_threshold<1:
            raise ValueError('tail threshold must lie strictly between zero and one')


def grid():
    return [Parameters(*v) for v in itertools.product((20,60),(.25,.5),(2,4))]


def runtime():
    return {name:version(name) for name in ('scikit-learn','scipy','joblib','threadpoolctl')}


@dataclass
class Prediction:
    symbols:tuple[str,...]
    data_sha256:str
    horizon:int
    expected:np.ndarray
    tail:np.ndarray
    ready:np.ndarray
    price:np.ndarray
    ema10:np.ndarray
    ema20:np.ndarray
    ema60:np.ndarray
    momentum5:np.ndarray
    ret1:np.ndarray
    fits:list[dict[str,Any]]

    def fingerprint(self):
        h=hashlib.sha256()
        for x in (self.expected,self.tail,self.ready):h.update(np.asarray(x,dtype='<f8').tobytes())
        return h.hexdigest()


def tree_hash(model):
    # Numeric fields are serialized independently, excluding struct padding.
    # Private fitted representation is deliberately bound to sklearn's version.
    h=hashlib.sha256(np.asarray(model._baseline_prediction,dtype='<f8').tobytes())
    for iteration in model._predictors:
        for predictor in iteration:
            for field in predictor.nodes.dtype.names:
                h.update(field.encode());h.update(np.ascontiguousarray(predictor.nodes[field]).tobytes())
    return h.hexdigest()


def training(x,ready,labels,cutoff,horizon):
    last=len(labels);first=max(0,last-504)
    valid=ready[first:last]&np.isfinite(labels[first:last]).all(axis=-1)
    counts=valid.sum(axis=1);dates=np.flatnonzero(counts)+first
    if len(dates)<40:return None
    decay=np.exp2((np.arange(first,last)-(last-1))/120.)
    weights=np.broadcast_to((decay/np.maximum(counts,1))[:,None],valid.shape)[valid].copy()
    weights*=len(dates)/weights.sum()
    features=x[first:last][valid];targets=labels[first:last][valid]
    digest=hashlib.sha256()
    for a in (features,targets,weights):digest.update(np.asarray(a,dtype='<f8').tobytes())
    receipt={'session':cutoff,'first_feature_session':int(dates[0]),'last_feature_session':int(dates[-1]),
             'last_label_session':int(dates[-1]+horizon),'distinct_dates':len(dates),'samples':len(weights),
             'training_sha256':digest.hexdigest()}
    return features,targets,weights,receipt


def forecast(market:Market,p:Parameters)->Prediction:
    x,valid,price,ema10,momentum5,prior=_features(market)
    quote=market.panel('close');open_=market.panel('open').to_numpy()
    raw=quote.to_numpy();close=quote.ffill()
    ready=valid&((quote.notna()&market.panel('volume').gt(0)).cumsum().to_numpy()>=20)
    expected=prior.copy()*p.horizon/60.;tail=np.full(prior.shape,.1)
    returns=None;risk=None;fits=[]
    for i in range(len(market.calendar)):
        if i%10==0:
            # Fit each task only on its own independently matured label boundary.
            for task,horizon in (('return',p.horizon),('tail',5)):
                labels=matured_labels(raw,open_,horizon=horizon,cutoff=i)
                sample=training(x,valid,labels,i,horizon)
                if sample is None:continue
                xx,yy,weights,receipt=sample
                receipt['task']=task;receipt['date']=str(market.calendar[i].date())
                with threadpool_limits(limits=1):
                    if task=='return':
                        returns=HistGradientBoostingRegressor(**MODEL).fit(xx,yy[:,0],sample_weight=weights)
                        receipt['tree_sha256']=tree_hash(returns)
                    else:
                        events=(yy[:,1]>=.08).astype(int)
                        if len(np.unique(events))==1:
                            risk=float((weights@events+1.)/(weights.sum()+2.))
                            receipt['smoothed_prevalence']=risk
                        else:
                            risk=HistGradientBoostingClassifier(**MODEL).fit(xx,events,sample_weight=weights)
                            receipt['tree_sha256']=tree_hash(risk)
                fits.append(receipt)
        good=valid[i]
        if good.any():
            with threadpool_limits(limits=1):
                if returns is not None:expected[i,good]=returns.predict(x[i,good])
                if risk is not None:
                    tail[i,good]=risk if isinstance(risk,float) else risk.predict_proba(x[i,good])[:,1]
    if not np.isfinite(expected).all() or not np.isfinite(tail).all():
        raise AssertionError('non-finite model forecast')
    result=Prediction(market.symbols,market.fingerprint(),p.horizon,expected,tail,ready,price,ema10,
        close.ewm(span=20,adjust=False).mean().to_numpy(),close.ewm(span=60,adjust=False).mean().to_numpy(),
        momentum5,close.pct_change(fill_method=None).to_numpy(),fits)
    for a in (result.expected,result.tail,result.ready,result.price,result.ema10,result.ema20,result.ema60,result.momentum5,result.ret1):
        a.setflags(write=False)
    return result


_CACHE:dict[tuple,Prediction]={}


def prepared(market,p):
    # Cache immutable predictions only, never inventory/ownership state. Changing
    # a subset, calendar, source dependency or learner runtime invalidates it.
    key=(market.fingerprint(),p.horizon,file_hash(Path(__file__)),
         file_hash(Path(__file__).with_name('expectation.py')),tuple(runtime().items()))
    if key not in _CACHE:_CACHE[key]=forecast(market,p)
    return _CACHE[key]


class Owner:
    def __init__(self,market:Market,p:Parameters):
        self.market,self.params=market,p;self.f=prepared(market,p)
        n=len(market.symbols)
        self.negative=np.zeros(n,dtype=int);self.healthy=np.zeros(n,dtype=int)
        self.previous=np.zeros(n,dtype=bool);self.exit_pending=np.zeros(n,dtype=bool)
        self.readmit=np.zeros(n,dtype=bool);self.last_session=-1

    def decide(self,o:CloseObservation)->CloseDecision:
        i=o.session;p,f=self.params,self.f
        if i<=self.last_session:raise ValueError('policy sessions must increase')
        self.last_session=i
        held=o.units>1e-10;sold=self.previous&~held
        self.readmit[sold&self.exit_pending]=True;self.healthy[sold]=0
        self.exit_pending[~held]=False
        good=f.ready[i]&(f.price[i]>f.ema10[i])&(f.tail[i]<p.tail_threshold*.5)
        self.healthy=np.where(good,self.healthy+1,0);self.readmit[self.healthy>=3]=False
        self.negative=np.where(f.expected[i]<-.02,self.negative+1,0)
        tail_alert=(f.tail[i]>=p.tail_threshold)&(f.ret1[i]<0)
        broken=(tail_alert|((self.negative>=3)&(f.price[i]<f.ema60[i]))|(f.ret1[i]<=-.08)|~f.ready[i])
        self.exit_pending|=held&broken
        want=o.weights.copy();want[self.exit_pending]=0.;why=[]
        if self.exit_pending.any():why.append('FULL_EXIT_RETRY')
        want=np.minimum(want,1. if len(held)==1 else .8)
        if not self.exit_pending.any():
            score=f.expected[i]
            ranked=sorted(np.flatnonzero(f.ready[i]&(score>.005)),key=lambda j:(-score[j],self.market.symbols[j]))
            entrants=[j for j in ranked if not held[j] and not self.readmit[j] and f.momentum5[i,j]>0
                      and f.price[i,j]>f.ema20[i,j] and f.tail[i,j]<p.tail_threshold]
            existing=list(np.flatnonzero(held));capacity=min(p.positions,len(held))
            if i%20==0 and len(existing)>=capacity and entrants:
                ranks={j:k+1 for k,j in enumerate(ranked)}
                worst=min(existing,key=lambda j:(score[j],self.market.symbols[j]))
                if ranks.get(worst,len(held)+1)>2*capacity and score[entrants[0]]>1.25*max(score[worst],.001):
                    want[worst]=0.;self.exit_pending[worst]=True
                    why.append('EXPECTED_RETURN_LEADER_REPLACEMENT')
            entrants=entrants[:max(0,capacity-len(existing))]
            cash=min(o.cash/o.nav,max(0.,1.-o.weights.sum()))
            if entrants and cash>=.01:
                want[entrants]=min(cash/len(entrants),1. if len(held)==1 else .6)
                why.append('FUNDED_EXPECTED_RETURN_ENTRY')
        self.previous=held.copy()
        # Signal-close units preserve inventory through overnight gaps. The engine
        # still controls affordability, lots, T+1 and all actual execution.
        marks=f.price[i]
        targets=np.divide(want*o.nav,marks,out=np.zeros_like(want),
                          where=np.isfinite(marks)&(marks>0))
        unchanged=want==o.weights
        targets[unchanged]=o.units[unchanged]
        return CloseDecision(want,'|'.join(why) if why else 'RETAIN_PREDICTED_TRENDS',
                             unit_targets=targets)

    def identity(self):
        return {'name':'separate_return_and_tail_forecasts','parameters':asdict(self.params),
            'implementation_sha256':file_hash(Path(__file__)),'feature_sha256':file_hash(Path(__file__).with_name('expectation.py')),
            'data_sha256':self.f.data_sha256,'forecast_sha256':self.f.fingerprint(),'fits':self.f.fits,
            'learner_parameters':MODEL,'learner_runtime':runtime(),'status':'RESEARCH_NOT_ACCEPTED'}

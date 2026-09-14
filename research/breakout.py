"""Independent event-driven breakthrough ownership experiment.

Only confirmed economic units create a cost basis or a holding peak. Stops are
closing signals, never intraday fills. The unchanged cash engine executes them
at a later open. This research adapter is deliberately separate from production.
"""
from dataclasses import dataclass,asdict
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
import numpy as np
from techquant.config import Config
from techquant.data import file_hash
import techquant.engine as engine

@dataclass(frozen=True)
class Parameters:
    entry_window:int=40
    stop_atr:float=2.5
    entry_cap:float=.5
    exit_style:str='trailing'
    def __post_init__(self):
        if self.entry_window<10 or not 0<self.stop_atr<10 or not 0<self.entry_cap<=1:
            raise ValueError('invalid breakthrough parameters')
        if self.exit_style not in ('trailing','confirmed'):raise ValueError('invalid exit style')

class Ownership:
    """Per-symbol confirmed position state, with persistent sell intent."""
    def __init__(self,n):
        self.units=np.zeros(n);self.cost=np.zeros(n);self.peak=np.zeros(n)
        self.stop=np.zeros(n);self.liquidating=np.zeros(n,bool)
        self.goal=np.zeros(n)
    def observe(self,units,opening,closing):
        units=np.asarray(units,float)
        if not np.isfinite(units).all() or (units<0).any():raise ValueError('invalid confirmed units')
        added=np.maximum(0.,units-self.units)
        increase=added>1e-7
        if np.any(increase & ~np.isfinite(opening)):raise ValueError('new units without observed opening quote')
        old_cost= self.cost*np.minimum(self.units,units)
        self.cost[increase]=(old_cost[increase]+added[increase]*opening[increase])/units[increase]
        flat=units<1e-7
        self.cost[flat]=0.;self.peak[flat]=0.;self.stop[flat]=0.;self.liquidating[flat]=False
        held=~flat
        self.peak[held]=np.maximum(self.peak[held],closing[held])
        self.units=units.copy()
    def stops(self,atr,multiple):
        distance=np.clip(multiple*np.asarray(atr,float),.06*self.peak,.22*self.peak)
        fresh=self.peak-distance
        self.stop=np.maximum(self.stop,np.nan_to_num(fresh,nan=0.))
        return np.maximum(self.stop,.88*self.cost)

class Policy:
    def __init__(self,params):self.params=params;self.nav=1.;self.state=None
    def build(self,market,config):
        p=self.params;q=market.panel('close');close=q.ffill()
        active=q.notna() & market.panel('volume').gt(0)
        count=active.cumsum();ready=active & count.ge(11) & active.rolling(10,min_periods=10).mean().ge(.8)
        first=q.where(active & count.eq(1)).ffill()
        r20=(close/close.shift(20)).fillna(close/first)
        r60=(close/close.shift(60)).fillna(close/first)
        score=(np.log(r20)+np.log(r60))/2
        high=close.shift().rolling(p.entry_window,min_periods=10).max()
        slow=close.rolling(60,min_periods=10).mean();fast=close.rolling(10,min_periods=10).mean()
        entry=ready & close.gt(high) & close.gt(slow) & r20.gt(1)
        prev=close.shift();h=market.panel('high');l=market.panel('low')
        tr=np.maximum(h-l,np.maximum((h-prev).abs(),(l-prev).abs()))
        atr=tr.rolling(20,min_periods=10).mean()
        self.state=Ownership(len(market.symbols))
        breadth=(close.gt(fast)&ready).sum(axis=1)/ready.sum(axis=1).replace(0,np.nan)
        return SimpleNamespace(symbols=market.symbols,ready=ready.to_numpy(),entry=entry.to_numpy(),
           score=score.where(ready,-np.inf).to_numpy(),close=close.to_numpy(),op=market.panel('open').to_numpy(),
           atr=atr.to_numpy(),fast=fast.to_numpy(),breadth=breadth.fillna(0.).to_numpy())
    def update(self,i,f,history,config):
        self.nav=history[-1]
        return 1.,'EVENT_DRIVEN_OWNERSHIP'
    def weights(self,i,f,current,config,**kwargs):
        s=self.state;p=self.params
        units=np.divide(current*self.nav,f.close[i],out=np.zeros_like(current),where=np.isfinite(f.close[i]))
        slip=config.slippage_bps/10000
        s.observe(units,f.op[i]*(1+slip),f.close[i])
        held=current>1e-9
        stop=s.stops(f.atr[i],p.stop_atr)
        touched=f.close[i]<=stop
        if p.exit_style=='confirmed':touched &= (f.close[i]<f.fast[i]) | (f.close[i]<=.88*s.cost)
        s.liquidating |= held & touched & f.ready[i]
        s.goal[s.liquidating]=0.
        want=current.copy();want[s.liquidating]=0.
        # Fixed-unit entry intentions complete capacity-limited fills; this is
        # not repeated averaging or a target reset when a winner appreciates.
        remaining=np.maximum(0.,s.goal-units)*np.nan_to_num(f.close[i])/self.nav
        canceled=(~f.ready[i]) | (f.close[i]<f.fast[i]) | s.liquidating | (remaining<.01)
        s.goal[canceled]=0.;remaining[canceled]=0.
        budget=max(0.,1-want.sum())
        pending=np.flatnonzero(remaining>0)
        for j in pending:
            add=min(budget,remaining[j]);want[j]+=add;budget-=add
        eligible=f.entry[i] & ~held & ~s.liquidating & (s.goal<=units+1e-7)
        order=sorted(np.flatnonzero(eligible),key=lambda j:(-f.score[i,j],f.symbols[j]))
        for j in order:
            if budget<.05:break
            amount=min(budget,max(p.entry_cap,1/len(current)))
            want[j]+=amount;budget-=amount
            s.goal[j]=units[j]+amount*self.nav/f.close[i,j]
        reasons=['CONFIRMED_HOLDING_STOP' if s.liquidating.any() else 'NO_EXIT']
        if np.any(want>current+1e-9):reasons.append('CASH_FUNDED_FRESH_BREAKOUT')
        return want,reasons

def run(market,params=None,**kwargs):
    params=params or Parameters();policy=Policy(params);config=Config()
    with patch.multiple(engine,build_features=policy.build,RiskState=lambda:policy,target_weights=policy.weights):
        result=engine.run(market,config,**kwargs)
    result.metadata['algorithm']={'name':'event_driven_breakout_ownership','parameters':asdict(params),
        'adapter_sha256':file_hash(Path(__file__)),'status':'RESEARCH_NOT_ACCEPTED'}
    return result

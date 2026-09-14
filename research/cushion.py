"""Permanent-high-water capital cushion over independent funded admission.

The 10% participation floor permits recovery; it is not a hard drawdown promise.
Unit-based reductions stay pending through blocked fills and price rebounds.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import itertools
import numpy as np
from techquant.data import Market,file_hash
from techquant.policy import CloseDecision,CloseObservation
from research.admission import Owner as Admission,Parameters as AdmissionParameters


@dataclass(frozen=True)
class Parameters:
    drawdown_budget:float=.2
    multiplier:float=5.

    def __post_init__(self):
        if not np.isfinite(self.drawdown_budget) or not 0<self.drawdown_budget<1:
            raise ValueError('drawdown budget must be finite and between zero and one')
        if not np.isfinite(self.multiplier) or self.multiplier<=0:
            raise ValueError('multiplier must be finite and positive')


def grid():
    return [Parameters(*v) for v in itertools.product((.16,.20),(5.,8.,12.))]


class Owner:
    def __init__(self,market:Market,p:Parameters):
        self.market,self.params=market,p
        self.inner=Admission(market,AdmissionParameters('responsive',20,.6))
        self.price=market.panel('close').ffill().to_numpy()
        self.peak=0.
        self.pending_units=np.full(len(market.symbols),np.nan)
        self.last_session=-1

    def decide(self,o:CloseObservation)->CloseDecision:
        if o.session<=self.last_session:raise ValueError('policy sessions must increase')
        self.last_session=o.session
        self.peak=max(self.peak,o.nav)
        price=self.price[o.session]
        cap=float(np.clip(self.params.multiplier*(o.nav-(1-self.params.drawdown_budget)*self.peak)/o.nav,.1,1.))
        requested=self.inner.decide(o)
        want=requested.weights.copy()
        why=[requested.reason]
        pending=np.isfinite(self.pending_units)
        # A full exit must really fill; other reductions allow ordinary lot dust.
        actual_gap=np.nan_to_num((o.units-self.pending_units)*price,nan=0.)
        done=pending & ((o.units<=self.pending_units+1e-10) |
                        ((self.pending_units>0)&(actual_gap<=.01*o.nav)))
        self.pending_units[done]=np.nan
        pending=np.isfinite(self.pending_units)
        want[pending]=np.minimum(want[pending],self.pending_units[pending]*price[pending]/o.nav)
        if pending.any():why.append('CUSHION_REDUCTION_RETRY')
        if want.sum()>cap+.01:
            want*=cap/want.sum()
            lower=want<o.weights-1e-10
            desired=np.zeros(len(price))
            np.divide(want*o.nav,price,out=desired,where=np.isfinite(price)&(price>0))
            self.pending_units[lower]=np.fmin(self.pending_units[lower],desired[lower])
            why.append('PERMANENT_PEAK_RISK_REDUCTION')
        elif not pending.any() and self.inner.admit[o.session] and cap-o.weights.sum()>=.10:
            # Restore only surviving intended members; zero-target exits stay zero.
            survivors=(o.units>1e-10)&(want>=o.weights-1e-10)&(want>0)
            if survivors.any():
                cash=min(o.cash/o.nav,max(0.,1.-o.weights.sum()))
                already=float(np.maximum(want-o.weights,0.).sum())
                room=min(max(0.,cash-already),max(0.,cap-want.sum()))
                if room>0:
                    share=want[survivors]/want[survivors].sum()
                    want[survivors]+=room*share
                    want=np.minimum(want,1. if len(price)==1 else .8)
                    why.append('CASH_FUNDED_CUSHION_RESTORATION')
        # Requested new positions may exceed cap even without actual exposure;
        # scaling requests never creates money and does not modify fill accounting.
        if want.sum()>1.+1e-10:raise AssertionError('cushion created unfunded exposure')
        return CloseDecision(want,'|'.join(why),cap=min(1.,max(cap,float(want.sum()))))

    def identity(self):
        return {'name':'capital_cushion_protection','parameters':asdict(self.params),
                'implementation_sha256':file_hash(Path(__file__)),
                'admission_sha256':file_hash(Path(__file__).with_name('admission.py')),
                'ownership_sha256':file_hash(Path(__file__).with_name('leadership.py')),
                'data_sha256':self.market.fingerprint(),'status':'RESEARCH_NOT_ACCEPTED'}

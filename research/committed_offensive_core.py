"""Event-driven offensive ownership: hold intact names until their own break.

Cross-sectional score is consulted only when an actual slot is empty.  Existing
inventory is never displaced merely because another eligible name ranks higher.
The policy uses only observed cash for new units and never anticipates sale
proceeds.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
from techquant.config import Config
from techquant.data import Market, file_hash
from techquant.features import build_features
from techquant.policy import CloseDecision, CloseObservation
from research.observed_trend import Parameters as TrendParameters, signals
from research.trend_book import Owner as ParentOwner, Parameters as ParentParameters, price_signals
from research.quantity_obligation import preserve_trace, verify_trace


@dataclass(frozen=True)
class Parameters:
    enabled: bool
    def __post_init__(self):
        if type(self.enabled) is not bool: raise ValueError('enabled must be boolean')


def grid(): return [Parameters(False), Parameters(True)]


class Owner:
    def __init__(self, market: Market, parameters: Parameters):
        if type(parameters) is not Parameters: raise ValueError('committed offensive core requires registered parameters')
        self.market=market; self.parameters=parameters; self.config=Config()
        self.parent=ParentOwner(market,ParentParameters(2))
        self.features=build_features(market,self.config)
        self.price_signals=price_signals(market)
        self.trend=signals(market,TrendParameters(trend_span=60,require_market_trend=True))
        self.last_session=-1; self.trace=[]

    def _decision(self, o: CloseObservation):
        i=o.session
        if i<=self.last_session: raise ValueError('policy sessions must increase')
        self.last_session=i
        p=self.price_signals; score=self.features.score[i]
        held=o.units>1e-10
        broken=self.trend.exit[i] | (p.ret1[i] <= -.08) | ~p.ready[i]
        broken_held=held & broken
        marks=np.nan_to_num(p.price[i],nan=0.)

        # Security-specific break is authoritative and never releases forecast
        # proceeds into a same-close replacement request.
        if broken_held.any():
            units=o.units.copy(); units[broken_held]=0.
            weights=np.divide(units*marks,o.nav,out=np.zeros_like(units),where=o.nav>0)
            symbols=[self.market.symbols[j] for j in np.flatnonzero(broken_held)]
            self.trace.append({'kind':'COMMITTED_OFFENSIVE_EVENT','date':o.date,'session':int(i),
                               'action':'SECURITY_EXIT','symbols':symbols})
            return CloseDecision(weights,'COMMITTED_OFFENSIVE|SECURITY_EXIT',1.,units)

        survivors=np.flatnonzero(held)
        vacancy=max(0,self.config.max_positions-len(survivors))
        if vacancy==0 or not self.trend.market[i]:
            return CloseDecision(o.weights.copy(),'COMMITTED_OFFENSIVE_HOLD',1.,o.units.copy())

        allowed=(p.ready[i] & self.trend.entry[i] & (p.price[i]>p.ema20[i]) &
                 (p.momentum5[i]>0) & np.isfinite(score) & (score>0) & ~held & ~broken)
        ranked=sorted(np.flatnonzero(allowed),key=lambda j:(-score[j],self.market.symbols[j]))
        chosen=ranked[:vacancy]
        if not chosen or o.cash<=1e-8:
            return CloseDecision(o.weights.copy(),'COMMITTED_OFFENSIVE_HOLD',1.,o.units.copy())

        units=o.units.copy()
        spend=float(o.cash)/len(chosen)
        for j in chosen:
            if marks[j]>0: units[j]=spend/marks[j]
        weights=np.divide(units*marks,o.nav,out=np.zeros_like(units),where=o.nav>0)
        # Floating arithmetic may put a cash-only fill a few ulps above NAV.
        # Repair only the last newly requested unit amount; never rescale owned
        # inventory and never divide through zero marks of unrelated securities.
        total_value=float((units*marks).sum())
        if total_value>o.nav and total_value<=o.nav*(1.+1e-12):
            j=chosen[-1]
            if marks[j]<=0: raise AssertionError('chosen vacancy fill requires a valid close mark')
            units[j]=max(0.,units[j]-(total_value-o.nav)/marks[j])
            weights=np.divide(units*marks,o.nav,out=np.zeros_like(units),where=o.nav>0)
        self.trace.append({'kind':'COMMITTED_OFFENSIVE_EVENT','date':o.date,'session':int(i),
                           'action':'VACANCY_FILL','symbols':[self.market.symbols[j] for j in chosen],
                           'observed_cash':float(o.cash)})
        return CloseDecision(weights,'COMMITTED_OFFENSIVE_FILL',1.,units)

    def decide(self,o:CloseObservation):
        if not self.parameters.enabled: return self.parent.decide(o)
        return self._decision(o)

    def identity(self):
        root=Path(__file__).parent
        return {'name':'committed_offensive_core','parameters':asdict(self.parameters),
                'implementation_sha256':file_hash(Path(__file__)),
                'contract_sha256':file_hash(root/'committed_offensive_core_contract.json'),
                'parent_sha256':file_hash(root/'trend_book.py'),'data_sha256':self.market.fingerprint(),
                'status':'RESEARCH_NOT_ACCEPTED'}

__all__=['Owner','Parameters','grid','preserve_trace','verify_trace']

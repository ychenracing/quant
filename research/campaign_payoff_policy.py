"""Use settled control campaigns for priority without giving forecasts authority."""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from techquant.config import Config
from techquant.data import Market, file_hash
from techquant.engine import run
from techquant.evidence import source_identity
from research.observed_admission_completion import Owner as Parent, Parameters as ParentParameters
from research.quantity_obligation import SupportIntent, preserve_trace, verify_trace
from research.relative_rank import cohort_features, rank_order
from research.campaign_payoff import SettledCampaign, learn_campaign_payoff
from research.ledger_attribution import attribute


@dataclass(frozen=True)
class Parameters:
    learn_payoff: bool = True

    def __post_init__(self):
        if type(self.learn_payoff) is not bool:
            raise ValueError('learn_payoff must be a declared boolean')


def grid():
    return [Parameters(False),Parameters(True)]


def control_account(market,config,*,costs=1.,delay=1):
    return run(market,config,policy_factory=lambda m,c:Parent(m,ParentParameters(),config=c),
               cost_multiplier=costs,delay=delay)


def settled_samples(market,reference,config,*,costs=1.,delay=1):
    expected={'source':source_identity(),'config':asdict(config),
              'data_sha256':market.fingerprint(),'universe':list(market.symbols),
              'delay':delay,'cost_multiplier':costs,'benchmark':None,
              'policy':Parent(market,ParentParameters(),config=config).identity()}
    if any(reference.metadata.get(k)!=v for k,v in expected.items()):
        raise ValueError('shadow control source/configuration/data/policy/execution mismatch')
    _,episodes,reconciliation=attribute(market,reference)
    lookup={str(d.date()):i for i,d in enumerate(market.calendar)}
    symbols={s:j for j,s in enumerate(market.symbols)}
    records=[];censored=0
    for row in episodes.to_dict('records'):
        if row['exit']=='OPEN':
            censored+=1
            continue
        origin,entry,end=(lookup[str(row[key])] for key in ('entry_signal','entry','exit'))
        if not origin<entry<end or row['buy_notional']<=0:
            raise ValueError('settled label has invalid actual-fill chronology or funding')
        records.append(SettledCampaign(origin,end,symbols[row['symbol']],
                                        float(row['net_cash']/row['buy_notional'])))
    records.sort(key=lambda r:(r.settled,r.formation,r.symbol))
    content=json.dumps([asdict(r) for r in records],sort_keys=True,allow_nan=False,separators=(',',':'))
    meta={'reference_identity':reference.metadata,'samples':len(records),'censored_open':censored,
          'samples_sha256':hashlib.sha256(content.encode()).hexdigest(),'ledger':reconciliation,
          'limitation':'selected control campaigns, not unselected opportunity outcomes'}
    return records,meta


class PayoffSupport(SupportIntent):
    def _allocation_order(self,i,indices):
        original=super()._allocation_order(i,indices)
        selected=rank_order(self.features.score[i],self.payoffs[i],self.market.symbols,indices)
        if len(original)>1:
            self.priority_trace.append({'kind':'SETTLED_CAMPAIGN_PAYOFF_PRIORITY',
                'session':i,'date':str(self.market.calendar[i].date()),
                'original_indices':[int(j) for j in original],'selected_indices':selected,
                'changed':selected!=original,
                'predictions':[float(v) if np.isfinite(v) else None for v in self.payoffs[i]],
                'note':'priority only; the unchanged parent owns all funding and protection'})
        return selected


class Owner(Parent):
    def __init__(self,market:Market,parameters:Parameters,*,config:Config|None=None,
                 reference=None,costs=1.,delay=1):
        if type(parameters) is not Parameters:
            raise ValueError('only the declared settled-payoff pair is supported')
        super().__init__(market,ParentParameters(),config=config)
        self.payoff_parameters=parameters
        self.payoffs=None;self.fits=[];self.samples=[];self.label_meta=None
        if parameters.learn_payoff:
            original=self.inner;cfg=original.config
            if reference is None:
                reference=control_account(market,cfg,costs=costs,delay=delay)
            self.samples,self.label_meta=settled_samples(market,reference,cfg,costs=costs,delay=delay)
            inner=PayoffSupport(market,original.params,config=cfg)
            inner.features,inner.atr,inner.support,inner.ready=(original.features,original.atr,
                                                             original.support,original.ready)
            features=cohort_features(market.panel('close').to_numpy(),inner.ready,
                                     fast=cfg.fast,slow=cfg.slow)
            self.payoffs,self.fits=learn_campaign_payoff(features,self.samples,refit=cfg.rebalance)
            inner.payoffs=self.payoffs;inner.priority_trace=self.trace
            self.inner=inner
        elif reference is not None:
            raise ValueError('a disabled policy must not consume a shadow label stream')

    def identity(self):
        root=Path(__file__).parent
        return {'name':'settled_campaign_payoff','parameters':asdict(self.payoff_parameters),
            'implementation_sha256':file_hash(Path(__file__)),
            'primitive_sha256':file_hash(root/'campaign_payoff.py'),
            'contract_sha256':file_hash(root/'campaign_payoff_contract.json'),
            'parent_definition':super().identity(),'label_origin':self.label_meta,
            'issued_payoffs_sha256':None if self.payoffs is None else
                hashlib.sha256(self.payoffs.tobytes()).hexdigest(),
            'data_sha256':self.market.fingerprint(),'status':'RESEARCH_NOT_ACCEPTED'}

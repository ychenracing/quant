"""Fixed prequential relative ranking of the parent's eligible funded opportunities."""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import numpy as np
from techquant.config import Config
from techquant.data import Market, file_hash
from research.observed_admission_completion import Owner as Parent, Parameters as ParentParameters
from research.quantity_obligation import SupportIntent, preserve_trace, verify_trace
from research.relative_rank import learn_relative_rank, rank_order


@dataclass(frozen=True)
class Parameters:
    learn_relative: bool = True

    def __post_init__(self):
        if type(self.learn_relative) is not bool:
            raise ValueError('learn_relative must be a registered boolean')


def grid():
    return [Parameters(False), Parameters(True)]


class RelativeSupport(SupportIntent):
    def _allocation_order(self, i, indices):
        original=super()._allocation_order(i,indices)
        selected=rank_order(self.features.score[i],self.relative[i],self.market.symbols,indices)
        if len(original)>1:
            self.priority_trace.append({'kind':'RELATIVE_LEADERSHIP_PRIORITY',
                'session':i, 'date':str(self.market.calendar[i].date()),
                'original_indices':[int(j) for j in original],
                'selected_indices':selected, 'changed':selected!=original,
                'predictions':[float(x) if np.isfinite(x) else None for x in self.relative[i]],
                'note':'priority call, not a fill or counterfactual profit'})
        return selected


class Owner(Parent):
    def __init__(self, market:Market, parameters:Parameters, *, config:Config|None=None):
        if type(parameters) is not Parameters:
            raise ValueError('only the fixed relative-leadership pair is supported')
        super().__init__(market,ParentParameters(),config=config)
        self.relative_parameters=parameters
        self.relative=None;self.fits=[]
        if parameters.learn_relative:
            original=self.inner;cfg=original.config
            inner=RelativeSupport(market,original.params,config=cfg)
            inner.features,inner.atr,inner.support,inner.ready=(original.features,
                original.atr,original.support,original.ready)
            quoted=market.panel('close')
            active=(quoted.notna() & market.panel('volume').gt(0)).to_numpy()
            self.relative,self.fits=learn_relative_rank(quoted.to_numpy(),
                market.panel('open').to_numpy(),active,inner.ready,
                fast=cfg.fast,slow=cfg.slow,horizon=cfg.rebalance,refit=cfg.rebalance)
            inner.relative=self.relative;inner.priority_trace=self.trace
            self.inner=inner

    def identity(self):
        return {'name':'relative_leadership', 'parameters':asdict(self.relative_parameters),
            'implementation_sha256':file_hash(Path(__file__)),
            'primitive_sha256':file_hash(Path(__file__).with_name('relative_rank.py')),
            'contract_sha256':file_hash(Path(__file__).with_name('relative_leadership_contract.json')),
            'parent_definition':super().identity(),
            'issued_ranks_sha256':None if self.relative is None else
                hashlib.sha256(self.relative.tobytes()).hexdigest(),
            'data_sha256':self.market.fingerprint(), 'status':'RESEARCH_NOT_ACCEPTED'}

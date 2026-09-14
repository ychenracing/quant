"""Confirm restored trend separately from the unchanged current purchase trigger."""
from dataclasses import dataclass
from pathlib import Path
import sys
from techquant.data import file_hash
from research.support_budget import Owner as SupportOwner, Parameters as SupportParameters
from research.finite_study import Study as FiniteStudy


@dataclass(frozen=True)
class Parameters:
    pass


def grid():
    return [Parameters()]


class Owner(SupportOwner):
    def __init__(self,market,parameters):
        if not isinstance(parameters,Parameters):
            raise ValueError('readmission accepts only its fixed candidate')
        super().__init__(market,SupportParameters(.10,2))
        self.fast=market.panel('close').ffill().rolling(
            self.config.fast,min_periods=self.config.fast).mean().to_numpy()

    def _readmission_health(self,i,allowed):
        # Health can mature before a breakout. It never replaces today's full
        # entry trigger, actual-cash allocation or the account risk authority.
        return self.ready[i] & ~self.features.exit[i] & (self.features.close[i]>self.fast[i])

    def identity(self):
        return {'name':'trend_health_readmission',
                'implementation_sha256':file_hash(Path(__file__)),
                'contract_sha256':file_hash(Path(__file__).with_name('readmission_contract.json')),
                'parent_policy':super().identity(), 'status':'RESEARCH_NOT_ACCEPTED'}


class Study(FiniteStudy):
    def __init__(self,family='readmission',*,issued_evidence=None):
        if family!='readmission' or issued_evidence is not None:
            raise ValueError('this study accepts only the registered fixed readmission candidate')
        self.family,self.module,self.issued=family,sys.modules[__name__],None

    def identity(self):
        identity=super().identity()
        root=Path(__file__).parent
        for name in ('support_budget.py','support_budget_contract.json'):
            identity['dependencies'][name]=file_hash(root/name)
        return identity

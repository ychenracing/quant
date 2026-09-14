"""Measure the preregistered completion repair on four fixed existing controls."""
from dataclasses import dataclass
from pathlib import Path
import sys
from techquant.data import file_hash
from research import support_budget, funded_risk, recovery_memory
from research.finite_study import Study as FiniteStudy


@dataclass(frozen=True)
class Parameters:
    control: str = 'support'

    def __post_init__(self):
        if self.control not in ('price','support','funded','cushion'):
            raise ValueError('undeclared completion control')


def grid():
    return [Parameters(name) for name in ('price','support','funded','cushion')]


def control_owner(market, control):
    Parameters(control)
    if control == 'price':
        return recovery_memory.Owner(market,recovery_memory.Parameters(False))
    if control == 'support':
        return support_budget.Owner(market,support_budget.Parameters(.10,2))
    return funded_risk.Owner(market,funded_risk.Parameters(control=='cushion'))


class Owner:
    def __init__(self,market,parameters):
        self.parameters=parameters
        self.owner=control_owner(market,parameters.control)

    def decide(self,observation):
        return self.owner.decide(observation)

    def identity(self):
        return {'name':'protective_completion', 'control':self.parameters.control,
                'implementation_sha256':file_hash(Path(__file__)),
                'contract_sha256':file_hash(Path(__file__).with_name('completion_contract.json')),
                'parent_policy':self.owner.identity(), 'status':'RESEARCH_NOT_ACCEPTED'}


class Study(FiniteStudy):
    """Reuse the unchanged selector/replayer without editing the concurrent runner."""
    def __init__(self,family='completion',*,issued_evidence=None):
        if family!='completion' or issued_evidence is not None:
            raise ValueError('this correction study accepts only fixed non-forecast controls')
        self.family=family
        self.module=sys.modules[__name__]
        self.issued=None

    def identity(self):
        identity=super().identity()
        root=Path(__file__).parent
        for name in ('support_budget','funded_risk','recovery_memory','trend_book',
                     'coherent','pathwise','observed_trend','nonlinear'):
            for suffix in ('.py','_contract.json'):
                path=root/(name+suffix)
                identity['dependencies'][path.name]=file_hash(path)
        return identity

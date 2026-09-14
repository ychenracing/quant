"""One registered interaction of observed readiness and cash-funded completion.

Reuse the frozen components, not their returns. One actual account owns the
entire state transition; an early estimate does not relax protective authority.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
from techquant.data import Market, file_hash
from research.observed_readiness import Owner as Readiness, Parameters as ReadinessParameters
from research.admission_budget_completion import Owner as Completion, Parameters as CompletionParameters
from research.quantity_obligation import preserve_trace, verify_trace


@dataclass(frozen=True)
class Parameters:
    pass


def grid():
    return [Parameters()]


class Owner(Completion):
    def __init__(self, market: Market, parameters: Parameters):
        if type(parameters) is not Parameters:
            raise ValueError('observed completion requires its registered singleton')
        # Both components use the same corrected support parent. Initialize that
        # parent once through Readiness, then reuse Completion's unchanged decide.
        # No owner or state is replaced after the first observed account close.
        parent = Readiness(market, ReadinessParameters())
        self.market, self.parameters = parent.market, parent.parameters
        self.inner, self.trace = parent.inner, parent.trace
        self.completion_parameters = CompletionParameters()
        self.interaction_parameters = parameters
        self.readiness_origin = parent.identity()

    def identity(self):
        return {'name':'observed_admission_completion',
            'parameters':asdict(self.interaction_parameters),
            'implementation_sha256':file_hash(Path(__file__)),
            'contract_sha256':file_hash(Path(__file__).with_name('observed_admission_completion_contract.json')),
            'readiness_origin':self.readiness_origin,
            'completion_origin':super().identity(),
            'data_sha256':self.market.fingerprint(), 'status':'RESEARCH_NOT_ACCEPTED'}

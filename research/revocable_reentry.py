"""Keep weaker health permission current until actual capital is acquired."""
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
from techquant.data import Market, file_hash
from techquant.policy import CloseObservation
from research.funded_reentry import Owner as Parent, Parameters as ParentParameters
from research.quantity_obligation import preserve_trace, verify_trace


@dataclass(frozen=True)
class Parameters:
    pass


def grid():
    return [Parameters()]


class Owner(Parent):
    def __init__(self, market: Market, parameters: Parameters):
        if type(parameters) is not Parameters:
            raise ValueError('revocable reentry accepts only its registered singleton')
        super().__init__(market, ParentParameters())
        self.permission_parameters = parameters

    def decide(self, observation: CloseObservation):
        p = self.inner
        if observation.session <= p.last_session:
            raise ValueError('policy sessions must increase')
        held = observation.units > 1e-10
        opened = held & (p.previous_units <= 1e-10)
        # An actual acquisition closes the flat admission question. It does not
        # reset the new campaign's pending stop or any protective obligation.
        p.readmit[opened] = False
        decision = super().decide(observation)
        released = np.asarray(self.trace[-1]['funded_reentry']['released'], dtype=bool)
        provisional = released & ~held & (p.healthy < p.config.recovery)
        # The unchanged parent's stronger confirmation remains durable. Only a
        # weaker, still-unfilled permission must be earned again at the next close.
        p.readmit[provisional] = True
        self.trace[-1]['revocable_reentry'] = {
            'actual_open': opened.tolist(), 'provisional_veto_restored': provisional.tolist(),
            'readmission_veto_after': p.readmit.tolist()}
        return decision

    def identity(self):
        return {'name': 'revocable_prefill_trend_health',
                'parameters': asdict(self.permission_parameters),
                'implementation_sha256': file_hash(Path(__file__)),
                'contract_sha256': file_hash(Path(__file__).with_name('revocable_reentry_contract.json')),
                'parent_definition': super().identity(),
                'data_sha256': self.market.fingerprint(), 'status': 'RESEARCH_NOT_ACCEPTED'}

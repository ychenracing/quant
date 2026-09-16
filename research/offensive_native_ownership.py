"""Native causal ownership with account-level de-risking removed for alpha discovery."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from techquant.data import Market, file_hash
from techquant.policy import CloseObservation
from research.quantity_obligation import preserve_trace, verify_trace
from research.trend_book import Owner as ParentOwner, Parameters as ParentParameters


@dataclass(frozen=True)
class Parameters:
    enabled: bool

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('enabled must be boolean')


def grid():
    return [Parameters(False), Parameters(True)]


class OffensiveFullCap:
    """Account authority for the offensive experiment; not a production risk model."""

    def __init__(self):
        self.cap = 1.0

    def update(self, *args):
        self.cap = 1.0
        return 1.0, 'OFFENSIVE_FULL_CAP'


class Owner:
    def __init__(self, market: Market, parameters: Parameters):
        if type(parameters) is not Parameters:
            raise ValueError('offensive native ownership requires registered parameters')
        self.market = market
        self.parameters = parameters
        self.parent = ParentOwner(market, ParentParameters(2))
        if parameters.enabled:
            self.parent.risk = OffensiveFullCap()
        self.trace = []

    def decide(self, observation: CloseObservation):
        decision = self.parent.decide(observation)
        if self.parameters.enabled:
            reason = decision.reason
            action = (
                'NATIVE_REPLACEMENT' if 'LATCHED_LEADER_REPLACEMENT' in reason else
                'SECURITY_OR_INVENTORY_REDUCTION' if ('PROTECTIVE_INVENTORY_RETRY' in reason or 'TREND_EXIT' in reason) else
                'NATIVE_OWNERSHIP_DECISION'
            )
            self.trace.append({
                'kind': 'OFFENSIVE_NATIVE_OWNERSHIP_EVENT',
                'date': observation.date,
                'session': int(observation.session),
                'action': action,
                'reason': reason,
                'account_cap': 1.0,
            })
        return decision

    def identity(self):
        root = Path(__file__).parent
        return {
            'name': 'offensive_native_ownership',
            'parameters': asdict(self.parameters),
            'implementation_sha256': file_hash(Path(__file__)),
            'contract_sha256': file_hash(root / 'offensive_native_ownership_contract.json'),
            'parent_sha256': file_hash(root / 'trend_book.py'),
            'inventory_owner_sha256': file_hash(root / 'coherent.py'),
            'data_sha256': self.market.fingerprint(),
            'status': 'RESEARCH_NOT_ACCEPTED',
        }


__all__ = ['Owner', 'Parameters', 'grid', 'preserve_trace', 'verify_trace']

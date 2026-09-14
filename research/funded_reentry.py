"""Restore trend health without replacing the current funded purchase trigger."""
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
from techquant.data import Market, file_hash
from techquant.policy import CloseObservation
from research.observed_admission_completion import Owner as Parent, Parameters as ParentParameters
from research.quantity_obligation import preserve_trace, verify_trace


@dataclass(frozen=True)
class Parameters:
    pass


def grid():
    return [Parameters()]


class Owner(Parent):
    def __init__(self, market: Market, parameters: Parameters):
        if type(parameters) is not Parameters:
            raise ValueError('funded reentry accepts only its registered singleton')
        super().__init__(market, ParentParameters())
        self.reentry_parameters = parameters
        self.fast = market.panel('close').ffill().rolling(
            self.inner.config.fast, min_periods=self.inner.config.fast).mean().to_numpy()
        self.health_closes = np.zeros(len(market.symbols), dtype=int)

    def decide(self, observation: CloseObservation):
        p, i = self.inner, observation.session
        # The new state must obey the same actual observation clock as the owner.
        if i <= p.last_session:
            raise ValueError('policy sessions must increase')
        held = observation.units > 1e-10
        sold = (p.previous_units > 1e-10) & ~held
        health = p.ready[i] & ~p.features.exit[i] & (p.features.close[i] > self.fast[i])
        self.health_closes[sold] = 0
        self.health_closes = np.where(health, self.health_closes + 1, 0)
        before = p.readmit.copy()
        outstanding = np.any((p.exit_pending & held) |
                             (observation.units > p.reduction_ceiling + 1e-10))
        release = (before & ~held & ~sold & ~p.exit_pending &
                   ~np.isfinite(p.reduction_ceiling) &
                   (self.health_closes >= p.config.recovery))
        if outstanding:
            release[:] = False
        # Only this completed-campaign eligibility veto changes. In particular,
        # no protective exit/ceiling, stop, risk cap or NAV history is cleared.
        p.readmit[release] = False
        decision = super().decide(observation)
        # Parent traces are sparse events; append a distinct close record rather
        # than modifying an earlier event or assuming one exists on a flat day.
        self.trace.append({'date': observation.date, 'session': i,
            'actual_units': observation.units.tolist(), 'funded_reentry': {
            'health': health.tolist(), 'healthy_closes': self.health_closes.tolist(),
            'actual_full_liquidation': sold.tolist(), 'veto_before': before.tolist(),
            'released': release.tolist(), 'outstanding_protection': bool(outstanding)}})
        return decision

    def identity(self):
        return {'name': 'funded_trend_health_reentry',
                'parameters': asdict(self.reentry_parameters),
                'implementation_sha256': file_hash(Path(__file__)),
                'contract_sha256': file_hash(Path(__file__).with_name('funded_reentry_contract.json')),
                'parent_definition': super().identity(),
                'data_sha256': self.market.fingerprint(), 'status': 'RESEARCH_NOT_ACCEPTED'}

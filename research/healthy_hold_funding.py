"""One funded holding-continuity comparison, not a production setting."""
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
from techquant.config import Config
from techquant.data import Market, file_hash
from research.observed_admission_completion import Owner as Parent, Parameters as ParentParameters
from research.quantity_obligation import SupportIntent, preserve_trace, verify_trace


@dataclass(frozen=True)
class Parameters:
    continuous_held_funding: bool = False

    def __post_init__(self):
        if type(self.continuous_held_funding) is not bool:
            raise ValueError('only the two registered boolean candidates are supported')


def grid():
    return [Parameters(False), Parameters(True)]


class ContinuousFundingSupport(SupportIntent):
    def _funding_eligible(self, o, held, allowed, price, stops):
        original = super()._funding_eligible(o,held,allowed,price,stops)
        selected = allowed & (price > stops)
        additional = selected & ~original
        if additional.any():
            if np.any(additional & ~held):
                raise AssertionError('the holding treatment may not invent a new admission')
            self.funding_trace.append(dict(kind='HEALTHY_HELD_FUNDING_ELIGIBILITY',date=o.date,
                session=o.session,actual_cash=o.cash,nav=o.nav,actual_units=o.units.tolist(),
                original_eligible=original.tolist(),selected_eligible=selected.tolist(),
                prices=price.tolist(),stops=stops.tolist(),risk_cap=self.risk.cap))
        return selected


class Owner(Parent):
    def __init__(self, market: Market, parameters: Parameters, *, config: Config | None = None):
        if type(parameters) is not Parameters:
            raise ValueError('only registered healthy-holding parameters are supported')
        super().__init__(market,ParentParameters(),config=config)
        self.funding_parameters = parameters
        if parameters.continuous_held_funding:
            original = self.inner
            inner = ContinuousFundingSupport(market,original.params,config=original.config)
            # Share immutable causal estimates at construction; never swap a live owner.
            inner.features,inner.atr,inner.support,inner.ready = (
                original.features,original.atr,original.support,original.ready)
            inner.funding_trace = self.trace
            self.inner = inner

    def identity(self):
        return dict(name='healthy_hold_funding',parameters=asdict(self.funding_parameters),
            implementation_sha256=file_hash(Path(__file__)),
            contract_sha256=file_hash(Path(__file__).with_name('healthy_hold_funding_contract.json')),
            parent_definition=super().identity(),data_sha256=self.market.fingerprint(),
            status='RESEARCH_NOT_ACCEPTED')

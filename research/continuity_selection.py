"""Use completed price-path information only to order already-qualified funding."""
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
from techquant.config import Config
from techquant.data import Market, file_hash
from research.observed_admission_completion import Owner as Parent, Parameters as ParentParameters
from research.quantity_obligation import SupportIntent, preserve_trace, verify_trace


@dataclass(frozen=True)
class Parameters:
    continuous_information_priority: bool = False

    def __post_init__(self):
        if type(self.continuous_information_priority) is not bool:
            raise ValueError('only the registered boolean comparison is supported')


def grid():
    return [Parameters(False), Parameters(True)]


def information_continuity(market: Market, window: int) -> np.ndarray:
    if type(window) is not int or window < 1:
        raise ValueError('window must be a positive integer')
    quoted, volume = market.panel('close'), market.panel('volume')
    active = quoted.notna() & volume.gt(0)
    valid = active & active.shift(fill_value=False)
    change = quoted.diff()
    up = (change.gt(0).astype(float) + .5*change.eq(0)).where(valid)
    count = valid.rolling(window, min_periods=1).sum()
    result = up.rolling(window, min_periods=1).sum().div(count.where(count > 0)).fillna(1.)
    return result.to_numpy()


class ContinuousSupport(SupportIntent):
    def _allocation_order(self, i, indices):
        indices = list(indices)
        score = self.features.score[i]
        selected = sorted(indices, key=lambda j: (
            -score[j]*self.continuity[i,j], -score[j], self.market.symbols[j]))
        original = super()._allocation_order(i, indices)
        self.priority_trace.append(dict(kind='INFORMATION_CONTINUITY_PRIORITY',
            session=i, date=str(self.market.calendar[i].date()),
            symbols=[self.market.symbols[j] for j in indices],
            original_scores=[float(score[j]) for j in indices],
            continuity=[float(self.continuity[i,j]) for j in indices],
            original_order=[self.market.symbols[j] for j in original],
            selected_order=[self.market.symbols[j] for j in selected]))
        return selected


class Owner(Parent):
    def __init__(self, market: Market, parameters: Parameters, *, config: Config | None = None):
        if type(parameters) is not Parameters:
            raise ValueError('only registered continuity parameters are supported')
        super().__init__(market, ParentParameters(), config=config)
        self.continuity_parameters = parameters
        if parameters.continuous_information_priority:
            original = self.inner
            inner = ContinuousSupport(market, original.params, config=original.config)
            # Reuse immutable causal estimates before the first observation only.
            inner.features, inner.atr, inner.support, inner.ready = (
                original.features, original.atr, original.support, original.ready)
            inner.continuity = information_continuity(market, original.config.slow)
            inner.priority_trace = self.trace
            self.inner = inner

    def identity(self):
        return dict(name='continuity_selection', parameters=asdict(self.continuity_parameters),
            implementation_sha256=file_hash(Path(__file__)),
            contract_sha256=file_hash(Path(__file__).with_name('continuity_selection_contract.json')),
            parent_definition=super().identity(), data_sha256=self.market.fingerprint(),
            status='RESEARCH_NOT_ACCEPTED')

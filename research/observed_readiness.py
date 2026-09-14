"""Expand observed support estimates before their unchanged full window matures.

A registered research choice, not a correction to the original twenty-session
contract. All funding, risk and execution remain owned by the selected parent.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from techquant.data import Market, file_hash
from research.quantity_obligation import Owner as Parent, Parameters as ParentParameters
from research.quantity_obligation import preserve_trace, verify_trace


@dataclass(frozen=True)
class Parameters:
    pass


def grid():
    return [Parameters()]


class Owner(Parent):
    def __init__(self, market: Market, parameters: Parameters):
        if type(parameters) is not Parameters:
            raise ValueError('only the registered parameter-free candidate is supported')
        super().__init__(market, ParentParameters('support'))
        self.readiness_parameters = parameters
        minimum = self.inner.config.fast
        close = market.panel('close').ffill()
        high, low, previous = market.panel('high'), market.panel('low'), close.shift()
        # Match the parent's true-range formula. NaN previous close deliberately
        # leaves no initial range observation, even if today's high/low exist.
        tr = pd.DataFrame(np.maximum.reduce([(high-low).to_numpy(),
            (high-previous).abs().to_numpy(), (low-previous).abs().to_numpy()]),
            index=close.index, columns=close.columns)
        self.inner.atr = tr.rolling(20, min_periods=minimum).mean().to_numpy()
        self.inner.support = close.shift().rolling(20, min_periods=minimum).min().to_numpy()
        active = market.panel('close').notna() & market.panel('volume').gt(0)
        self.inner.ready = ((active & active.cumsum().ge(minimum)).to_numpy()
                            & np.isfinite(self.inner.atr) & np.isfinite(self.inner.support))

    def identity(self):
        return {'name': 'expanding_observed_support_readiness',
                'parameters': asdict(self.readiness_parameters),
                'window': 20, 'minimum': self.inner.config.fast,
                'implementation_sha256': file_hash(Path(__file__)),
                'contract_sha256': file_hash(Path(__file__).with_name('observed_readiness_contract.json')),
                'parent_definition': super().identity(),
                'data_sha256': self.market.fingerprint(), 'status': 'RESEARCH_NOT_ACCEPTED'}

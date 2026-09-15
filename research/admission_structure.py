"""Withhold fresh admissions over a declining support base; retain obligations.

This registered comparison changes new-entry selection, not stops or execution.
The condition is recomputed from completed observations without a fitted gate.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
from techquant.config import Config
from techquant.data import Market, file_hash
from research.observed_admission_completion import Owner as Parent, Parameters as ParentParameters
from research.quantity_obligation import SupportIntent, preserve_trace, verify_trace


@dataclass(frozen=True)
class Parameters:
    require_stable_support: bool = True

    def __post_init__(self):
        if type(self.require_stable_support) is not bool:
            raise ValueError('require_stable_support must be a registered boolean')


def grid():
    return [Parameters(False), Parameters(True)]


def fresh_admission_mask(current, previous, held):
    current, previous = (np.asarray(x, dtype=float) for x in (current, previous))
    held = np.asarray(held, dtype=bool)
    if current.ndim != 1 or current.shape != previous.shape or current.shape != held.shape:
        raise ValueError('support and actual holding vectors must have matching shapes')
    known = np.isfinite(current) & np.isfinite(previous)
    return held | ~known | (current >= previous)


def _finite_values(values):
    return [float(x) if np.isfinite(x) else None for x in values]


class StructuredSupport(SupportIntent):
    def _allocate(self, o, desired, price, initial, allowed, cap):
        i, lag = o.session, self.config.fast
        current = self.support[i]
        previous = self.support[i-lag] if i >= lag else np.full_like(current, np.nan)
        held = o.units > 1e-10
        admissible = fresh_admission_mask(current, previous, held)
        withheld = allowed & ~admissible
        if withheld.any():
            self.structure_trace.append({'kind':'FRESH_SUPPORT_STRUCTURE', 'date':o.date,
                'session':i, 'lag':lag, 'symbols':list(self.market.symbols),
                'actual_units':o.units.tolist(), 'original_allowed':allowed.tolist(),
                'current_support':_finite_values(current), 'lagged_support':_finite_values(previous),
                'history_known':(np.isfinite(current)&np.isfinite(previous)).tolist(),
                'withheld':withheld.tolist()})
        return super()._allocate(o, desired, price, initial, allowed & admissible, cap)


class Owner(Parent):
    def __init__(self, market: Market, parameters: Parameters, *, config: Config | None = None):
        if type(parameters) is not Parameters:
            raise ValueError('only the registered support-structure comparison is supported')
        super().__init__(market, ParentParameters(), config=config)
        self.structure_parameters = parameters
        if parameters.require_stable_support:
            original = self.inner
            inner = StructuredSupport(market, original.params, config=original.config)
            # Reuse the parent's causal estimates at construction, never live state.
            inner.features, inner.atr, inner.support, inner.ready = (
                original.features, original.atr, original.support, original.ready)
            inner.structure_trace = self.trace
            self.inner = inner

    def identity(self):
        return {'name':'admission_structure', 'parameters':asdict(self.structure_parameters),
            'implementation_sha256':file_hash(Path(__file__)),
            'contract_sha256':file_hash(Path(__file__).with_name('admission_structure_contract.json')),
            'parent_definition':super().identity(), 'data_sha256':self.market.fingerprint(),
            'status':'RESEARCH_NOT_ACCEPTED'}

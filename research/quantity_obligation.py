"""Keep an economic risk obligation distinct from an executable declaration.

No original policy, engine, threshold or old result is changed. The declaration
rounding is still conservative, but it cannot become fresh risk authority.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import hashlib
import json
import numpy as np
from techquant.config import Config
from techquant.data import Market, file_hash
from techquant.policy import CloseObservation
from research.support_budget import Owner as SupportOwner, Parameters as SupportParameters
from research.funded_risk import Owner as FundedOwner, Parameters as FundedParameters


@dataclass(frozen=True)
class Parameters:
    authority: str = 'support'

    def __post_init__(self):
        if type(self.authority) is not str or self.authority not in ('support','funded','cushion'):
            raise ValueError('authority must be one of the three fixed registered parents')


def grid():
    return [Parameters(name) for name in ('support','funded','cushion')]


class EconomicCeiling:
    def _round_reductions(self, current, desired, i):
        # The parent latches this result. Keep its original economic intent,
        # not a declaration whose legal rounding may overshoot that intent.
        return desired.copy()


class SupportIntent(EconomicCeiling, SupportOwner):
    pass


class FundedIntent(EconomicCeiling, FundedOwner):
    pass


class Owner:
    def __init__(self, market: Market, parameters: Parameters, *, config: Config | None = None):
        if config is not None and parameters.authority != "support":
            raise ValueError("explicit configuration is supported only by support authority")
        self.market, self.parameters = market, parameters
        self.inner = (SupportIntent(market, SupportParameters(.10, 2), config=config)
            if parameters.authority == 'support' else
            FundedIntent(market, FundedParameters(parameters.authority == 'cushion')))
        self.trace = []

    def decide(self, observation: CloseObservation):
        before = self.inner.reduction_ceiling.copy()
        decision = self.inner.decide(observation)
        economic = decision.unit_targets
        if economic is None:
            raise AssertionError('the fixed parent must issue economic-unit intentions')
        # Use the original declaration algorithm outside the state transition.
        # The rounded output can only further reduce an economic intention.
        declared = SupportOwner._round_reductions(self.inner, observation.units,
                                                  economic, observation.session)
        if np.any(declared > economic + 1e-10):
            raise AssertionError('a declaration cannot relax a risk obligation')
        price = np.nan_to_num(self.inner.features.close[observation.session], nan=0.)
        after = self.inner.reduction_ceiling
        affected = np.isfinite(before) | np.isfinite(after) | (declared < observation.units-1e-10)
        for j in np.flatnonzero(affected):
            self.trace.append({'date':observation.date, 'symbol':self.market.symbols[j],
                'actual_units':float(observation.units[j]),
                'prior_ceiling':float(before[j]) if np.isfinite(before[j]) else None,
                'economic_ceiling':float(after[j]) if np.isfinite(after[j]) else None,
                'economic_target':float(economic[j]), 'declared_target':float(declared[j]),
                'obligation_complete':bool(np.isfinite(before[j]) and observation.units[j]<=before[j]+1e-10),
                'exit_pending':bool(self.inner.exit_pending[j])})
        result = replace(decision, weights=declared*price/observation.nav,
            unit_targets=declared, reason=decision.reason+'|ECONOMIC_CEILING_SEPARATE_FROM_DECLARATION')
        result.validated_weights(len(self.market.symbols))
        result.validated_unit_targets(price, observation.nav)
        return result

    def identity(self):
        root = Path(__file__).parent
        return {'name':'separate_economic_obligation_and_declaration',
            'parameters':asdict(self.parameters), 'implementation_sha256':file_hash(Path(__file__)),
            'contract_sha256':file_hash(root/'quantity_obligation_contract.json'),
            'parent_policy':self.inner.identity(), 'data_sha256':self.market.fingerprint(),
            'status':'RESEARCH_NOT_ACCEPTED'}


def trace_digest(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False,
                                    separators=(',',':')).encode()).hexdigest()


def verify_trace(path: Path, identity: dict):
    if not path.is_file() or path.is_symlink():
        raise ValueError('missing or invalid intent trace')
    saved = json.loads(path.read_text())
    if (set(saved) != {'identity','trace','trace_sha256'}
            or saved['identity'] != identity or not isinstance(saved['trace'],list)
            or saved['trace_sha256'] != trace_digest(saved['trace'])):
        raise ValueError('intent trace identity or integrity mismatch')
    return saved


def preserve_trace(path: Path, identity: dict, rows: list):
    value = {'identity':identity,'trace':rows,'trace_sha256':trace_digest(rows)}
    if path.exists():
        if verify_trace(path,identity) != value:
            raise ValueError('existing intent trace has different decisions')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary = path.with_suffix('.partial')
        temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
        temporary.replace(path)

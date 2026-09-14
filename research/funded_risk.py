"""Protect actual held-book stop risk; retain original warning-driven entry freezes.

This is an explicitly registered alternative authority, not a reinterpretation
of the earlier support-budget evidence or a guarantee of maximum drawdown.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
from techquant.data import Market, file_hash
from techquant.policy import CloseObservation
from techquant.strategy import RiskState
from research.support_budget import Owner as SupportOwner, Parameters as SupportParameters


@dataclass(frozen=True)
class Parameters:
    use_cushion: bool = True

    def __post_init__(self):
        if type(self.use_cushion) is not bool:
            raise ValueError('use_cushion must be a declared boolean candidate')


def grid():
    return [Parameters(False), Parameters(True)]


class RiskAuthority:
    """Keep legacy warnings, a permanent actual-NAV peak and a strict risk ceiling."""
    def __init__(self, owner):
        self.owner = owner
        self.warnings = RiskState()
        self.global_peak = owner.config.initial_cash
        self.budget = 0.
        self.cap = 1.

    def update(self, i, features, history, config):
        o = self.owner.observation
        if o is None or o.session != i or history[-1] != o.nav:
            raise ValueError('funded risk authority requires the current actual observation')
        warning_cap, warning = self.warnings.update(i, features, history, config)
        self.global_peak = max(config.initial_cash, self.global_peak, o.nav)
        self.budget = self.owner.params.risk_budget*o.nav
        if self.owner.parameters.use_cushion:
            room = o.nav-(1-config.risk_drawdown)*self.global_peak
            self.budget = min(self.budget, max(.01*o.nav, room))
        price = np.nan_to_num(features.close[i], nan=0.)
        projected = np.minimum(o.units, self.owner.reduction_ceiling)
        projected[self.owner.exit_pending] = 0.
        # Match the parent's pending concentration reductions BEFORE translating
        # stop risk into a notional ceiling. Cutting a low-risk name changes the
        # remaining book's risk density; an earlier scalar ratio can undercut risk.
        weights = projected*price/o.nav
        admission_cap = max(config.single_cap, 1/len(projected))
        over = weights > (1. if len(projected) == 1 else .8)+1e-12
        projected[over] *= admission_cap/weights[over]
        if len(set(features.sectors)) > 1:
            for sector in sorted(set(features.sectors)):
                group = np.array([s == sector for s in features.sectors])
                weight = float((projected*price/o.nav)[group].sum())
                if weight > config.sector_cap+config.trade_band+1e-12:
                    projected[group] *= config.sector_cap/weight
        distance = np.maximum(price-self.owner.stop, .02*price)
        risk = float(projected@distance)
        exposure = float(projected@price/o.nav)
        # Projection determines outstanding protective requests, not spendable cash.
        self.cap = min(1., exposure*self.budget/risk) if risk > self.budget+1e-8 else 1.
        reason = (warning+'|FUNDED_STOP_RISK_BUDGET_'+format(self.budget/o.nav,'.8g')
                  +'|ENTRY_WARNING_CAP_'+format(warning_cap,'.8g'))
        return self.cap, reason


class Owner(SupportOwner):
    def __init__(self, market: Market, parameters: Parameters):
        super().__init__(market, SupportParameters(.10, 2))
        self.parameters = parameters
        self.observation = None
        self.risk = RiskAuthority(self)

    def identity(self):
        root = Path(__file__).parent
        return {'name':'funded_stop_risk', 'parameters':asdict(self.parameters),
                'implementation_sha256':file_hash(Path(__file__)),
                'contract_sha256':file_hash(root/'funded_risk_contract.json'),
                'parent_policy':super().identity(),
                'data_sha256':self.market.fingerprint(), 'status':'RESEARCH_NOT_ACCEPTED'}

    def decide(self, o: CloseObservation):
        self.observation = o
        return super().decide(o)

    def _exposure_ceiling(self, cap, cut):
        # A recovering cap still cannot add drift allowance to a live loss breach.
        return cap

    def _risk_limit(self, o, cap):
        # Legacy warnings govern additions even when an intact holding is retained.
        return self.risk.budget*self.risk.warnings.cap

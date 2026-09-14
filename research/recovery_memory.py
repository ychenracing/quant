"""Unresolved risk-event prices coordinate fresh entries with actual recovery.

This declared experiment extends only quant's independent price signal source.
The owner below cannot execute orders, alter cash, or reset account risk history.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import numpy as np
from techquant.data import Market, file_hash
from techquant.policy import CloseDecision, CloseObservation
from research.coherent import SignalInputs
from research.trend_book import Owner as PriceOwner, Parameters as PriceParameters


@dataclass(frozen=True)
class Parameters:
    remember_risk: bool = True

    def __post_init__(self):
        if type(self.remember_risk) is not bool:
            raise ValueError('remember_risk must be a boolean')


def grid():
    return [Parameters(False), Parameters(True)]


class Owner(PriceOwner):
    def __init__(self, market: Market, p: Parameters):
        self.memory_parameters = p
        super().__init__(market, PriceParameters(2))
        n = len(market.symbols)
        self.event_price = np.zeros(n)
        self.recovery_closes = np.zeros(n, dtype=int)
        self.observed_units = np.zeros(n)
        quoted = market.panel('close').to_numpy()
        self.fresh = (np.isfinite(quoted) & (quoted > 0)
                      & (market.panel('volume').to_numpy() > 0))

    def _signal_inputs(self, i: int) -> SignalInputs:
        signal = super()._signal_inputs(i)
        allowed = signal.allowed & ~signal.broken
        if self.memory_parameters.remember_risk:
            allowed &= (self.event_price == 0) | (self.recovery_closes >= 3)
        return replace(signal, allowed=allowed)

    def decide(self, o: CloseObservation) -> CloseDecision:
        i = o.session
        if i <= self.last_session:
            raise ValueError('policy sessions must increase')
        prices = self.price_signals.price[i]
        reasons = []
        if self.memory_parameters.remember_risk:
            anchored = self.event_price > 0
            healthy = (self.fresh[i] & (prices >= self.event_price)
                       & (prices > self.price_signals.ema10[i]))
            funded = (anchored & (self.recovery_closes >= 3) & healthy
                      & (o.units > self.observed_units + 1e-10))
            if funded.any():
                reasons.append('FUNDED_PRICE_RECOVERY:' + ','.join(
                    self.market.symbols[j] for j in np.flatnonzero(funded)))
                self.event_price[funded] = 0.
            self.recovery_closes = np.where(
                anchored & ~funded & healthy, self.recovery_closes + 1, 0)

        old_cap = self.risk.cap
        old_exit = self.exit_pending.copy()
        decision = super().decide(o)
        if self.memory_parameters.remember_risk:
            new_exit = self.exit_pending & ~old_exit
            cap_cut = self.risk.cap < old_cap - 1e-10
            affected = self.fresh[i] & (new_exit | cap_cut)
            if affected.any():
                self.event_price[affected] = np.maximum(
                    self.event_price[affected], prices[affected])
                self.recovery_closes[affected] = 0
                reasons.append('RISK_EVENT_PRICE_RECORDED:' + ','.join(
                    f'{self.market.symbols[j]}:{self.event_price[j]:.17g}'
                    for j in np.flatnonzero(affected)))
                # RiskState updates inside the shared owner's decision. A newly
                # issued cap cut must also gate today's proposed new inventory.
                # Keep every protective sale and the account's cap unchanged.
                units = decision.unit_targets.copy()
                units[affected] = np.minimum(units[affected], o.units[affected])
                values = np.zeros_like(units)
                np.multiply(units, prices, out=values, where=units > 0)
                decision = replace(decision, unit_targets=units, weights=values / o.nav)
        self.observed_units = o.units.copy()
        if reasons:
            decision = replace(decision, reason=decision.reason + '|' + '|'.join(reasons))
        return decision

    def identity(self):
        parent = super().identity()
        return {'name': 'risk_event_price_memory',
                'parameters': asdict(self.memory_parameters),
                'implementation_sha256': file_hash(Path(__file__)),
                'contract_sha256': file_hash(Path(__file__).with_name('recovery_memory_contract.json')),
                'parent_policy': parent,
                'data_sha256': self.market.fingerprint(),
                'status': 'RESEARCH_NOT_ACCEPTED'}

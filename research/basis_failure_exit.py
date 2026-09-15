"""Exit an actually funded campaign after confirmed owned-capital failure."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from techquant.config import Config
from techquant.data import Market, file_hash
from techquant.policy import CloseObservation
from research.coherent import SignalInputs
from research.trend_book import Owner as ParentOwner, Parameters as ParentParameters


@dataclass(frozen=True)
class Parameters:
    enabled: bool

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("basis failure exit requires a boolean switch")


def grid():
    return [Parameters(False), Parameters(True)]


class Owner(ParentOwner):
    def __init__(
        self,
        market: Market,
        parameters: Parameters,
        *,
        config: Config | None = None,
    ):
        if type(parameters) is not Parameters:
            raise ValueError("basis failure exit requires its registered parameters")
        # The price-book parent has no configurable economic surface beyond its
        # registered two-position book. Config is accepted only for the common
        # policy-factory signature and must remain the frozen default.
        if config is not None and config != Config():
            raise ValueError("basis failure exit requires the frozen Config")
        super().__init__(market, ParentParameters(2))
        self.failure_parameters = parameters
        self.parent_identity = super().identity()
        close = market.panel("close").ffill()
        self.ema40 = close.ewm(
            span=Config().slow, adjust=False, min_periods=Config().slow
        ).mean().to_numpy()
        self.open = market.panel("open").to_numpy()
        size = len(market.symbols)
        self.acquisition_basis = np.zeros(size)
        self.observed_units = np.zeros(size)
        self.failure_streak = np.zeros(size, dtype=int)
        self.basis_failure = np.zeros(size, dtype=bool)
        self.events: list[dict] = []

    def _observe_fills(self, observation: CloseObservation) -> None:
        units = observation.units
        added = units > self.observed_units + 1e-10
        opened = added & (self.observed_units <= 1e-10)
        prices = self.open[observation.session]
        if np.any(added & (~np.isfinite(prices) | (prices <= 0))):
            raise ValueError("observed additions require a valid current open")
        self.acquisition_basis[opened] = prices[opened]
        enlarged = added & ~opened
        delta = units - self.observed_units
        self.acquisition_basis[enlarged] = (
            self.observed_units[enlarged] * self.acquisition_basis[enlarged]
            + delta[enlarged] * prices[enlarged]
        ) / units[enlarged]
        flat = units <= 1e-10
        self.acquisition_basis[flat] = 0.
        self.failure_streak[flat] = 0
        self.observed_units = units.copy()

    def _signal_inputs(self, session: int) -> SignalInputs:
        observed = super()._signal_inputs(session)
        if not self.failure_parameters.enabled:
            self.basis_failure[:] = False
            return observed
        held = self.observed_units > 1e-10
        price = self.price_signals.price[session]
        failing = (
            held
            & (self.acquisition_basis > 0)
            & np.isfinite(price)
            & np.isfinite(self.ema40[session])
            & (price < self.acquisition_basis)
            & (price < self.ema40[session])
        )
        self.failure_streak = np.where(failing, self.failure_streak + 1, 0)
        self.basis_failure = held & (self.failure_streak >= 2)
        for index in np.flatnonzero(held & (self.failure_streak == 2)):
            self.events.append({
                "date": str(self.market.calendar[session].date()),
                "session": session,
                "symbol": self.market.symbols[index],
                "price": float(price[index]),
                "ema40": float(self.ema40[session, index]),
                "acquisition_basis": float(self.acquisition_basis[index]),
                "actual_units": float(self.observed_units[index]),
            })
        return replace(observed, broken=observed.broken | self.basis_failure)

    def decide(self, observation: CloseObservation):
        self._observe_fills(observation)
        decision = super().decide(observation)
        if self.failure_parameters.enabled and self.basis_failure.any():
            return replace(
                decision,
                reason=decision.reason + "|BASIS_FAILURE_EXIT",
            )
        return decision

    def identity(self):
        root = Path(__file__).parent
        return {
            "name": "actual_basis_campaign_failure_exit",
            "parameters": asdict(self.failure_parameters),
            "implementation_sha256": file_hash(Path(__file__)),
            "contract_sha256": file_hash(root / "basis_failure_exit_contract.json"),
            "parent_policy": self.parent_identity,
            "data_sha256": self.market.fingerprint(),
            "status": "RESEARCH_NOT_ACCEPTED",
        }


__all__ = ["Owner", "Parameters", "grid"]

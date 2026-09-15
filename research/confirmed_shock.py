"""Confirm a pure market shock before abandoning an otherwise intact book.

This research wrapper keeps the complete observed-admission parent.  It changes
only the treatment risk-state transition declared in the adjacent contract.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from techquant.config import Config
from techquant.data import Market, file_hash
from techquant.features import Features
from techquant.strategy import RiskState
from research.observed_admission_completion import (
    Owner as ParentOwner,
    Parameters as ParentParameters,
)
from research.quantity_obligation import preserve_trace, verify_trace


@dataclass(frozen=True)
class Parameters:
    confirm_market_shock: bool

    def __post_init__(self):
        if type(self.confirm_market_shock) is not bool:
            raise ValueError("confirm_market_shock must be boolean")


def grid():
    return [Parameters(False), Parameters(True)]


class ConfirmedMarketShockState(RiskState):
    """The existing risk state plus one non-numerical confirmation latch."""

    def __init__(
        self,
        cap: float = 0.,
        healthy: int = 0,
        episode_peak: float = 0.,
        *,
        trace: list | None = None,
    ):
        super().__init__(cap=cap, healthy=healthy, episode_peak=episode_peak)
        self.confirmation_pending = False
        self.trace = [] if trace is None else trace

    def _signals(
        self,
        i: int,
        features: Features,
        equity: list[float],
        config: Config,
    ) -> tuple[str | None, bool]:
        if not features.ready[i].any():
            return None, False
        shock = (
            features.market_return[i]
            < -max(.025, config.shock_z * features.market_vol[i])
            and features.breadth[i] < .5
        ) or features.shock_fraction[i] >= .6
        loss3 = (
            float(np.prod(1 + features.market_return[i - 2:i + 1]) - 1)
            if i >= 2
            else 0.
        )
        accumulated = (
            i >= 2
            and features.breadth[i] < .5
            and loss3
            < -max(
                .025,
                config.shock_z * features.market_vol[i - 2] * np.sqrt(3),
            )
        )
        market_reason = (
            "CROSS_SECTION_SHOCK"
            if shock
            else "ACCUMULATED_MARKET_SHOCK" if accumulated else None
        )
        peak = max(self.episode_peak, equity[-1])
        drawdown = 1 - equity[-1] / peak
        loss = equity[-1] / equity[-2] - 1 if len(equity) > 1 else 0.
        account_drawdown = drawdown >= config.risk_drawdown and loss < -.01
        return market_reason, account_drawdown

    def update(
        self,
        i: int,
        features: Features,
        equity: list[float],
        config: Config,
    ) -> tuple[float, str]:
        before_cap = self.cap
        before_pending = self.confirmation_pending
        market_reason, account_drawdown = self._signals(
            i, features, equity, config
        )
        _, reason = super().update(i, features, equity, config)

        # Account drawdown is independently computed because the production
        # branch order reports a simultaneous market shock first.  The contract
        # requires this already-existing forced zero to remain immediate.
        if account_drawdown:
            self.cap, self.healthy = 0., 0
            self.confirmation_pending = False
            reason = "PORTFOLIO_DRAWDOWN_SHOCK"
        elif market_reason is not None:
            if before_cap > .5 and not before_pending:
                self.cap, self.healthy = .5, 0
                self.confirmation_pending = True
                reason = "MARKET_SHOCK_CONFIRMATION_HALF_RISK"
            else:
                self.cap, self.healthy = 0., 0
                self.confirmation_pending = False
                reason = "CONFIRMED_" + market_reason
        else:
            self.confirmation_pending = False

        self.trace.append(
            {
                "kind": "CONFIRMED_MARKET_SHOCK_STATE",
                "session": i,
                "market_shock": market_reason,
                "account_drawdown_shock": account_drawdown,
                "before_cap": before_cap,
                "after_cap": self.cap,
                "before_pending": before_pending,
                "after_pending": self.confirmation_pending,
                "decision_reason": reason,
            }
        )
        return self.cap, reason


class Owner(ParentOwner):
    def __init__(
        self,
        market: Market,
        parameters: Parameters,
        *,
        config: Config | None = None,
    ):
        if type(parameters) is not Parameters:
            raise ValueError("confirmed shock requires its registered parameters")
        super().__init__(market, ParentParameters(), config=config)
        self.confirmed_shock_parameters = parameters
        self.parent_identity = super().identity()
        if parameters.confirm_market_shock:
            original = self.inner.risk
            self.inner.risk = ConfirmedMarketShockState(
                cap=original.cap,
                healthy=original.healthy,
                episode_peak=original.episode_peak,
                trace=self.trace,
            )

    def identity(self):
        return {
            "name": "confirmed_market_shock",
            "parameters": asdict(self.confirmed_shock_parameters),
            "implementation_sha256": file_hash(Path(__file__)),
            "contract_sha256": file_hash(
                Path(__file__).with_name("confirmed_shock_contract.json")
            ),
            "parent_policy": self.parent_identity,
            "data_sha256": self.market.fingerprint(),
            "status": "RESEARCH_NOT_ACCEPTED",
        }


__all__ = [
    "ConfirmedMarketShockState",
    "Owner",
    "Parameters",
    "grid",
    "preserve_trace",
    "verify_trace",
]

"""High-exposure causal trend ownership for the alpha-first offensive screen.

The control delegates exactly to the existing two-position price book.  The
fixed treatment keeps its causal entry/score and security-level break signals,
but removes account/market risk liquidation, volatility targeting and sector
budgeting.  It stays cash-only and long-only through the unchanged engine.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from techquant.config import Config
from techquant.data import Market, file_hash
from techquant.features import build_features
from techquant.policy import CloseDecision, CloseObservation
from research.observed_trend import Parameters as TrendParameters, signals
from research.trend_book import (
    Owner as ParentOwner,
    Parameters as ParentParameters,
    price_signals,
)
from research.quantity_obligation import preserve_trace, verify_trace


@dataclass(frozen=True)
class Parameters:
    enabled: bool

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be boolean")


def grid():
    return [Parameters(False), Parameters(True)]


class Owner:
    def __init__(self, market: Market, parameters: Parameters):
        if type(parameters) is not Parameters:
            raise ValueError("offensive core requires registered parameters")
        self.market = market
        self.parameters = parameters
        self.config = Config()
        self.parent = ParentOwner(market, ParentParameters(2))
        self.features = build_features(market, self.config)
        self.price_signals = price_signals(market)
        self.trend = signals(
            market, TrendParameters(trend_span=60, require_market_trend=True)
        )
        self.last_session = -1
        self.trace: list[dict] = []

    def _treatment(self, o: CloseObservation) -> CloseDecision:
        i = o.session
        if i <= self.last_session:
            raise ValueError("policy sessions must increase")
        self.last_session = i

        p = self.price_signals
        f = self.features
        held = o.units > 1e-10
        broken = self.trend.exit[i] | (p.ret1[i] <= -.08) | ~p.ready[i]
        survivors = held & ~broken

        score = f.score[i]
        allowed = (
            p.ready[i]
            & self.trend.entry[i]
            & (p.price[i] > p.ema20[i])
            & (p.momentum5[i] > 0)
            & np.isfinite(score)
            & (score > 0)
        )
        if not self.trend.market[i]:
            # Broad weakness gates only new commitments; it never liquidates an
            # intact funded security by itself.
            allowed[:] = False

        candidate = allowed | (survivors & np.isfinite(score) & (score > 0))
        ranked = sorted(
            np.flatnonzero(candidate),
            key=lambda j: (-score[j], self.market.symbols[j]),
        )
        selected = ranked[: self.config.max_positions]
        selected_set = set(selected)
        survivor_set = set(np.flatnonzero(survivors))
        forced = held & broken
        scheduled = (i % self.config.rebalance) == 0
        vacancy = len(survivor_set) < min(self.config.max_positions, len(ranked))
        review = scheduled or bool(forced.any()) or vacancy

        if not review:
            return CloseDecision(o.weights.copy(), "OFFENSIVE_RETAIN", 1.0)

        # If a scheduled review reaches the same intact members, do not harvest a
        # winner merely to rebalance percentages.  Actual drift remains owned.
        if not forced.any() and selected_set == survivor_set and survivor_set:
            self.trace.append({
                "kind": "OFFENSIVE_CORE_REVIEW",
                "date": o.date,
                "session": int(i),
                "scheduled": bool(scheduled),
                "selected": [self.market.symbols[j] for j in sorted(selected_set)],
                "forced_exits": [],
                "action": "RETAIN_INTACT_MEMBERSHIP",
            })
            return CloseDecision(o.weights.copy(), "OFFENSIVE_RETAIN", 1.0)

        weights = np.zeros(len(self.market.symbols), dtype=float)
        if selected:
            weights[selected] = 1.0 / len(selected)
        forced_symbols = [self.market.symbols[j] for j in np.flatnonzero(forced)]
        reason = "OFFENSIVE_SELECTION"
        if forced_symbols:
            reason += "|SECURITY_EXIT"
        self.trace.append({
            "kind": "OFFENSIVE_CORE_REVIEW",
            "date": o.date,
            "session": int(i),
            "scheduled": bool(scheduled),
            "selected": [self.market.symbols[j] for j in selected],
            "forced_exits": forced_symbols,
            "action": "FULL_EXPOSURE_SELECTION" if selected else "CASH_NO_ELIGIBLE_NAME",
        })
        return CloseDecision(weights, reason, 1.0)

    def decide(self, o: CloseObservation) -> CloseDecision:
        if not self.parameters.enabled:
            return self.parent.decide(o)
        return self._treatment(o)

    def identity(self):
        root = Path(__file__).parent
        return {
            "name": "offensive_causal_trend_core",
            "parameters": asdict(self.parameters),
            "implementation_sha256": file_hash(Path(__file__)),
            "contract_sha256": file_hash(root / "offensive_core_contract.json"),
            "parent_sha256": file_hash(root / "trend_book.py"),
            "data_sha256": self.market.fingerprint(),
            "status": "RESEARCH_NOT_ACCEPTED",
        }


__all__ = ["Owner", "Parameters", "grid", "preserve_trace", "verify_trace"]

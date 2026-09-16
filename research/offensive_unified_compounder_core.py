"""Standalone offensive ownership engine with a causally proven compounder state.

Enabled treatment does not delegate decisions to any historical research owner.
It reuses only immutable causal feature/signal builders and the public policy
interface.  Discovery campaigns keep the existing slow-trend self-exit until
actual funded price AND causal alpha both improve from admission.  Once proven,
a compounder ignores the generic slow-trend clock and yields capital only to
acute failure, missing data, or an executable stronger opportunity.
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
from research.trend_book import price_signals
from research.offensive_campaign_peak_authority import (
    Owner as ChampionOwner,
    Parameters as ChampionParameters,
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
            raise ValueError("unified compounder core requires registered parameters")
        self.market = market
        self.parameters = parameters
        self.control = ChampionOwner(market, ChampionParameters(True))
        self.config = Config()
        self.features = build_features(market, self.config)
        self.price_signals = price_signals(market)
        self.trend = signals(market, TrendParameters(trend_span=60, require_market_trend=True))
        self.execution_open = market.panel("open").to_numpy(dtype=float)
        n = len(market.symbols)
        self.pending_alpha_reference = np.full(n, np.nan)
        self.admission_alpha = np.full(n, np.nan)
        self.acquisition_basis = np.full(n, np.nan)
        self.peak_close = np.full(n, np.nan)
        self.peak_alpha = np.full(n, np.nan)
        self.proven = np.zeros(n, dtype=bool)
        self.observed_units = np.zeros(n)
        self.was_held = np.zeros(n, dtype=bool)
        self.retired = np.zeros(n, dtype=bool)
        self.retired_since = np.full(n, -1, dtype=int)
        self.invalidated = np.zeros(n, dtype=bool)
        self.invalidated_since = np.full(n, -1, dtype=int)
        self.failed_reference = np.full(n, np.nan)
        self.saw_nonentry = np.zeros(n, dtype=bool)
        self.recovered_without_fresh_epoch = np.zeros(n, dtype=bool)
        self.recovered_saw_nonentry = np.zeros(n, dtype=bool)
        self.last_session = -1
        self.trace: list[dict] = []

    def _event(self, observation: CloseObservation, action: str, mask: np.ndarray | None = None, **extra) -> None:
        if mask is not None and not np.any(mask):
            return
        row = {
            "kind": "UNIFIED_COMPOUNDER_CORE_EVENT",
            "date": observation.date,
            "session": int(observation.session),
            "action": action,
        }
        if mask is not None:
            row["symbols"] = [self.market.symbols[j] for j in np.flatnonzero(mask)]
        row.update(extra)
        self.trace.append(row)

    def _observe_inventory(self, observation: CloseObservation, score: np.ndarray) -> np.ndarray:
        units = np.asarray(observation.units, dtype=float)
        prior_units = self.observed_units.copy()
        prior_held = self.was_held.copy()
        held = units > 1e-10
        added = units > prior_units + 1e-10
        newly_held = added & (prior_units <= 1e-10)
        open_price = self.execution_open[observation.session]
        if np.any(added & (~np.isfinite(open_price) | (open_price <= 0))):
            raise ValueError("observed additions require a valid execution open")

        for j in np.flatnonzero(newly_held):
            if not np.isfinite(self.pending_alpha_reference[j]):
                raise ValueError("new funded campaign requires a pending causal admission alpha")
            self.admission_alpha[j] = self.pending_alpha_reference[j]
            self.pending_alpha_reference[j] = np.nan
            self.acquisition_basis[j] = open_price[j]
            self.proven[j] = False
            self.peak_close[j] = np.nan
            self.peak_alpha[j] = np.nan

        enlarged = added & ~newly_held
        if np.any(enlarged & ~np.isfinite(self.acquisition_basis)):
            raise ValueError("observed addition requires an existing acquisition basis")
        delta = units - prior_units
        self.acquisition_basis[enlarged] = (
            prior_units[enlarged] * self.acquisition_basis[enlarged]
            + delta[enlarged] * open_price[enlarged]
        ) / units[enlarged]

        actual_recovered = held & self.recovered_without_fresh_epoch
        if np.any(actual_recovered):
            self._event(observation, "RECOVERED_CAMPAIGN_ACTUAL_ESTABLISHMENT", actual_recovered)
            self.recovered_without_fresh_epoch[actual_recovered] = False
            self.recovered_saw_nonentry[actual_recovered] = False

        closed = ~held & prior_held
        self.admission_alpha[closed] = np.nan
        self.acquisition_basis[closed] = np.nan
        self.peak_close[closed] = np.nan
        self.peak_alpha[closed] = np.nan
        self.proven[closed] = False

        self.observed_units = units.copy()
        self.was_held = held.copy()
        return prior_held

    def _rearm_failed_campaigns(
        self,
        observation: CloseObservation,
        held: np.ndarray,
        prior_held: np.ndarray,
        score: np.ndarray,
        acute: np.ndarray,
    ) -> None:
        i = observation.session
        p = self.price_signals
        later_flat = self.invalidated & ~held & (i > self.invalidated_since)
        became_nonentry = later_flat & ~self.trend.entry[i]
        newly_nonentry = became_nonentry & ~self.saw_nonentry
        self.saw_nonentry[became_nonentry] = True
        self._event(observation, "FAILED_CAMPAIGN_FRESH_EPOCH_ARMED", newly_nonentry)

        ordinary = (
            p.ready[i]
            & self.trend.entry[i]
            & ~self.trend.exit[i]
            & (p.price[i] > p.ema20[i])
            & (p.momentum5[i] > 0)
            & np.isfinite(score)
            & (score > 0)
            & ~held
            & ~acute
            & ~self.retired
        )
        recovered = (
            later_flat
            & ordinary
            & np.isfinite(self.failed_reference)
            & (score > self.failed_reference)
            & ~prior_held
        )
        if np.any(recovered):
            self._event(observation, "REFERENCE_ALPHA_REARM", recovered)
            self.invalidated[recovered] = False
            self.invalidated_since[recovered] = -1
            self.failed_reference[recovered] = np.nan
            self.saw_nonentry[recovered] = False
            self.recovered_without_fresh_epoch[recovered] = True
            self.recovered_saw_nonentry[recovered] = False

        edge = (
            self.invalidated
            & ~held
            & (i > self.invalidated_since)
            & self.saw_nonentry
            & self.trend.entry[i]
            & ~prior_held
        )
        if np.any(edge):
            self._event(observation, "TREND_EDGE_REARM", edge)
            self.invalidated[edge] = False
            self.invalidated_since[edge] = -1
            self.failed_reference[edge] = np.nan
            self.saw_nonentry[edge] = False
            self.recovered_without_fresh_epoch[edge] = False
            self.recovered_saw_nonentry[edge] = False

        marked = self.recovered_without_fresh_epoch & ~held
        observed_false = marked & ~self.trend.entry[i]
        self.recovered_saw_nonentry[observed_false] = True
        fresh = marked & self.recovered_saw_nonentry & self.trend.entry[i]
        if np.any(fresh):
            self._event(observation, "RECOVERED_CAMPAIGN_FRESH_EPOCH", fresh)
            self.recovered_without_fresh_epoch[fresh] = False
            self.recovered_saw_nonentry[fresh] = False

    def _weights(self, units: np.ndarray, marks: np.ndarray, nav: float) -> np.ndarray:
        return np.divide(units * marks, nav, out=np.zeros_like(units), where=nav > 0)

    def _decision(self, observation: CloseObservation) -> CloseDecision:
        i = observation.session
        if i <= self.last_session:
            raise ValueError("policy sessions must increase")
        self.last_session = i
        p = self.price_signals
        score = self.features.score[i]
        close = p.price[i]
        held = observation.units > 1e-10
        prior_held = self._observe_inventory(observation, score)
        held = observation.units > 1e-10
        acute = p.ret1[i] <= -0.08
        unready = ~p.ready[i]

        newly_invalidated = held & acute & ~self.invalidated
        finite_reference = newly_invalidated & np.isfinite(self.admission_alpha)
        self.failed_reference[finite_reference] = self.admission_alpha[finite_reference]
        self.invalidated[newly_invalidated] = True
        self.invalidated_since[newly_invalidated] = i
        self.saw_nonentry[newly_invalidated] = False
        self._event(observation, "ACUTE_CAMPAIGN_INVALIDATION", newly_invalidated)
        self._rearm_failed_campaigns(observation, held, prior_held, score, acute)

        basic_break = self.trend.exit[i] | acute | unready
        released = self.retired & basic_break & (i > self.retired_since)
        if np.any(released):
            self._event(observation, "RETIREMENT_RELEASE", released)
            self.retired[released] = False
            self.retired_since[released] = -1

        # Update observed campaign quality before deciding whether the current
        # completed close has earned the persistent compounder state.
        finite_close = held & np.isfinite(close)
        prior_peak = self.peak_close.copy()
        self.peak_close[finite_close] = np.where(
            np.isfinite(self.peak_close[finite_close]),
            np.maximum(self.peak_close[finite_close], close[finite_close]),
            close[finite_close],
        )
        finite_score = held & np.isfinite(score)
        self.peak_alpha[finite_score] = np.where(
            np.isfinite(self.peak_alpha[finite_score]),
            np.maximum(self.peak_alpha[finite_score], score[finite_score]),
            score[finite_score],
        )
        newly_proven = (
            held
            & ~self.proven
            & np.isfinite(self.acquisition_basis)
            & np.isfinite(self.admission_alpha)
            & np.isfinite(close)
            & np.isfinite(score)
            & (close > self.acquisition_basis)
            & (score > self.admission_alpha)
        )
        self.proven[newly_proven] = True
        self._event(observation, "COMPOUNDER_PROVEN", newly_proven)

        slow_break = self.trend.exit[i] & ~self.proven
        proven_ignored = held & self.proven & self.trend.exit[i] & ~acute & ~unready
        self._event(observation, "PROVEN_SLOW_EXIT_IGNORED", proven_ignored)
        broken_held = held & (acute | unready | slow_break)
        marks = np.nan_to_num(close, nan=0.0)
        if np.any(broken_held):
            units = observation.units.copy()
            units[broken_held] = 0.0
            self._event(observation, "SECURITY_EXIT", broken_held)
            return CloseDecision(
                self._weights(units, marks, observation.nav),
                "UNIFIED_COMPOUNDER|SECURITY_EXIT",
                1.0,
                units,
            )

        retired_held = held & self.retired
        if np.any(retired_held):
            units = observation.units.copy()
            units[retired_held] = 0.0
            self._event(observation, "RETIRED_EXIT_RETRY", retired_held)
            return CloseDecision(
                self._weights(units, marks, observation.nav),
                "UNIFIED_COMPOUNDER|RETIRED_EXIT_RETRY",
                1.0,
                units,
            )

        survivors = np.flatnonzero(held)
        vacancy = max(0, self.config.max_positions - len(survivors))
        entry_broken = self.trend.exit[i] | acute | unready
        allowed = (
            p.ready[i]
            & self.trend.entry[i]
            & (p.price[i] > p.ema20[i])
            & (p.momentum5[i] > 0)
            & np.isfinite(score)
            & (score > 0)
            & ~held
            & ~entry_broken
            & ~self.retired
            & ~self.invalidated
        )

        if vacancy == 0 and self.trend.market[i]:
            decay = score - self.admission_alpha
            decayed = held & np.isfinite(decay) & (decay < 0)
            authority = allowed & ~self.recovered_without_fresh_epoch
            challengers = np.flatnonzero(authority)
            if np.any(decayed) and len(challengers):
                incumbent = min(
                    np.flatnonzero(decayed),
                    key=lambda j: (decay[j], self.market.symbols[j]),
                )
                challenger = min(
                    challengers,
                    key=lambda j: (-score[j], self.market.symbols[j]),
                )
                if score[challenger] > self.admission_alpha[incumbent]:
                    at_peak = (
                        np.isfinite(self.peak_close[incumbent])
                        and np.isfinite(close[incumbent])
                        and close[incumbent] >= self.peak_close[incumbent]
                    )
                    if at_peak:
                        self._event(
                            observation,
                            "FUNDED_PEAK_DISPLACEMENT_BLOCK",
                            None,
                            incumbent=self.market.symbols[incumbent],
                            challenger=self.market.symbols[challenger],
                            incumbent_score=float(score[incumbent]),
                            challenger_score=float(score[challenger]),
                        )
                    else:
                        units = observation.units.copy()
                        units[incumbent] = 0.0
                        self.retired[incumbent] = True
                        self.retired_since[incumbent] = i
                        self._event(
                            observation,
                            "ACTIVE_DISPLACEMENT",
                            None,
                            symbol=self.market.symbols[incumbent],
                            challenger=self.market.symbols[challenger],
                            admission_alpha=float(self.admission_alpha[incumbent]),
                            incumbent_score=float(score[incumbent]),
                            challenger_score=float(score[challenger]),
                            incumbent_proven=bool(self.proven[incumbent]),
                        )
                        return CloseDecision(
                            self._weights(units, marks, observation.nav),
                            "UNIFIED_COMPOUNDER|ACTIVE_DISPLACEMENT",
                            1.0,
                            units,
                        )

        if vacancy == 0 or not self.trend.market[i]:
            return CloseDecision(
                observation.weights.copy(),
                "UNIFIED_COMPOUNDER_HOLD",
                1.0,
                observation.units.copy(),
            )

        ranked = sorted(
            np.flatnonzero(allowed),
            key=lambda j: (-score[j], self.market.symbols[j]),
        )
        chosen = ranked[:vacancy]
        if not chosen or observation.cash <= 1e-8:
            return CloseDecision(
                observation.weights.copy(),
                "UNIFIED_COMPOUNDER_HOLD",
                1.0,
                observation.units.copy(),
            )

        units = observation.units.copy()
        spend = float(observation.cash) / len(chosen)
        for j in chosen:
            if marks[j] > 0:
                units[j] = spend / marks[j]
                self.pending_alpha_reference[j] = score[j]
        weights = self._weights(units, marks, observation.nav)
        total_value = float((units * marks).sum())
        if total_value > observation.nav and total_value <= observation.nav * (1.0 + 1e-12):
            j = chosen[-1]
            units[j] = max(0.0, units[j] - (total_value - observation.nav) / marks[j])
            weights = self._weights(units, marks, observation.nav)
        self._event(
            observation,
            "VACANCY_FILL",
            None,
            symbols=[self.market.symbols[j] for j in chosen],
            observed_cash=float(observation.cash),
        )
        return CloseDecision(weights, "UNIFIED_COMPOUNDER_FILL", 1.0, units)

    def decide(self, observation: CloseObservation):
        if not self.parameters.enabled:
            return self.control.decide(observation)
        return self._decision(observation)

    def identity(self):
        root = Path(__file__).parent
        return {
            "name": "offensive_unified_compounder_core",
            "parameters": asdict(self.parameters),
            "implementation_sha256": file_hash(Path(__file__)),
            "contract_sha256": file_hash(root / "offensive_unified_compounder_core_contract.json"),
            "control_sha256": file_hash(root / "offensive_campaign_peak_authority.py"),
            "data_sha256": self.market.fingerprint(),
            "decision_architecture": "standalone_enabled_treatment",
            "status": "RESEARCH_NOT_ACCEPTED",
        }


__all__ = ["Owner", "Parameters", "grid", "preserve_trace", "verify_trace"]

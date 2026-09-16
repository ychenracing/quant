"""High-exposure ownership with campaign-specific alpha references.

The treatment preserves intact inventory and uses only observed cash for actual
vacancies.  Campaign alpha references are captured causally at the close that
requests a vacancy fill and become owned references only after inventory is
observed.  Displacement behavior is added separately under tests.
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
from research.trend_book import Owner as ParentOwner, Parameters as ParentParameters, price_signals
from research.quantity_obligation import preserve_trace, verify_trace


@dataclass(frozen=True)
class Parameters:
    enabled: bool

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('enabled must be boolean')


def grid():
    return [Parameters(False), Parameters(True)]


class Owner:
    def __init__(self, market: Market, parameters: Parameters):
        if type(parameters) is not Parameters:
            raise ValueError('alpha-decay displacement requires registered parameters')
        self.market = market
        self.parameters = parameters
        self.config = Config()
        self.parent = ParentOwner(market, ParentParameters(2))
        self.features = build_features(market, self.config)
        self.price_signals = price_signals(market)
        self.trend = signals(market, TrendParameters(trend_span=60, require_market_trend=True))
        n = len(market.symbols)
        self.pending_alpha_reference = np.full(n, np.nan)
        self.owned_alpha_reference = np.full(n, np.nan)
        self.was_held = np.zeros(n, dtype=bool)
        self.retired = np.zeros(n, dtype=bool)
        self.retired_since = np.full(n, -1, dtype=int)
        self.last_session = -1
        self.trace = []

    def _observe_inventory(self, held: np.ndarray) -> None:
        newly_held = held & ~self.was_held
        for j in np.flatnonzero(newly_held):
            if np.isfinite(self.pending_alpha_reference[j]):
                self.owned_alpha_reference[j] = self.pending_alpha_reference[j]
                self.pending_alpha_reference[j] = np.nan
        closed = ~held & self.was_held
        self.owned_alpha_reference[closed] = np.nan
        self.was_held = held.copy()

    def _decision(self, o: CloseObservation) -> CloseDecision:
        i = o.session
        if i <= self.last_session:
            raise ValueError('policy sessions must increase')
        self.last_session = i
        p = self.price_signals
        score = self.features.score[i]
        held = o.units > 1e-10
        self._observe_inventory(held)
        broken = self.trend.exit[i] | (p.ret1[i] <= -.08) | ~p.ready[i]
        released = self.retired & broken & (i > self.retired_since)
        if released.any():
            symbols = [self.market.symbols[j] for j in np.flatnonzero(released)]
            self.retired[released] = False
            self.retired_since[released] = -1
            self.trace.append({
                'kind': 'ALPHA_DECAY_DISPLACEMENT_EVENT', 'date': o.date,
                'session': int(i), 'action': 'RETIREMENT_RELEASE', 'symbols': symbols,
            })
        broken_held = held & broken
        marks = np.nan_to_num(p.price[i], nan=0.)

        if broken_held.any():
            units = o.units.copy()
            units[broken_held] = 0.
            weights = np.divide(units * marks, o.nav, out=np.zeros_like(units), where=o.nav > 0)
            symbols = [self.market.symbols[j] for j in np.flatnonzero(broken_held)]
            self.trace.append({
                'kind': 'ALPHA_DECAY_DISPLACEMENT_EVENT', 'date': o.date, 'session': int(i),
                'action': 'SECURITY_EXIT', 'symbols': symbols,
            })
            return CloseDecision(weights, 'ALPHA_DECAY_DISPLACEMENT|SECURITY_EXIT', 1., units)

        retired_held = held & self.retired
        if retired_held.any():
            units = o.units.copy()
            units[retired_held] = 0.
            weights = np.divide(units * marks, o.nav, out=np.zeros_like(units), where=o.nav > 0)
            symbols = [self.market.symbols[j] for j in np.flatnonzero(retired_held)]
            self.trace.append({
                'kind': 'ALPHA_DECAY_DISPLACEMENT_EVENT', 'date': o.date,
                'session': int(i), 'action': 'RETIRED_EXIT_RETRY', 'symbols': symbols,
            })
            return CloseDecision(weights, 'ALPHA_DECAY_DISPLACEMENT|RETIRED_EXIT_RETRY', 1., units)

        survivors = np.flatnonzero(held)
        vacancy = max(0, self.config.max_positions - len(survivors))
        allowed = (
            p.ready[i] & self.trend.entry[i] & (p.price[i] > p.ema20[i]) &
            (p.momentum5[i] > 0) & np.isfinite(score) & (score > 0) &
            ~held & ~broken & ~self.retired
        )

        if vacancy == 0 and self.trend.market[i]:
            decay = score - self.owned_alpha_reference
            decayed = held & np.isfinite(decay) & (decay < 0)
            challengers = np.flatnonzero(allowed)
            if decayed.any() and len(challengers):
                incumbent = min(
                    np.flatnonzero(decayed),
                    key=lambda j: (decay[j], self.market.symbols[j]),
                )
                challenger = min(
                    challengers,
                    key=lambda j: (-score[j], self.market.symbols[j]),
                )
                if score[challenger] > self.owned_alpha_reference[incumbent]:
                    units = o.units.copy()
                    units[incumbent] = 0.
                    weights = np.divide(units * marks, o.nav, out=np.zeros_like(units), where=o.nav > 0)
                    self.retired[incumbent] = True
                    self.retired_since[incumbent] = int(i)
                    self.trace.append({
                        'kind': 'ALPHA_DECAY_DISPLACEMENT_EVENT', 'date': o.date,
                        'session': int(i), 'action': 'ALPHA_DECAY_DISPLACEMENT',
                        'symbol': self.market.symbols[incumbent],
                        'challenger': self.market.symbols[challenger],
                        'owned_alpha_reference': float(self.owned_alpha_reference[incumbent]),
                        'incumbent_score': float(score[incumbent]),
                        'challenger_score': float(score[challenger]),
                    })
                    return CloseDecision(
                        weights, 'ALPHA_DECAY_DISPLACEMENT', 1., units
                    )

        if vacancy == 0 or not self.trend.market[i]:
            return CloseDecision(o.weights.copy(), 'ALPHA_DECAY_DISPLACEMENT_HOLD', 1., o.units.copy())

        ranked = sorted(np.flatnonzero(allowed), key=lambda j: (-score[j], self.market.symbols[j]))
        chosen = ranked[:vacancy]
        if not chosen or o.cash <= 1e-8:
            return CloseDecision(o.weights.copy(), 'ALPHA_DECAY_DISPLACEMENT_HOLD', 1., o.units.copy())

        units = o.units.copy()
        spend = float(o.cash) / len(chosen)
        for j in chosen:
            if marks[j] > 0:
                units[j] = spend / marks[j]
                self.pending_alpha_reference[j] = score[j]
        weights = np.divide(units * marks, o.nav, out=np.zeros_like(units), where=o.nav > 0)
        total_value = float((units * marks).sum())
        if total_value > o.nav and total_value <= o.nav * (1. + 1e-12):
            j = chosen[-1]
            if marks[j] <= 0:
                raise AssertionError('chosen vacancy fill requires a valid close mark')
            units[j] = max(0., units[j] - (total_value - o.nav) / marks[j])
            weights = np.divide(units * marks, o.nav, out=np.zeros_like(units), where=o.nav > 0)
        self.trace.append({
            'kind': 'ALPHA_DECAY_DISPLACEMENT_EVENT', 'date': o.date, 'session': int(i),
            'action': 'VACANCY_FILL', 'symbols': [self.market.symbols[j] for j in chosen],
            'observed_cash': float(o.cash),
        })
        return CloseDecision(weights, 'ALPHA_DECAY_DISPLACEMENT_FILL', 1., units)

    def decide(self, observation: CloseObservation):
        if not self.parameters.enabled:
            return self.parent.decide(observation)
        return self._decision(observation)

    def identity(self):
        root = Path(__file__).parent
        return {
            'name': 'offensive_alpha_decay_displacement',
            'parameters': asdict(self.parameters),
            'implementation_sha256': file_hash(Path(__file__)),
            'contract_sha256': file_hash(root / 'offensive_alpha_decay_displacement_contract.json'),
            'parent_sha256': file_hash(root / 'trend_book.py'),
            'data_sha256': self.market.fingerprint(),
            'status': 'RESEARCH_NOT_ACCEPTED',
        }


__all__ = ['Owner', 'Parameters', 'grid', 'preserve_trace', 'verify_trace']

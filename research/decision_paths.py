"""Causal path selection across this project's independently written policies.

Shadow accounting is evidence, never executable inventory. The real engine
remains solely responsible for every fill and for funding the selected target.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import itertools
import numpy as np
from techquant.config import Config
from techquant.data import Market, file_hash
from techquant.engine import Result, run
from techquant.evidence import source_identity
from techquant.policy import CloseDecision, CloseObservation
from research.admission import Owner as Admission, Parameters as AdmissionParameters
from research.leadership import Owner as Leadership, Parameters as LeadershipParameters


@dataclass(frozen=True)
class Parameters:
    window: int = 60
    drawdown_penalty: float = 1.

    def __post_init__(self):
        if isinstance(self.window, bool) or not isinstance(self.window, int) or self.window < 1:
            raise ValueError('window must be a positive integer')
        if not np.isfinite(self.drawdown_penalty) or self.drawdown_penalty < 0:
            raise ValueError('drawdown penalty must be nonnegative and finite')


def grid():
    return [Parameters(*values) for values in itertools.product((20, 60, 120), (1., 2.))]


def build_shadows(market: Market) -> dict[str, Result]:
    return {
        'incumbent': run(market),
        'leadership': run(market, policy_factory=lambda m,c: Leadership(m, LeadershipParameters(60,20,4,False))),
        'admission': run(market, policy_factory=lambda m,c: Admission(m, AdmissionParameters('responsive',20,.6))),
        'buy_hold': run(market, benchmark='buy_hold')}


class Owner:
    def __init__(self, market: Market, parameters: Parameters, shadows: dict[str, Result]):
        self.market, self.parameters = market, parameters
        self.names = ('incumbent', 'leadership', 'admission', 'buy_hold')
        if set(shadows) != set(self.names):
            raise ValueError('shadow identity requires the four declared independent paths')
        source = source_identity()
        expected_policies = {
            'incumbent': None, 'buy_hold': None,
            'leadership': Leadership(market, LeadershipParameters(60,20,4,False)).identity(),
            'admission': Admission(market, AdmissionParameters('responsive',20,.6)).identity()}
        for name in self.names:
            result = shadows[name]
            meta = result.metadata
            if (meta['data_sha256'] != market.fingerprint() or meta['universe'] != list(market.symbols)
                    or meta['source'] != source or meta['config'] != asdict(Config())
                    or meta['delay'] != 1 or meta['cost_multiplier'] != 1.
                    or meta.get('policy') != expected_policies[name]
                    or meta['benchmark'] != ('buy_hold' if name == 'buy_hold' else None)
                    or not result.equity.index.equals(market.calendar)
                    or not result.targets.index.equals(market.calendar)
                    or not result.targets.columns.equals(market.panel('close').columns)):
                raise ValueError('shadow source/data/config/calendar identity mismatch')
        self.nav = np.column_stack([shadows[name].equity.nav.to_numpy() for name in self.names])
        self.initial_cash = Config().initial_cash
        self.targets = np.stack([shadows[name].targets.to_numpy() for name in self.names], axis=1)
        self.prices = market.panel('close').ffill().to_numpy()
        self.shadow_identity = {name: shadows[name].metadata for name in self.names}
        digest = hashlib.sha256()
        digest.update(np.asarray(self.nav, dtype='<f8').tobytes())
        digest.update(np.asarray(self.targets, dtype='<f8').tobytes())
        self.shadow_sha256 = digest.hexdigest()
        self.selected = -1
        self.last_change = -10**9
        self.last_session = -1
        self.observations = []

    def decide(self, o: CloseObservation) -> CloseDecision:
        i = o.session
        if i <= self.last_session:
            raise ValueError('decision sessions must increase')
        self.last_session = i
        first = max(0, i - self.parameters.window + 1)
        predecessor = self.nav[first-1] if first else np.full(len(self.names), self.initial_cash)
        # Slice at the observed close BEFORE deriving scores or trailing peaks.
        history = self.nav[first:i+1]
        peak = np.maximum(history.max(axis=0), predecessor)
        scores = np.log(history[-1] / predecessor) - self.parameters.drawdown_penalty * (1 - history[-1] / peak)
        best = int(np.argmax(scores)) if scores.max() > 0 else -1
        previous = self.selected
        current_score = float(scores[previous]) if previous >= 0 else 0.
        if previous >= 0 and current_score <= 0:
            self.selected = best
        elif previous < 0:
            self.selected = best
        elif (best >= 0 and best != previous and i - self.last_change >= 5
              and scores[best] > current_score + .02):
            self.selected = best
        if self.selected != previous:
            self.last_change = i
        weights = self.targets[i, self.selected].copy() if self.selected >= 0 else np.zeros(len(o.units))
        units = np.zeros_like(weights)
        prices = self.prices[i]
        if ((weights > 0) & (~np.isfinite(prices) | (prices <= 0))).any():
            raise ValueError('selected positive allocation has no observed close')
        np.divide(weights * o.nav, prices, out=units, where=weights > 0)
        name = self.names[self.selected] if self.selected >= 0 else 'cash'
        self.observations.append(dict(date=o.date, path=name, scores=dict(zip(self.names,map(float,scores))),
                                      actual_nav=o.nav, actual_cash=o.cash))
        return CloseDecision(weights, 'OBSERVED_DECISION_PATH_' + name.upper(), unit_targets=units)

    def identity(self):
        root = Path(__file__).parent
        return dict(name='causal_independent_decision_paths', parameters=asdict(self.parameters),
            research_files={p:file_hash(root/p) for p in ('decision_paths.py','admission.py','leadership.py')},
            data_sha256=self.market.fingerprint(), shadow_sources=self.shadow_identity,
            shadow_arrays_sha256=self.shadow_sha256, status='RESEARCH_NOT_ACCEPTED')

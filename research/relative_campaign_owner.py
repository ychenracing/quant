"""Benchmark-relative campaign lifecycle on actual funded inventory.

Initial discovery, ranking, funding and execution coordination are the unchanged
two-position observed price book.  Once an actually funded campaign reaches its
fixed review, this module becomes the complete security-level continuation and
exit authority for that campaign.  It consumes no reference-strategy source.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from techquant.data import Market, file_hash
from techquant.policy import CloseObservation
from research.coherent import SignalInputs
from research.trend_book import Owner as ParentOwner, Parameters as ParentParameters


FAST_AGE = 4
FULL_AGE = 9


@dataclass(frozen=True)
class Parameters:
    enabled: bool

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be boolean")


def grid():
    return [Parameters(False), Parameters(True)]


class Owner(ParentOwner):
    def __init__(
        self,
        market: Market,
        parameters: Parameters,
        *,
        benchmark_open,
        benchmark_close,
        benchmark_identity: str,
    ):
        if type(parameters) is not Parameters:
            raise ValueError("relative campaign owner requires registered parameters")
        opened = np.asarray(benchmark_open, dtype=float)
        closed = np.asarray(benchmark_close, dtype=float)
        if opened.shape != (len(market.calendar),) or closed.shape != opened.shape:
            raise ValueError("benchmark must align exactly to the market calendar")
        if not np.isfinite(opened).all() or not np.isfinite(closed).all():
            raise ValueError("benchmark must be finite")
        if (opened <= 0).any() or (closed <= 0).any():
            raise ValueError("benchmark prices must be positive")
        if not isinstance(benchmark_identity, str) or not benchmark_identity:
            raise ValueError("benchmark identity is required")
        self.campaign_parameters = parameters
        self.benchmark_open = opened.copy()
        self.benchmark_close = closed.copy()
        self.benchmark_identity = benchmark_identity
        super().__init__(market, ParentParameters(2))
        self.parent_identity = super().identity()
        count = len(market.symbols)
        self.open = market.panel("open").to_numpy()
        self.observed_units = np.zeros(count)
        self.campaign_age = np.full(count, -1, dtype=int)
        self.campaign_entry_session = np.full(count, -1, dtype=int)
        self.campaign_entry_open = np.zeros(count)
        self.campaign_benchmark_open = np.zeros(count)
        self.fast_relative_positive = np.zeros(count, dtype=bool)
        self.quality_active = np.zeros(count, dtype=bool)
        self.trace: list[dict] = []

    def _observe_inventory(self, observation: CloseObservation) -> None:
        held = observation.units > 1e-10
        previous = self.observed_units > 1e-10
        opened = held & ~previous
        continued = held & previous
        prices = self.open[observation.session]
        if np.any(opened & (~np.isfinite(prices) | (prices <= 0))):
            raise ValueError("actual funded campaign requires a valid opening price")
        self.campaign_age[continued] += 1
        self.campaign_age[opened] = 0
        self.campaign_entry_session[opened] = observation.session
        self.campaign_entry_open[opened] = prices[opened]
        self.campaign_benchmark_open[opened] = self.benchmark_open[observation.session]
        self.fast_relative_positive[opened] = False
        self.quality_active[opened] = False

        fast = held & (self.campaign_age == FAST_AGE)
        current = self.price_signals.price[observation.session]
        valid = (
            fast
            & np.isfinite(current)
            & (current > 0)
            & (self.campaign_entry_open > 0)
            & (self.campaign_benchmark_open > 0)
        )
        excess = np.full(len(held), -np.inf)
        excess[valid] = (
            np.log(current[valid] / self.campaign_entry_open[valid])
            - np.log(
                self.benchmark_close[observation.session]
                / self.campaign_benchmark_open[valid]
            )
        )
        self.fast_relative_positive[fast] = excess[fast] > 0

        closed = ~held
        self.campaign_age[closed] = -1
        self.campaign_entry_session[closed] = -1
        self.campaign_entry_open[closed] = 0.0
        self.campaign_benchmark_open[closed] = 0.0
        self.fast_relative_positive[closed] = False
        self.quality_active[closed] = False
        self.observed_units = observation.units.copy()

    def _campaign_quality(self, session: int) -> tuple[np.ndarray, np.ndarray]:
        held = self.observed_units > 1e-10
        mature = held & (self.campaign_age >= FULL_AGE)
        current = self.price_signals.price[session]
        valid = (
            mature
            & np.isfinite(current)
            & (current > 0)
            & (self.campaign_entry_open > 0)
            & (self.campaign_benchmark_open > 0)
        )
        absolute = np.zeros(len(held), dtype=bool)
        relative = np.zeros(len(held), dtype=bool)
        absolute[valid] = current[valid] > self.campaign_entry_open[valid]
        relative[valid] = (
            np.log(current[valid] / self.campaign_entry_open[valid])
            - np.log(
                self.benchmark_close[session]
                / self.campaign_benchmark_open[valid]
            )
        ) > 0
        quality = mature & self.fast_relative_positive & absolute & relative
        return mature, quality

    def _signal_inputs(self, session: int) -> SignalInputs:
        observed = super()._signal_inputs(session)
        if not self.campaign_parameters.enabled:
            return observed
        mature, quality = self._campaign_quality(session)
        hard = (self.price_signals.ret1[session] <= -0.08) | ~observed.ready
        broken = observed.broken.copy()
        broken[mature] = hard[mature] | ~quality[mature]

        entered = quality & ~self.quality_active
        lost = mature & ~quality & self.quality_active
        for index in np.flatnonzero(entered | lost | (mature & (self.campaign_age == FULL_AGE))):
            self.trace.append({
                "date": str(self.market.calendar[session].date()),
                "session": int(session),
                "symbol": self.market.symbols[index],
                "campaign_entry_session": int(self.campaign_entry_session[index]),
                "campaign_age": int(self.campaign_age[index]),
                "kind": (
                    "QUALITY_ACTIVATED" if quality[index]
                    else "QUALITY_REVOKED" if lost[index]
                    else "QUALITY_REJECTED"
                ),
            })
        self.quality_active[mature] = quality[mature]
        return SignalInputs(
            observed.price,
            observed.ready,
            observed.healthy,
            broken,
            observed.score,
            observed.allowed,
            observed.capacity,
        )

    def decide(self, observation: CloseObservation):
        self._observe_inventory(observation)
        return super().decide(observation)

    def identity(self):
        root = Path(__file__).parent
        return {
            "name": "benchmark_relative_campaign_owner",
            "parameters": asdict(self.campaign_parameters),
            "implementation_sha256": file_hash(Path(__file__)),
            "contract_sha256": file_hash(root / "relative_campaign_owner_contract.json"),
            "benchmark_identity": self.benchmark_identity,
            "parent": self.parent_identity,
            "data_sha256": self.market.fingerprint(),
            "status": "RESEARCH_NOT_ACCEPTED",
        }

"""Rare close-only campaign launch that opens otherwise idle cash capacity."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from techquant.data import Market, file_hash
from research.coherent import SignalInputs
from research.trend_book import Owner as ParentOwner, Parameters as ParentParameters


@dataclass(frozen=True)
class Parameters:
    enabled: bool

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be boolean")


def grid():
    return [Parameters(False), Parameters(True)]


def select_explosive(
    ready,
    return20,
    fresh_high,
    volume_confirmed,
    sectors,
    symbols,
):
    ready = np.asarray(ready, dtype=bool)
    return20 = np.asarray(return20, dtype=float)
    fresh_high = np.asarray(fresh_high, dtype=bool)
    volume_confirmed = np.asarray(volume_confirmed, dtype=bool)
    sectors = np.asarray(tuple(sectors), dtype=object)
    symbols = tuple(symbols)
    if not (
        return20.shape == ready.shape == fresh_high.shape == volume_confirmed.shape
        and len(sectors) == len(symbols) == len(ready)
    ):
        raise ValueError("explosive inputs must describe one exact universe")
    candidates = np.flatnonzero(ready & np.isfinite(return20))
    selected = np.zeros(len(ready), dtype=bool)
    if not len(candidates):
        return selected
    count = max(1, int(np.ceil(len(candidates) / 10)))
    ranked = sorted(candidates, key=lambda j: (-return20[j], symbols[j]))
    selected[ranked[:count]] = True
    selected &= (return20 > 0) & fresh_high & volume_confirmed
    confirmed = np.zeros(len(ready), dtype=bool)
    for j in np.flatnonzero(selected):
        peers = ready & (sectors == sectors[j])
        peers[j] = False
        confirmed[j] = bool(np.any(peers & (return20 > 0)))
    return selected & confirmed


def build_explosive(market: Market):
    quoted = market.panel("close")
    volume = market.panel("volume")
    active = quoted.notna() & volume.gt(0)
    price = quoted.ffill()
    ready = active & active.cumsum().ge(60)
    return20 = price.pct_change(20, fill_method=None)
    prior_high = price.shift().rolling(20, min_periods=20).max()
    prior_volume = volume.shift().rolling(20, min_periods=20).mean()
    fresh_high = price.ge(prior_high)
    volume_confirmed = volume.gt(prior_volume)
    sectors = tuple(market.sectors.get(symbol, "unknown") for symbol in market.symbols)
    output = np.zeros(price.shape, dtype=bool)
    for i in range(len(price)):
        output[i] = select_explosive(
            ready.iloc[i].to_numpy(),
            return20.iloc[i].to_numpy(),
            fresh_high.iloc[i].to_numpy(),
            volume_confirmed.iloc[i].to_numpy(),
            sectors,
            market.symbols,
        )
    return output


def apply_accelerator(observed: SignalInputs, explosive):
    explosive = np.asarray(explosive, dtype=bool)
    if explosive.shape != observed.allowed.shape:
        raise ValueError("explosive mask has a different universe")
    if not explosive.any():
        return observed
    return replace(
        observed,
        allowed=observed.allowed | explosive,
        capacity=max(4, observed.capacity),
    )


class Owner(ParentOwner):
    def __init__(self, market: Market, parameters: Parameters):
        if type(parameters) is not Parameters:
            raise ValueError("accelerator requires registered parameters")
        self.accelerator_parameters = parameters
        super().__init__(market, ParentParameters(2))
        self.explosive = build_explosive(market)
        self.trace = []

    def _signal_inputs(self, i: int) -> SignalInputs:
        observed = super()._signal_inputs(i)
        if not self.accelerator_parameters.enabled:
            return observed
        accelerated = apply_accelerator(observed, self.explosive[i])
        if accelerated is not observed:
            self.trace.append({
                "kind": "EXPLOSIVE_CAMPAIGN_ACCELERATOR",
                "session": i,
                "date": str(self.market.calendar[i].date()),
                "symbols": [
                    self.market.symbols[j]
                    for j in np.flatnonzero(self.explosive[i])
                ],
                "newly_allowed": [
                    self.market.symbols[j]
                    for j in np.flatnonzero(self.explosive[i] & ~observed.allowed)
                ],
                "capacity_before": observed.capacity,
                "capacity_after": accelerated.capacity,
            })
        return accelerated

    def identity(self):
        root = Path(__file__).parent
        return {
            "name": "explosive_campaign_accelerator",
            "parameters": asdict(self.accelerator_parameters),
            "implementation_sha256": file_hash(Path(__file__)),
            "contract_sha256": file_hash(
                root / "explosive_campaign_accelerator_contract.json"
            ),
            "parent_policy": super().identity(),
            "data_sha256": self.market.fingerprint(),
            "reference_runtime_inputs": False,
            "status": "RESEARCH_NOT_ACCEPTED",
        }


__all__ = [
    "Owner",
    "Parameters",
    "apply_accelerator",
    "build_explosive",
    "grid",
    "select_explosive",
    "ParentOwner",
    "ParentParameters",
]

"""One-slot offensive portfolio using the current campaign-peak lifecycle.

The enabled treatment changes exactly one economic architecture choice before
any decision: the deep offensive owner's capacity is one funded campaign rather
than two.  Discovery, entry, campaign references, rearm, campaign-peak blocking,
displacement and execution remain the current champion implementation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

from techquant.data import Market, file_hash
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
            raise ValueError("single dominant campaign requires registered parameters")
        self.market = market
        self.parameters = parameters
        self.parent = ChampionOwner(market, ChampionParameters(True))
        if parameters.enabled:
            deep = self.parent.parent.base
            deep.config = replace(deep.config, max_positions=1)

    @property
    def trace(self):
        return self.parent.trace

    def decide(self, observation):
        return self.parent.decide(observation)

    def identity(self):
        root = Path(__file__).parent
        return {
            "name": "offensive_single_dominant_campaign",
            "parameters": asdict(self.parameters),
            "implementation_sha256": file_hash(Path(__file__)),
            "contract_sha256": file_hash(root / "offensive_single_dominant_campaign_contract.json"),
            "control_sha256": file_hash(root / "offensive_campaign_peak_authority.py"),
            "data_sha256": self.market.fingerprint(),
            "offensive_max_positions": 1 if self.parameters.enabled else 2,
            "status": "RESEARCH_NOT_ACCEPTED",
        }


__all__ = ["Owner", "Parameters", "grid", "preserve_trace", "verify_trace"]

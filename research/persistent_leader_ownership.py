"""Market-gated entry with security-owned exits for persistent leaders."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from techquant.data import Market, file_hash
from techquant.strategy import RiskState
from research.trend_book import Owner as ParentOwner, Parameters as ParentParameters


@dataclass(frozen=True)
class Parameters:
    enabled: bool

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be boolean")


def grid():
    return [Parameters(False), Parameters(True)]


class EntryOnlyRisk:
    """Observe parent risk alarms without giving them liquidation authority.

    The parent signal source still gates every new unit on the causal broad
    market trend. Already funded units remain owned by the security-specific
    broken latch, actual inventory and execution constraints.
    """

    def __init__(self, trace):
        self.cap = 0.0
        self.healthy = 0
        self.episode_peak = 0.0
        self.shadow = RiskState()
        self.trace = trace

    def update(self, i, features, equity, config):
        shadow_cap, shadow_reason = self.shadow.update(i, features, equity, config)
        self.episode_peak = max(self.episode_peak, equity[-1])
        if not features.ready[i].any():
            self.cap, self.healthy = 0.0, 0
            reason = "WARMUP_OR_NO_FRESH_QUOTES"
        else:
            self.cap, self.healthy = 1.0, 0
            reason = "SECURITY_OWNED_EXIT"
        if shadow_cap < 1.0 or shadow_reason != "TREND_OPEN":
            self.trace.append({
                "kind": "PERSISTENT_LEADER_RISK_OBSERVATION",
                "session": int(i),
                "owner_cap": self.cap,
                "owner_reason": reason,
                "shadow_cap": float(shadow_cap),
                "shadow_reason": shadow_reason,
            })
        return self.cap, reason


class Owner(ParentOwner):
    def __init__(self, market: Market, parameters: Parameters):
        if type(parameters) is not Parameters:
            raise ValueError("persistent leader ownership requires registered parameters")
        self.ownership_parameters = parameters
        self.trace = []
        super().__init__(market, ParentParameters(2))
        if parameters.enabled:
            self.risk = EntryOnlyRisk(self.trace)

    def identity(self):
        root = Path(__file__).parent
        return {
            "name": "persistent_leader_ownership",
            "parameters": asdict(self.ownership_parameters),
            "implementation_sha256": file_hash(Path(__file__)),
            "contract_sha256": file_hash(
                root / "persistent_leader_ownership_contract.json"
            ),
            "parent_policy": super().identity(),
            "data_sha256": self.market.fingerprint(),
            "status": "RESEARCH_NOT_ACCEPTED",
        }


__all__ = ["EntryOnlyRisk", "Owner", "Parameters", "grid"]

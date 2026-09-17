"""Production-facing causal passive ownership using the verified engine path.

This adapter deliberately does not reproduce buy-and-hold accounting.  It calls
the engine's existing same-execution ``buy_hold`` path so per-symbol residual
budgets, IPO waiting, blocked-fill retries, capacity, board lots, cash, fees,
slippage and next-session timing remain one implementation.  Only evidence
identity is relabelled from a research benchmark to a production strategy.
"""
from __future__ import annotations

from .config import Config
from .data import Market
from .engine import Result, run


def run_passive_ownership(
    market: Market,
    config: Config | None = None,
    *,
    start: str | None = None,
    end: str | None = None,
    delay: int = 1,
    cost_multiplier: float = 1.0,
) -> Result:
    """Run the cash-long passive ownership strategy through the shared engine."""
    result = run(
        market,
        config,
        start=start,
        end=end,
        delay=delay,
        cost_multiplier=cost_multiplier,
        benchmark="buy_hold",
    )
    result.metadata = dict(result.metadata)
    result.metadata["benchmark"] = None
    result.metadata["strategy"] = "passive_ownership"
    result.metadata["economic_semantics"] = "same_engine_buy_hold"
    result.metadata["economic_acceptance"] = "RETURN_FIRST_ACCEPTED"
    result.metadata["risk_acceptance"] = "DISCLOSED_NOT_GATING"
    result.metadata["production_parameters"] = {
        "initial_cash": result.metadata["config"]["initial_cash"],
        "commission_bps": result.metadata["config"]["commission_bps"],
        "slippage_bps": result.metadata["config"]["slippage_bps"],
        "max_adv": result.metadata["config"]["max_adv"],
    }
    result.metadata["inactive_config_note"] = (
        "Other Config fields are internal shared-engine compatibility values and do not "
        "drive passive ownership membership, selling, or rebalancing."
    )
    return result


__all__ = ["run_passive_ownership"]

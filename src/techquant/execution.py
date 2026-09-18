"""Next-open research execution primitives, not a broker or corporate-action ledger."""
from __future__ import annotations

import math


MIN_ORDER_NAV_FRACTION = 0.01


def is_material_order(notional: float, nav: float) -> bool:
    """Return whether an ordinary order reaches the shared NAV floor."""

    if (
        not math.isfinite(notional)
        or notional < 0
        or not math.isfinite(nav)
        or nav <= 0
    ):
        raise ValueError("order notional and NAV must be finite and valid")
    return notional + 1e-8 >= nav * MIN_ORDER_NAV_FRACTION


def stamp_rate(date: str) -> float:
    """Seller-only rate; the 2023-08-28 legal change is not an alpha date switch."""
    return .001 if date < '2023-08-28' else .0005


def round_quantity(symbol: str, quantity: float) -> int:
    """Minimum declaration/step for ordinary buys (not tax lots)."""
    if not math.isfinite(quantity) or quantity < 0:
        raise ValueError('quantity must be finite and nonnegative')
    q = math.floor(quantity + 1e-9)
    if symbol.startswith('sh688'):
        return q if q >= 200 else 0
    if symbol.startswith('bj'):
        return q if q >= 100 else 0
    return q // 100 * 100


def daily_limit(symbol: str) -> float:
    if symbol.startswith('bj'):
        return .30
    return .20 if symbol.startswith(('sz300', 'sz301', 'sh688')) else .10


def fee(notional: float, side: str, date: str, commission_bps: float,
        multiplier: float = 1.) -> float:
    """Minimum CNY 5 commission; sell stamp duty; 0.1bp transfer assumption.

    Fee schedules and account-specific rebates require human verification.
    A multiplier stresses every cash cost, including the minimum commission.
    """
    if notional <= 0:
        return 0.
    return multiplier * (max(5., notional * commission_bps / 10_000) +
                         notional * (.00001 + (stamp_rate(date) if side == 'SELL' else 0)))

"""Cash-conserving, next-session adjusted-unit research replay.

A unit is NOT an actual share: adjusted-price returns embed corporate actions.
Raw prices only estimate board-lot and liquidity constraints. Every result is
labelled accordingly, including when a native comparator uses the same prices.
There are no intraday stops or invented fills at a breached stop price.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .data import Market
from .execution import daily_limit, fee, round_quantity
from .features import build_features
from .policy import CloseObservation, ClosePolicy
from .strategy import RiskState, target_weights


@dataclass
class Result:
    equity: pd.DataFrame
    targets: pd.DataFrame
    orders: list[dict[str, Any]]
    metadata: dict[str, Any]


def run(market: Market, config: Config | None = None, *,
        start: str | None = None, end: str | None = None,
        targets: pd.DataFrame | None = None, delay: int = 1,
        cost_multiplier: float = 1., benchmark: str | None = None,
        policy_factory: Callable[[Market, Config], ClosePolicy] | None = None) -> Result:
    cfg = config or Config()
    if policy_factory is not None and (benchmark is not None or targets is not None):
        raise ValueError('choose one policy, benchmark or external targets')
    if isinstance(delay, bool) or not isinstance(delay, int) or delay < 1:
        raise ValueError('execution delay must be at least one session')
    if not np.isfinite(cost_multiplier) or cost_multiplier < 0:
        raise ValueError('invalid cost multiplier')
    if benchmark not in (None, 'buy_hold', 'equal_weight') or (benchmark and targets is not None):
        raise ValueError('invalid or conflicting benchmark policy')
    if end is not None:
        if pd.Timestamp(end) > market.calendar[-1]:
            raise ValueError('requested end exceeds verified snapshot coverage')
        market = market.prefix(end)
    start_date = pd.Timestamp(start) if start else market.calendar[0]
    if start_date < pd.Timestamp('2023-01-01'):
        raise ValueError('pre-2023 measurement is outside scope')
    begin = int(market.calendar.searchsorted(start_date))
    if begin >= len(market.calendar):
        raise ValueError('empty measurement window')
    n, names = len(market.calendar), market.symbols
    # A factory, rather than a reused mutable object, isolates every replay.
    active_policy = policy_factory(market, cfg) if policy_factory is not None else None
    if targets is not None:
        if not targets.index.equals(market.calendar) or set(targets.columns) != set(names):
            raise ValueError('targets must cover the exact market calendar and universe')
        external = targets.loc[:, list(names)].to_numpy(dtype=float)
        if not np.isfinite(external).all() or (external < 0).any() or (external.sum(axis=1) > 1 + 1e-10).any():
            raise ValueError('targets must be finite, long-only and cash-funded')
    else:
        external = None
    op = market.panel('open').to_numpy()
    rop = market.panel('raw_open').to_numpy()
    qclose = market.panel('close')
    close = qclose.ffill().to_numpy()
    prev_close = qclose.ffill().shift().to_numpy()
    # Nothing from today's total volume/high/low participates in an opening fill.
    adv = (market.panel('volume') * market.panel('raw_close')).rolling(
        20, min_periods=1).mean().shift().to_numpy()
    f = build_features(market, cfg)
    units = np.zeros(len(names))
    cash = cfg.initial_cash
    decisions = np.zeros((n, len(names)))
    unit_decisions = np.zeros_like(decisions)
    inventory_intent = np.zeros(n, dtype=bool)
    inventory_side = np.zeros_like(decisions)
    actions = np.zeros_like(decisions, dtype=bool)
    urgent = np.zeros_like(decisions, dtype=bool)
    buy_hold_budget = np.full(len(names), cfg.initial_cash / len(names))
    rows, orders, history = [], [], []
    risk = RiskState()
    last_cap = 0.
    for i in range(begin, n):
        date = str(market.calendar[i].date())
        if i >= begin + delay:
            signal_i = i - delay
            pending = decisions[signal_i]
            open_marks = np.where(np.isfinite(op[i]), op[i], prev_close[i])
            value = np.nan_to_num(units * open_marks, nan=0.)
            opening_nav = cash + value.sum()
            difference = pending * opening_nav - value
            if inventory_intent[signal_i]:
                # Freeze intended inventory at its signal close. A price gap
                # must not reverse a unit reduction into an opening purchase.
                difference = (unit_decisions[signal_i] - units) * open_marks
                # Earlier queued fills may already have reached this target.
                # A stale increase/reduction must never execute its opposite.
                difference[difference * inventory_side[signal_i] <= 0] = 0.
            if benchmark == 'buy_hold':
                # Spend each original cash slice once; never sell to restore weights.
                difference = buy_hold_budget.copy()
            difference[~actions[signal_i]] = 0.
            # Sells first. Stable symbol order ensures replay/subset order invariance.
            for side in ('SELL', 'BUY'):
                for j, symbol in enumerate(names):
                    delta = float(difference[j])
                    protective = side == 'SELL' and (pending[j] == 0 or urgent[signal_i, j]) and units[j] > 1e-10
                    if (side == 'SELL' and delta >= -1e-7) or (side == 'BUY' and delta <= 1e-7):
                        continue
                    # The signal generator handles bands; executable materiality is
                    # only 1% NAV. A zero-target protective exit is never filtered.
                    if not protective and abs(delta) < opening_nav * .01:
                        continue
                    order = {'date': date, 'signal_date': str(market.calendar[signal_i].date()),
                             'symbol': symbol, 'side': side, 'status': 'BLOCKED',
                             'reason': '', 'units': 0., 'raw_quantity_equivalent': 0.,
                             'notional': 0., 'fee': 0., 'slippage': 0.}
                    if not np.isfinite(op[i, j]) or not np.isfinite(rop[i, j]):
                        order['reason'] = 'NO_OPEN'
                        orders.append(order)
                        continue
                    gap = op[i, j] / prev_close[i, j] - 1 if np.isfinite(prev_close[i, j]) else 0.
                    limit = daily_limit(symbol) - .002
                    if (side == 'BUY' and gap >= limit) or (side == 'SELL' and gap <= -limit):
                        order['reason'] = 'OPEN_LIMIT'
                        orders.append(order)
                        continue
                    capacity = adv[i, j] * cfg.max_adv
                    if not np.isfinite(capacity) or capacity <= 0:
                        order['reason'] = 'NO_PRIOR_CAPACITY'
                        orders.append(order)
                        continue
                    raw_quantity = round_quantity(symbol, min(abs(delta), capacity) / rop[i, j])
                    quantity = raw_quantity * rop[i, j] / op[i, j]
                    if side == 'SELL':
                        quantity = min(quantity, units[j])
                        # Full odd-lot disposal, only when the capacity permits it.
                        if pending[j] == 0 and units[j] * op[i, j] <= capacity + 1e-8:
                            quantity = units[j]
                            raw_quantity = quantity * op[i, j] / rop[i, j]
                    if side == 'BUY':
                        slip = cfg.slippage_bps / 10_000 * cost_multiplier
                        available = max(0., cash - 5 * cost_multiplier)
                        affordable = available / (rop[i, j] * (1 + slip) *
                                                   (1 + (.00001 + cfg.commission_bps / 10_000) * cost_multiplier))
                        raw_quantity = min(raw_quantity, round_quantity(symbol, affordable))
                        quantity = raw_quantity * rop[i, j] / op[i, j]
                    if quantity <= 1e-10:
                        order['reason'] = 'MINIMUM_LOT_OR_CASH'
                        orders.append(order)
                        continue
                    direction = 1 if side == 'BUY' else -1
                    slipped = op[i, j] * (1 + direction * cfg.slippage_bps / 10_000 * cost_multiplier)
                    if slipped <= 0:
                        raise ValueError('stress assumptions imply nonpositive execution price')
                    notional = quantity * slipped
                    charge = fee(notional, side, date, cfg.commission_bps, cost_multiplier)
                    if side == 'BUY' and notional + charge > cash + 1e-7:
                        raise AssertionError('affordability calculation violated cash budget')
                    units[j] += direction * quantity
                    cash -= direction * notional + charge
                    if abs(units[j]) < 1e-8:
                        units[j] = 0.
                    if benchmark == 'buy_hold' and side == 'BUY':
                        buy_hold_budget[j] = max(0., buy_hold_budget[j] - notional - charge)
                    order.update(status='FILLED', reason='NEXT_OPEN', units=float(quantity),
                                 raw_quantity_equivalent=float(raw_quantity), notional=float(notional),
                                 fee=float(charge), slippage=float(abs(slipped - op[i, j]) * quantity))
                    orders.append(order)
        holdings = np.nan_to_num(units * close[i], nan=0.)
        nav = float(cash + holdings.sum())
        if cash < -1e-6 or (units < -1e-8).any() or not np.isfinite(nav) or nav <= 0:
            raise AssertionError('cash/long-only/finite-equity invariant failed')
        history.append(nav)
        weights = holdings / nav
        if active_policy is not None:
            observation = CloseObservation.from_inventory(i, date, nav, float(cash), units, weights)
            decision = active_policy.decide(observation)
            decisions[i] = decision.validated_weights(len(names))
            requested_units = decision.validated_unit_targets(close[i], nav)
            if requested_units is not None:
                inventory_intent[i] = True
                unit_decisions[i] = requested_units
                inventory_side[i] = np.sign(requested_units - units)
            cap, reason = float(decision.cap), decision.reason
        elif external is not None:
            cap, reason = 1., 'EXTERNAL_TARGETS'
            decisions[i] = external[i]
        elif benchmark is not None:
            cap, reason = 1., benchmark.upper()
            available = np.isfinite(op[i]) & np.isfinite(qclose.iloc[i].to_numpy())
            if benchmark == 'equal_weight':
                if (i - begin) % 20 == 0 and available.any():
                    decisions[i, available] = 1 / available.sum()
                else:
                    decisions[i] = weights
            else:
                decisions[i] = weights
                # Preserve unspent initial-capital slices across IPOs and blocked fills.
                desired = np.where(available, buy_hold_budget / nav, 0.)
                budget = max(0., 1 - weights.sum())
                if desired.sum() > budget:
                    desired *= budget / desired.sum()
                decisions[i] += desired
        else:
            cap, reason = risk.update(i, f, history, cfg)
            rebalance = (i - begin) % cfg.rebalance == 0 or cap > last_cap + 1e-10
            decisions[i], why = target_weights(i, f, weights, cfg, cap=cap, rebalance=rebalance,
                                                     risk_reduction=cap < last_cap - 1e-10,
                                                     risk_restoration=cap > last_cap + 1e-10)
            reason = '|'.join([reason, *why])
        actions[i] = np.abs(decisions[i] - weights) > 1e-10
        # A target reduction is deliberate, not noise: the policy already applies
        # its hysteresis. Keep risk reductions executable below ordinary materiality.
        urgent[i] = decisions[i] < weights - 1e-10
        last_cap = cap
        rows.append({'date': market.calendar[i], 'nav': nav, 'cash': float(cash),
                     'holdings': float(holdings.sum()), 'exposure': float(weights.sum()),
                     'target_cap': float(cap), 'breadth': float(f.breadth[i]), 'reason': reason})
    from .evidence import source_identity
    result = Result(pd.DataFrame(rows).set_index('date'),
                  pd.DataFrame(decisions[begin:], index=market.calendar[begin:], columns=names),
                  orders, {'config': asdict(cfg), 'universe': list(names), 'quality': market.quality,
                           'data_sha256': market.fingerprint(), 'source': source_identity(),
                           'provenance': market.provenance, 'delay': delay,
                           'cost_multiplier': cost_multiplier, 'benchmark': benchmark,
                           'start': str(market.calendar[begin].date()), 'end': str(market.calendar[-1].date()),
                           'economic_acceptance': 'UNVERIFIED',
                           'accounting': 'adjusted economic units, not actual shares'})

    if active_policy is not None:
        import json
        identity = active_policy.identity()
        json.dumps(identity, allow_nan=False)  # Fail before issuing unsavable evidence.
        result.metadata['policy'] = identity
    return result

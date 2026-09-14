"""Independent two-pass research policy, not a promoted trading strategy.

A causal paper portfolio keeps evolving while the real account is in cash.
Only paper values already observed at a close may change the next-open budget.
The private, sequential adapter temporarily replaces three engine hooks and
restores them in finally; it must not be used by concurrent production callers.
"""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import techquant.engine as engine


def alpha_features(market, config):
    quoted = market.panel('close')
    active = quoted.notna() & market.panel('volume').gt(0)
    close = quoted.ffill()
    ready = active & (active.cumsum() >= config.fast) & (active.rolling(config.fast).mean() >= .8)
    first = quoted.where(active & (active.cumsum() == 1)).ffill()
    log_returns = [np.log(close / close.shift(k)).fillna(np.log(close / first))
                   for k in (config.fast, config.slow, 2 * config.slow)]
    score = .2 * log_returns[0] + .4 * log_returns[1] + .4 * log_returns[2]
    return SimpleNamespace(symbols=market.symbols, ready=ready.to_numpy(),
        score=score.where(ready, -np.inf).fillna(-np.inf).to_numpy(),
        breadth=ready.mean(axis=1).to_numpy())


class FullBudget:
    def update(self, i, features, equity, config):
        return 1., 'UNPROTECTED_SHADOW'


def alpha_targets(i, features, current, config, *, cap, rebalance,
                  risk_reduction=False, risk_restoration=False):
    current = np.asarray(current, dtype=float)
    want = current.copy()
    want[~features.ready[i]] = 0.
    vacant = np.count_nonzero(want > .01) < min(config.max_positions, np.count_nonzero(features.ready[i]))
    if not rebalance and not vacant:
        return want, ['KEEP_ECONOMIC_UNITS']
    priority = features.score[i] + np.where(want > 0, .05, 0.)
    ranked = sorted(np.flatnonzero(features.ready[i]),
                    key=lambda j: (-priority[j], features.symbols[j]))
    chosen = ranked[:config.max_positions]
    keep = np.zeros(len(want), dtype=bool)
    keep[chosen] = True
    want[~keep] = 0.
    ceiling = max(config.single_cap, 1 / len(want))
    # Material ownership changes may trim concentration; daily drift does not.
    want = np.minimum(want, np.where(want > ceiling + config.trade_band, ceiling, want))
    room = max(0., 1 - want.sum())
    if room > config.trade_band:
        for _ in range(len(chosen) + 1):
            eligible = [j for j in chosen if want[j] < ceiling - 1e-10]
            if not eligible or room < 1e-10:
                break
            share = room / len(eligible)
            for j in eligible:
                addition = min(share, ceiling - want[j])
                want[j] += addition
                room -= addition
    if want.sum() > 1:
        want /= want.sum()
    return want, ['SHADOW_MEMBERSHIP_REVIEW']


def risk_budget(nav, variant):
    allowed = ('ema10', 'ema20', 'drawdown8', 'drawdown12', 'overheat10', 'overheat20')
    if variant not in allowed:
        raise ValueError('unknown shadow risk variant')
    nav = pd.Series(nav, dtype=float)
    if len(nav) == 0 or not np.isfinite(nav).all() or (nav <= 0).any():
        raise ValueError('shadow NAV must be nonempty, finite and positive')
    fast = 10 if variant.endswith('10') else 20
    ema = nav.ewm(span=fast, adjust=False).mean()
    slow = nav.ewm(span=60, adjust=False).mean()
    one = nav.pct_change(fill_method=None).fillna(0)
    three = nav.pct_change(3, fill_method=None).fillna(0)
    if variant.startswith('drawdown'):
        distance = .08 if variant == 'drawdown8' else .12
        alarm = (1 - nav / nav.rolling(20, min_periods=1).max() > distance) & (nav < ema)
    else:
        alarm = nav < ema * .98
    overheat = (nav > slow * 1.30) & ((one < -.03) | (three < -.06)) if variant.startswith('overheat') else pd.Series(False, index=nav.index)
    healthy = (nav > ema * 1.01) & (three > 0)
    cap = 1.
    recovery = 0
    locked_until = -1
    budgets = np.empty(len(nav))
    reasons = []
    for i in range(len(nav)):
        if alarm.iloc[i] or overheat.iloc[i]:
            cap = 0.
            recovery = 0
            if overheat.iloc[i]:
                locked_until = max(locked_until, i + 10)
            reason = 'OVERHEAT_REVERSAL' if overheat.iloc[i] else 'SHADOW_TREND_DAMAGE'
        elif cap == 0:
            recovery = recovery + 1 if healthy.iloc[i] and i >= locked_until else 0
            if recovery >= 3:
                cap = 1.
                recovery = 0
                reason = 'SHADOW_RECOVERY'
            else:
                reason = 'SHADOW_RECOVERY_WAIT'
        else:
            reason = 'SHADOW_TREND_OWNERSHIP'
        budgets[i] = cap
        reasons.append(reason)
    return budgets, reasons


def shadow(market, config, *, delay=1, cost_multiplier=1.):
    old = engine.build_features, engine.RiskState, engine.target_weights
    try:
        engine.build_features = alpha_features
        engine.RiskState = FullBudget
        engine.target_weights = alpha_targets
        return engine.run(market, config, delay=delay, cost_multiplier=cost_multiplier)
    finally:
        engine.build_features, engine.RiskState, engine.target_weights = old


def overlay(market, config, paper, variant, *, delay=1, cost_multiplier=1.):
    if not paper.targets.index.equals(market.calendar):
        raise ValueError('paper calendar does not match the executable market')
    budgets, reasons = risk_budget(paper.equity.nav, variant)
    result = engine.run(market, config, targets=paper.targets.mul(budgets, axis=0),
                        delay=delay, cost_multiplier=cost_multiplier)
    result.equity['target_cap'] = budgets
    result.equity['reason'] = reasons
    result.metadata['policy'] = {'name': 'shadow_portfolio_overlay', 'risk_variant': variant,
        'shadow_data_sha256': paper.metadata['data_sha256'],
        'shadow_config': paper.metadata['config'], 'causal_clock': 'close->next_open'}
    return result

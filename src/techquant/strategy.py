"""Independent persistent-trend allocation with asymmetric de-risking/recovery.

The objective is a testable trade-off, not simultaneous pathwise optimal return
and drawdown. Numerical policy constants below are global, never name/date keys.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .features import Features


@dataclass
class RiskState:
    cap: float = 0.
    healthy: int = 0

    def update(self, i: int, f: Features, equity: list[float],
               config: Config) -> tuple[float, str]:
        if not f.ready[i].any():
            self.cap, self.healthy = 0., 0
            return 0., 'WARMUP_OR_NO_FRESH_QUOTES'
        peak = max(equity[-config.slow:])
        dd = 1 - equity[-1] / peak
        loss = equity[-1] / equity[-2] - 1 if len(equity) > 1 else 0.
        shock = ((f.market_return[i] < -max(.025, config.shock_z * f.market_vol[i])
                  and f.breadth[i] < .5) or f.shock_fraction[i] >= .6)
        if shock:
            desired, reason = 0., 'CROSS_SECTION_SHOCK'
        elif dd >= config.risk_drawdown and loss < -.01:
            desired, reason = 0., 'PORTFOLIO_DRAWDOWN_SHOCK'
        elif f.market_dd[i] >= config.risk_drawdown and f.weak[i]:
            desired, reason = .25, 'MARKET_DRAWDOWN'
        elif f.breadth[i] < .2 and f.weak[i]:
            desired, reason = .25, 'BREADTH_BREAKDOWN'
        elif f.weak[i] and f.breadth[i] < .5:
            desired, reason = .5, 'WEAK_TREND'
        elif dd >= config.risk_drawdown * 2 / 3 and loss < -.005:
            desired, reason = .5, 'PORTFOLIO_WARNING'
        else:
            desired, reason = 1., 'TREND_OPEN'
        if desired < self.cap:
            self.cap, self.healthy = desired, 0
        elif desired > self.cap:
            # A sequence of observed healthy closes, not a known crash-end date.
            self.healthy = self.healthy + 1 if f.breadth[i] >= .5 and not f.weak[i] else 0
            if self.healthy >= config.recovery:
                self.cap = min(desired, self.cap + 1 / config.recovery)
                reason = 'CONFIRMED_RECOVERY'
            else:
                reason = 'RECOVERY_WAIT'
        else:
            self.healthy = 0 if desired < 1 else self.healthy
        return self.cap, reason


def _allocate(indices: list[int], score: np.ndarray, sectors: tuple[str, ...],
              config: Config, universe_size: int) -> np.ndarray:
    result = np.zeros(universe_size)
    if not indices:
        return result
    cap = max(config.single_cap, 1 / universe_size)
    sector_cap = config.sector_cap if len(set(sectors)) > 1 else 1.
    remaining = list(indices)
    # Water-fill only into slack. Capped groups are not normalized back above cap.
    for _ in range(len(indices) + 1):
        budget = 1 - result.sum()
        if budget <= 1e-12 or not remaining:
            break
        strengths = np.sqrt(np.maximum(score[remaining], .001))
        proposal = budget * strengths / strengths.sum()
        changed = False
        for j, amount in zip(remaining, proposal, strict=True):
            group = sectors[j]
            used = sum(result[k] for k in indices if sectors[k] == group)
            addition = min(float(amount), cap - result[j], sector_cap - used)
            result[j] += max(0., addition)
            changed |= addition > 1e-12
        remaining = [j for j in remaining if result[j] < cap - 1e-10 and
                     sum(result[k] for k in indices if sectors[k] == sectors[j]) < sector_cap - 1e-10]
        if not changed:
            break
    return result


def target_weights(i: int, f: Features, current: np.ndarray, config: Config,
                   *, cap: float, rebalance: bool) -> tuple[np.ndarray, list[str]]:
    current = np.asarray(current, dtype=float)
    if not np.isfinite(current).all() or (current < -1e-10).any():
        raise ValueError('invalid current portfolio weights')
    want = current.copy()
    reasons = []
    forced = f.exit[i] | ~f.ready[i]
    if np.any(forced & (want > 0)):
        want[forced] = 0.
        reasons.append('TREND_EXIT_OR_STALE')
    if current.sum() > cap + 1e-10:
        reasons.append('RISK_REDUCTION')
    if cap <= 0:
        return np.zeros_like(current), reasons
    if rebalance:
        eligible = f.ready[i] & ~f.exit[i] & (f.entry[i] | (current > 0)) & (f.score[i] > 0)
        # Incumbent preference is a turnover control, never a permanent exemption.
        priority = f.score[i] + np.where(current > 0, .10 * np.maximum(f.score[i], 0), 0)
        indices = sorted(np.flatnonzero(eligible), key=lambda j: (-priority[j], f.symbols[j]))
        indices = indices[:config.max_positions]
        candidate = _allocate(indices, f.score[i], f.sectors, config, len(current))
        if candidate.sum() > 0:
            # Perfect-correlation volatility bound: do not assume sector peers diversify.
            sigma = float(np.dot(candidate, np.nan_to_num(f.vol[i], nan=.05))) * np.sqrt(252)
            candidate *= min(cap, config.target_vol / max(sigma, .01))
        reductions = (candidate < current - config.trade_band) | (candidate == 0)
        increases = candidate > current + config.trade_band
        want[reductions | increases] = candidate[reductions | increases]
        want[forced] = 0.
        reasons.append('SCHEDULED_SELECTION')
    # Ordinary drift has a declared band. Actual risk reductions do not.
    symbol_cap = max(config.single_cap, 1 / len(current))
    over = want > symbol_cap + config.trade_band
    want[over] = symbol_cap
    if want.sum() > cap:
        want *= cap / want.sum()
    return np.maximum(want, 0.), reasons

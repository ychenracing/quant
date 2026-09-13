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
    episode_peak: float = 0.

    def update(self, i: int, f: Features, equity: list[float],
               config: Config) -> tuple[float, str]:
        if not f.ready[i].any():
            self.cap, self.healthy = 0., 0
            return 0., 'WARMUP_OR_NO_FRESH_QUOTES'
        self.episode_peak = max(self.episode_peak, equity[-1])
        dd = 1 - equity[-1] / self.episode_peak
        loss = equity[-1] / equity[-2] - 1 if len(equity) > 1 else 0.
        shock = ((f.market_return[i] < -max(.025, config.shock_z * f.market_vol[i])
                  and f.breadth[i] < .5) or f.shock_fraction[i] >= .6)
        # A slow sell-off need not contain a single large down day. The scale
        # was observable before the three-session loss interval, not fitted to it.
        loss3 = float(np.prod(1 + f.market_return[i - 2:i + 1]) - 1) if i >= 2 else 0.
        accumulated_shock = (i >= 2 and f.breadth[i] < .5 and
            loss3 < -max(.025, config.shock_z * f.market_vol[i - 2] * np.sqrt(3)))
        if shock:
            desired, reason = 0., 'CROSS_SECTION_SHOCK'
        elif accumulated_shock:
            desired, reason = 0., 'ACCUMULATED_MARKET_SHOCK'
        elif dd >= config.risk_drawdown and loss < -.01:
            desired, reason = 0., 'PORTFOLIO_DRAWDOWN_SHOCK'
        elif (f.market_dd[i] >= config.risk_drawdown and f.weak[i]
              and f.breadth[i] < .2):
            desired = .5 if bool(np.any(f.entry[i] & ~f.exit[i])) else .25
            reason = 'BROAD_TREND_BREAKDOWN'
        elif (dd >= config.risk_drawdown * 2 / 3
              and loss < -max(.02, 1.5 * f.market_vol[i])):
            desired, reason = .5, 'PORTFOLIO_WARNING'
        else:
            # An index warning is not an automatic veto on intact technology
            # leadership. Distinguish observation from an actual risk-budget cut.
            desired = 1.
            reason = 'MARKET_WEAK_WARNING' if f.weak[i] else 'TREND_OPEN'
        if desired < self.cap:
            self.cap, self.healthy = desired, 0
        elif desired > self.cap:
            # A sequence of observed healthy closes, not a known crash-end date.
            broad_recovery = f.breadth[i] >= .5 and not f.weak[i]
            needed = min(config.max_positions, len(f.symbols))
            selective_recovery = (f.market_return[i] >= 0 and
                np.count_nonzero(f.ready[i] & f.entry[i] & ~f.exit[i]) >= needed)
            self.healthy = self.healthy + 1 if broad_recovery or selective_recovery else 0
            if self.healthy >= config.recovery:
                # Re-arm intervention for new risk capital, rather than repeatedly
                # liquidating on the same old loss. Reported NAV/drawdown never reset.
                if self.cap == 0:
                    self.episode_peak = equity[-1]
                # First restore at most half risk. A second confirmation interval
                # is required for full exposure; every new alarm cuts immediately.
                self.cap = min(desired, .5 if self.cap < .5 else 1.)
                self.healthy = 0
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
                   *, cap: float, rebalance: bool, risk_reduction: bool = False,
                   risk_restoration: bool = False) -> tuple[np.ndarray, list[str]]:
    current = np.asarray(current, dtype=float)
    if not np.isfinite(current).all() or (current < -1e-10).any():
        raise ValueError('invalid current portfolio weights')
    want = current.copy()
    reasons = []
    forced = f.exit[i] | ~f.ready[i]
    if np.any(forced & (want > 0)):
        want[forced] = 0.
        reasons.append('TREND_EXIT_OR_STALE')
    ceiling = cap if risk_reduction or cap <= 0 else min(1., cap + config.trade_band)
    if current.sum() > ceiling + 1e-10:
        reasons.append('RISK_REDUCTION')
    if cap <= 0:
        return np.zeros_like(current), reasons
    if rebalance:
        eligible = f.ready[i] & ~f.exit[i] & (f.entry[i] | (current > 0)) & (f.score[i] > 0)
        # Incumbent preference is a turnover control, never a permanent exemption.
        priority = f.score[i] + np.where(current > 0, np.maximum(f.score[i], 0), 0)
        indices = sorted(np.flatnonzero(eligible), key=lambda j: (-priority[j], f.symbols[j]))
        indices = indices[:config.max_positions]
        same_members = set(indices) == set(np.flatnonzero(current > 1e-10))
        if same_members and not risk_restoration:
            # Preserve an intact economic position, not its old target percentage.
            # Score changes alone do not harvest winners or top up laggards.
            reasons.append('RETAIN_INTACT_MEMBERSHIP')
        else:
            candidate = _allocate(indices, f.score[i], f.sectors, config, len(current))
            if candidate.sum() > 0:
                sigma = float(np.dot(candidate, np.nan_to_num(f.vol[i], nan=.05))) * np.sqrt(252)
                candidate *= min(cap, config.target_vol / max(sigma, .01))
            reductions = (candidate < current - config.trade_band) | (candidate == 0)
            increases = candidate > current + config.trade_band
            want[reductions | increases] = candidate[reductions | increases]
            want[forced] = 0.
            reasons.append('RISK_RESTORATION' if risk_restoration else 'SCHEDULED_SELECTION')
    # Ordinary drift has a declared band. Actual risk reductions do not.
    symbol_cap = max(config.single_cap, 1 / len(current))
    over = want > symbol_cap + config.trade_band
    want[over] = symbol_cap
    if len(set(f.sectors)) > 1:
        for sector in sorted(set(f.sectors)):
            group = np.array([s == sector for s in f.sectors])
            used = float(want[group].sum())
            if used > config.sector_cap + config.trade_band:
                want[group] *= config.sector_cap / used
                reasons.append('SECTOR_LIMIT')
    # The same perfect-correlation bound also applies to retained positions.
    # Hysteresis avoids resizing for every small daily volatility change.
    sigma = float(np.dot(want, np.nan_to_num(f.vol[i], nan=.05))) * np.sqrt(252)
    if sigma > config.target_vol * (1 + config.trade_band):
        want *= config.target_vol / sigma
        reasons.append('VOLATILITY_REDUCTION')
    if want.sum() > ceiling:
        want *= cap / want.sum()
    return np.maximum(want, 0.), reasons

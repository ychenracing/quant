"""Independent persistent leadership with fill-aware asymmetric pressure control.

Long-horizon membership is separate from daily protection. Pressure saves an
absolute unit ceiling, not a repeatedly halved weight. Orders remain intentions:
only the independent engine can change cash and actual inventory.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
from techquant.data import Market, file_hash
from techquant.policy import CloseDecision, CloseObservation


@dataclass(frozen=True)
class Parameters:
    leadership_window: int = 120
    exit_span: int = 60
    positions: int = 2
    pressure_guard: bool = True

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in
               (self.leadership_window, self.exit_span, self.positions)):
            raise ValueError('windows and positions must be positive integers')
        if not isinstance(self.pressure_guard, bool):
            raise ValueError('pressure guard must be boolean')


@dataclass
class Signals:
    score: np.ndarray
    price: np.ndarray
    open: np.ndarray
    ema: np.ndarray
    ema10: np.ndarray
    ready: np.ndarray
    momentum5: np.ndarray
    pressure: np.ndarray
    recovery: np.ndarray


def signals(market: Market, p: Parameters) -> Signals:
    quote, volume = market.panel('close'), market.panel('volume')
    price = quote.ffill()
    active = quote.notna() & volume.gt(0)
    count = active.cumsum()
    first = quote.where(active & count.eq(1)).ffill()
    age = count.sub(1).clip(lower=1, upper=p.leadership_window)
    score = np.log(price / price.shift(p.leadership_window).fillna(first)) * 252. / age
    ema = price.ewm(span=p.exit_span, adjust=False).mean()
    ema10 = price.ewm(span=10, adjust=False).mean()
    daily = price.pct_change(fill_method=None)
    down = np.sqrt(daily.clip(upper=0).pow(2).rolling(20, min_periods=5).mean()).shift().clip(lower=.01)
    pressure = daily.lt(-2.5 * down) & price.lt(price.shift(3)) & active
    recovery = (price.gt(ema10).rolling(3, min_periods=3).sum().eq(3) &
                price.ge(price.shift().rolling(10, min_periods=10).max()) & active)
    ready = active & count.ge(20) & np.isfinite(score)
    return Signals(np.where(ready, score, -np.inf), price.to_numpy(), market.panel('open').to_numpy(),
                   ema.to_numpy(), ema10.to_numpy(), ready.to_numpy(),
                   price.pct_change(5, fill_method=None).to_numpy(),
                   pressure.to_numpy(), recovery.to_numpy())


class Owner:
    def __init__(self, market: Market, p: Parameters, prepared: Signals | None = None):
        self.market, self.params, self.s = market, p, prepared or signals(market, p)
        n = len(market.symbols)
        self.previous = np.zeros(n, dtype=bool)
        self.peak = np.zeros(n)
        self.below = np.zeros(n, dtype=int)
        self.above = np.zeros(n, dtype=int)
        self.exit_pending = np.zeros(n, dtype=bool)
        self.readmit = np.zeros(n, dtype=bool)
        self.pressure_ceiling = np.full(n, np.nan)
        self.restore_units = np.zeros(n)
        self.restore_pending = np.zeros(n, dtype=bool)
        self.last_session = -1

    def decide(self, o: CloseObservation) -> CloseDecision:
        i, p, s = o.session, self.params, self.s
        if i <= self.last_session:
            raise ValueError('policy state requires increasing sessions')
        self.last_session = i
        held = o.units > 1e-10
        new = held & ~self.previous
        sold = ~held & self.previous
        self.peak[new] = np.maximum(s.price[i, new], s.open[i, new])
        self.peak[held] = np.maximum(self.peak[held], s.price[i, held])
        self.peak[~held] = 0.
        self.pressure_ceiling[new | sold] = np.nan
        self.restore_units[new | sold] = 0.
        self.restore_pending[new | sold] = False
        self.readmit[sold & self.exit_pending] = True
        self.exit_pending[~held] = False
        self.above = np.where(s.ready[i] & (s.price[i] > s.ema10[i]), self.above + 1, 0)
        self.readmit[self.above >= 3] = False
        self.below = np.where(s.price[i] < .97 * s.ema[i], self.below + 1, 0)
        drop = np.zeros(len(held))
        np.divide(s.price[i], self.peak, out=drop, where=held & (self.peak > 0))
        broken = held & ((self.below >= 2) | (1. - drop >= .14) | ~s.ready[i])
        self.exit_pending |= broken
        want, reasons = o.weights.copy(), []
        want[self.exit_pending] = 0.
        self.pressure_ceiling[self.exit_pending] = np.nan
        self.restore_units[self.exit_pending] = 0.
        self.restore_pending[self.exit_pending] = False
        if self.exit_pending.any():
            reasons.append('PROTECTIVE_EXIT_RETRY')
        pressure_new = held & s.pressure[i] & ~self.exit_pending & np.isnan(self.pressure_ceiling)
        if p.pressure_guard:
            self.pressure_ceiling[pressure_new] = o.units[pressure_new] * .5
            self.restore_units[pressure_new] = o.units[pressure_new]
        guarded = held & np.isfinite(self.pressure_ceiling)
        for j in np.flatnonzero(guarded):
            # A price change cannot make an unfilled pressure reduction disappear.
            desired_units = self.pressure_ceiling[j]
            if s.recovery[i, j] and o.units[j] <= desired_units + 1e-8:
                self.restore_pending[j] = True
            if self.restore_pending[j]:
                # Restoration cannot borrow planned sale proceeds or liquidate peers.
                add = min(max(0., self.restore_units[j] - o.units[j]) * s.price[i, j] / o.nav,
                          max(0., o.cash / o.nav - np.maximum(want-o.weights, 0.).sum()))
                want[j] = min(o.weights[j] + add, 1. if len(held) == 1 else .8)
                reasons.append('CASH_FUNDED_RESTORATION')
            else:
                want[j] = min(o.weights[j], desired_units * s.price[i, j] / o.nav)
                reasons.append('PRESSURE_REDUCTION')
        # After a restored fill, lift the saved ceiling instead of reducing again.
        restored = guarded & self.restore_pending & (o.units >= self.restore_units - 1e-8) & ~pressure_new
        self.pressure_ceiling[restored] = np.nan
        self.restore_units[restored] = 0.
        self.restore_pending[restored] = False
        if restored.any():
            want[restored] = o.weights[restored]
        drift = 1. if len(held) == 1 else .8
        want = np.minimum(want, drift)
        capacity = min(p.positions, len(held))
        scores = s.score[i]
        eligible = s.ready[i] & (scores > 0) & (s.price[i] > s.ema10[i]) & (s.price[i] > s.ema[i]) & (s.momentum5[i] > 0)
        ranked = sorted(np.flatnonzero(s.ready[i] & (scores > 0)), key=lambda j: (-scores[j], self.market.symbols[j]))
        entrants = [j for j in ranked if eligible[j] and not held[j] and not self.readmit[j]]
        # A new pressure event is a warning, not a source of speculative buying.
        blocked = self.exit_pending.any() or (p.pressure_guard and pressure_new.any())
        if not blocked:
            existing = list(np.flatnonzero(held))
            if i % 20 == 0 and len(existing) >= capacity and entrants:
                ranks = {j:k+1 for k,j in enumerate(ranked)}
                worst = min(existing, key=lambda j:(scores[j],self.market.symbols[j]))
                if ranks.get(worst,len(held)+1) > 2*capacity and scores[entrants[0]] > 1.25*max(scores[worst],.001):
                    want[worst] = 0.
                    reasons.append('LONG_LEADERSHIP_REPLACEMENT')
            entrants = entrants[:max(0,capacity-len(existing))]
            free = min(max(0.,o.cash/o.nav-np.maximum(want-o.weights,0.).sum()),max(0.,1-want.sum()))
            if entrants and free >= .01:
                amount=min(free/len(entrants),1. if len(held)==1 else .6)
                want[entrants] = amount
                reasons.append('PERSISTENT_LEADERSHIP_ENTRY')
        self.previous = held.copy()
        return CloseDecision(want,'|'.join(reasons) if reasons else 'RETAIN_FUNDED_LEADERS')

    def identity(self):
        return {'name':'persistent_leadership','parameters':asdict(self.params),
                'implementation_sha256':file_hash(Path(__file__)),
                'data_sha256':self.market.fingerprint(),'status':'RESEARCH_NOT_ACCEPTED'}

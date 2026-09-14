"""Preregistered price-led ownership with unchanged actual-fill risk coordination.

No forecast is fitted, loaded or consulted. This is a signal-source extension
of quant's independently written owner, not a reference-strategy derivative.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
from techquant.data import Market, file_hash
from research.coherent import Owner as FilledOwner, Parameters as FilledParameters, SignalInputs


@dataclass(frozen=True)
class Parameters:
    positions: int = 2

    def __post_init__(self):
        if type(self.positions) is not int or self.positions not in (2, 4):
            raise ValueError('positions must be one of the two registered capacities')


def grid():
    return [Parameters(2), Parameters(4)]


@dataclass(frozen=True)
class PriceSignals:
    price: np.ndarray
    ready: np.ndarray
    ema10: np.ndarray
    ema20: np.ndarray
    momentum5: np.ndarray
    ret1: np.ndarray


def price_signals(market: Market) -> PriceSignals:
    quoted = market.panel('close')
    active = quoted.notna() & market.panel('volume').gt(0)
    price = quoted.ffill()
    # Forward-filled marks do not count as observations or confer readiness.
    ready = active & active.cumsum().ge(20)
    return PriceSignals(price.to_numpy(), ready.to_numpy(),
        price.ewm(span=10, adjust=False).mean().to_numpy(),
        price.ewm(span=20, adjust=False).mean().to_numpy(),
        price.pct_change(5, fill_method=None).to_numpy(),
        price.pct_change(fill_method=None).to_numpy())


class Owner(FilledOwner):
    def __init__(self, market: Market, p: Parameters):
        self.book_parameters = p
        self.price_signals = price_signals(market)
        super().__init__(market, FilledParameters('price'))

    def _validate_signal_source(self, market: Market, prediction):
        if prediction is not None:
            raise ValueError('the registered price book cannot consume forecasts')

    def _signal_inputs(self, i: int) -> SignalInputs:
        p = self.price_signals
        score = self.features.score[i]
        allowed = (p.ready[i] & self.s.entry[i] & (p.price[i] > p.ema20[i])
                   & (p.momentum5[i] > 0) & np.isfinite(score) & (score > 0))
        if not self.s.market[i]:
            allowed[:] = False
        return SignalInputs(p.price[i], p.ready[i],
            p.ready[i] & (p.price[i] > p.ema10[i]),
            self.s.exit[i] | (p.ret1[i] <= -.08) | ~p.ready[i],
            score, allowed, self.book_parameters.positions)

    def identity(self):
        root = Path(__file__).parent
        return {'name': 'observed_price_trend_book',
                'parameters': asdict(self.book_parameters),
                'implementation_sha256': file_hash(Path(__file__)),
                'inventory_owner_sha256': file_hash(root / 'coherent.py'),
                'observed_trend_sha256': file_hash(root / 'observed_trend.py'),
                'contract_sha256': file_hash(root / 'trend_book_contract.json'),
                'data_sha256': self.market.fingerprint(),
                'forecast_mode': 'no_forecast_inputs_or_fit',
                'status': 'RESEARCH_NOT_ACCEPTED'}

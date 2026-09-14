"""Reconcile actual-fill holding episodes and symbol PnL to an immutable replay."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd
from techquant.data import Market
from techquant.engine import Result


def attribute(market: Market, result: Result) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Break daily PnL into carried inventory and actual execution/fee effects.

    Episode return divides cash PnL by actual buy notional, not capital at risk.
    It is an attribution statistic and not a reinvestable portfolio return.
    """
    if (list(market.symbols) != result.metadata['universe'] or
            market.fingerprint() != result.metadata['data_sha256']):
        raise ValueError('attribution market identity mismatch')
    close = market.panel('close').ffill().loc[result.equity.index]
    names = list(market.symbols)
    index = {name: j for j, name in enumerate(names)}
    fills = defaultdict(list)
    for order in result.orders:
        if order['status'] == 'FILLED':
            fills[order['date']].append(order)
    units = np.zeros(len(names))
    live, finished, days, errors = {}, [], [], []
    previous_nav = float(result.metadata['config']['initial_cash'])
    previous_prices = close.iloc[0].to_numpy()
    for date, quote in close.iterrows():
        day = str(date.date())
        prices, before = quote.to_numpy(), units.copy()
        carry = np.nan_to_num(before * (prices - previous_prices), nan=0.)
        execution = np.zeros(len(names))
        sold = np.zeros(len(names))
        for order in fills[day]:
            j = index[order['symbol']]
            direction = 1 if order['side'] == 'BUY' else -1
            quantity = float(order['units'])
            if pd.Timestamp(order['signal_date']) >= date:
                raise ValueError('fill precedes an executable post-close signal')
            if direction < 0:
                sold[j] += quantity
                if sold[j] > before[j] + 1e-7:
                    raise ValueError('sales exceed opening sellable inventory')
            if units[j] < 1e-7 and direction > 0:
                live[j] = dict(symbol=order['symbol'], entry=day, entry_signal=order['signal_date'],
                               net_cash=0., buy_notional=0., buys=0, sells=0,
                               entry_reason=result.equity.loc[order['signal_date'], 'reason'])
            episode = live[j]
            episode['net_cash'] -= direction * order['notional'] + order['fee']
            episode['buy_notional'] += order['notional'] if direction > 0 else 0.
            episode['buys' if direction > 0 else 'sells'] += 1
            execution[j] += direction * (quantity * prices[j] - order['notional']) - order['fee']
            units[j] += direction * quantity
            if units[j] < 1e-7:
                units[j] = 0.
                episode.update(exit=day, exit_signal=order['signal_date'], pnl=episode['net_cash'],
                               exit_reason=result.equity.loc[order['signal_date'], 'reason'])
                finished.append(episode)
                del live[j]
        pnl = carry + execution
        nav = float(result.equity.loc[date, 'nav'])
        error = float(pnl.sum() - (nav - previous_nav))
        if abs(error) > max(1e-6, nav * 1e-10):
            raise ValueError(f'ledger does not reconcile on {day}: {error}')
        errors.append(error)
        for j in range(len(names)):
            if before[j] > 0 or units[j] > 0 or abs(pnl[j]) > 1e-8:
                days.append(dict(date=day, symbol=names[j], pnl=pnl[j], carry=carry[j],
                                 execution=execution[j], units_before=before[j], units_after=units[j]))
        previous_prices, previous_nav = prices, nav
    for j, episode in live.items():
        episode.update(exit='OPEN', pnl=episode['net_cash'] + units[j] * previous_prices[j])
        finished.append(episode)
    for episode in finished:
        episode['return_on_buys'] = episode['pnl'] / episode['buy_notional']
    nav = result.equity.nav
    drawdown = 1 - nav / nav.cummax().clip(lower=result.metadata['config']['initial_cash'])
    trough = drawdown.idxmax()
    peak = nav.loc[:trough].idxmax()
    peak_label = (str(peak.date()) if nav.loc[peak] >= result.metadata['config']['initial_cash']
                  else 'INITIAL_CASH')
    summary = dict(max_reconciliation_error=max(map(abs, errors), default=0.),
                   episodes=len(finished), wins=int(sum(e['pnl'] > 0 for e in finished)),
                   drawdown_peak=peak_label, drawdown_trough=str(trough.date()),
                   max_drawdown=float(drawdown.max()),
                   episode_return_basis='cash PnL divided by actual buy notional')
    return pd.DataFrame(days), pd.DataFrame(finished), summary

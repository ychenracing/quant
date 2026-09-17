"""Offline decision support and explicit research benchmarks; never sends broker orders."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import Config
from .data import Market, load_market
from .engine import Result, run
from .passive import run_passive_ownership
from .evidence import metrics, save_result, verify_evidence



_PRODUCTION_CONFIG_FIELDS = {"initial_cash", "commission_bps", "slippage_bps", "max_adv"}
_BLOCK_REASON_TEXT = {
    "NO_OPEN": "下一交易日没有可用开盘价",
    "OPEN_LIMIT": "下一交易日开盘接近涨跌停约束，未模拟成交",
    "NO_PRIOR_CAPACITY": "信号日前成交额不足以支持容量估计",
    "MINIMUM_LOT_OR_CASH": "整手或剩余现金不足",
    "BELOW_MINIMUM_NOTIONAL": "容量/整手/现金裁剪后低于普通最小成交金额",
    "OPEN_REBOUND_DEFER": "下一交易日高开，暂缓保护性卖出并等待收盘后重新判断",
}


def _load_production_config(path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("production config must be a JSON object")
    unknown = set(raw) - _PRODUCTION_CONFIG_FIELDS
    if unknown:
        raise ValueError(
            "passive production config does not use: " + ", ".join(sorted(unknown))
        )
    return Config(**raw)


def inspection_report(market: Market, result: Result, names: dict) -> dict:
    """Explain the passive replay book without pretending it is a broker account."""
    units = {symbol: 0.0 for symbol in market.symbols}
    last_fill = {symbol: None for symbol in market.symbols}
    last_block = {symbol: None for symbol in market.symbols}
    for order in result.orders:
        symbol = order["symbol"]
        if order["status"] == "FILLED":
            units[symbol] += order["units"] * (1 if order["side"] == "BUY" else -1)
            last_fill[symbol] = order
        elif order["status"] == "BLOCKED":
            last_block[symbol] = order

    last = result.equity.iloc[-1]
    targets = result.targets.iloc[-1]
    rows, buys, holds = [], [], []
    for symbol in market.symbols:
        history = market.frames[symbol].close.loc[: result.equity.index[-1]]
        price = float(history.iloc[-1]) if len(history) else float("nan")
        current_value = units[symbol] * price if price == price else 0.0
        current_weight = current_value / float(last.nav)
        target_weight = float(targets[symbol])
        pending_value = max(0.0, (target_weight - current_weight) * float(last.nav))
        block = last_block[symbol]
        fill = last_fill[symbol]
        block_is_current = bool(
            block and (fill is None or block["date"] >= fill["date"]) and pending_value > 1e-6
        )
        if pending_value > 1e-6:
            action = "BUY_PENDING"
            explanation = "仍有初始持有预算未成交；仅在下一可执行开盘继续尝试买入"
            buys.append({"symbol": symbol, "name": names.get(symbol, symbol),
                         "estimated_value": pending_value,
                         "blocked_reason": _BLOCK_REASON_TEXT.get(block["reason"], block["reason"]) if block_is_current else None})
        elif current_weight > 1e-10:
            action = "HOLD"
            explanation = "继续持有；默认策略不主动卖出、不主动调仓"
            holds.append({"symbol": symbol, "name": names.get(symbol, symbol),
                          "model_value": current_value})
        else:
            action = "WAITING_FOR_FIRST_BUY"
            explanation = "尚未形成可执行持仓；等待下一可执行开盘"
        rows.append({
            "symbol": symbol, "name": names.get(symbol, symbol), "action": action,
            "model_units": units[symbol], "model_weight": current_weight,
            "target_weight": target_weight, "pending_value": pending_value,
            "blocked_reason": _BLOCK_REASON_TEXT.get(block["reason"], block["reason"]) if block_is_current else None,
            "explanation": explanation,
        })

    peak = result.equity.nav.cummax()
    max_drawdown = float((1 - result.equity.nav / peak).max())
    return {
        "status": "RETURN_FIRST_PRODUCTION_MODE",
        "strategy": "passive_ownership",
        "asof": str(result.equity.index[-1].date()),
        "earliest_action": "下一经过核验的交易日；仅供人工复核，不生成券商订单",
        "cash_balance": float(last.cash), "model_nav": float(last.nav),
        "model_exposure": float(last.exposure),
        "buy_candidates": buys, "sell_candidates": [], "holds": holds,
        "securities": rows,
        "risk": {
            "max_drawdown_in_replay": max_drawdown,
            "active_risk_control": False,
            "disclosure": "当前收益优先生产模式不主动止损、择时卖出或调仓；风险指标继续披露但不作为当前晋级阻塞。",
        },
        "warning": "回放使用复权经济单位，不等于真实股数或真实账户；执行前必须人工核对资金、持仓、公司行动、停牌、涨跌停和可卖数量。",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    verify = commands.add_parser('verify', help='verify an immutable evidence directory')
    verify.add_argument('path', type=Path)
    for command in ('audit', 'backtest', 'inspect'):
        p = commands.add_parser(command)
        p.add_argument('--data', type=Path, required=True)
        p.add_argument('--supplement', type=Path)
        p.add_argument('--catalog', type=Path, required=True)
        if command != 'audit':
            p.add_argument('--config', type=Path, default=Path('config/production.json'))
            scope = p.add_mutually_exclusive_group(required=True)
            scope.add_argument('--pool')
            scope.add_argument('--symbols')
            p.add_argument('--start', default='2023-01-03')
            p.add_argument('--end')
            if command == 'backtest':
                p.add_argument('--output', type=Path, required=True)
                p.add_argument('--benchmark', choices=('buy_hold', 'equal_weight'))
    a = parser.parse_args(argv)
    try:
        if a.command == 'verify':
            verify_evidence(a.path)
            response = {'integrity': 'PASS', 'economic_acceptance': 'NOT_IMPLIED'}
        else:
            catalog = json.loads(a.catalog.read_text(encoding='utf-8'))
            market = load_market(a.data, supplement=a.supplement, sectors=catalog['sectors'])
            if a.command == 'audit':
                response = {'quality': market.quality, 'sessions': len(market.calendar),
                    'start': str(market.calendar[0].date()), 'end': str(market.calendar[-1].date()),
                    'symbols': market.symbols, 'data_sha256': market.fingerprint(),
                    'provenance': market.provenance, 'economic_acceptance': 'NOT_IMPLIED'}
            else:
                cfg = _load_production_config(a.config)
                symbols = catalog['pools'][a.pool] if a.pool else [s.strip() for s in a.symbols.split(',')]
                # Only explicit members of the reviewed technology catalog are accepted.
                if set(symbols) - set(catalog['sectors']):
                    raise ValueError('symbol outside reviewed technology catalog')
                market = market.subset(symbols)
                benchmark = getattr(a, 'benchmark', None)
                if benchmark is None:
                    result = run_passive_ownership(market, cfg, start=a.start, end=a.end)
                else:
                    result = run(market, cfg, start=a.start, end=a.end, benchmark=benchmark)
                if a.command == 'backtest':
                    save_result(result, a.output)
                    response = {'status': ('RETURN_FIRST_PRODUCTION_MODE' if benchmark is None else 'RESEARCH_BENCHMARK'),
                                'strategy': result.metadata.get('strategy'), 'benchmark': benchmark,
                                'output': str(a.output), 'metrics': metrics(result),
                                'execution_config': {k: getattr(cfg, k) for k in sorted(_PRODUCTION_CONFIG_FIELDS)}}
                else:
                    if benchmark is not None:
                        raise ValueError('inspect is for the default passive production path; benchmark is not accepted')
                    response = inspection_report(market, result, catalog.get('names', {}))
        print(json.dumps(response, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(json.dumps({'status': 'ERROR', 'error': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

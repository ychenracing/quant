"""Offline research commands. No brokerage connection or executable order export."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .config import Config
from .data import Market, load_market
from .engine import Result, run
from .evidence import metrics, save_result, verify_evidence
from .features import build_features



_REASON_TEXT = {
    'WARMUP_OR_NO_FRESH_QUOTES': '有效历史不足或无新行情，不新增风险',
    'CROSS_SECTION_SHOCK': '组合内标的出现同步冲击，模型要求撤出风险',
    'PORTFOLIO_DRAWDOWN_SHOCK': '模型净值回撤伴随单日损失，要求降低风险',
    'BROAD_TREND_BREAKDOWN': '整体趋势和上涨广度恶化，降低模型仓位上限',
    'PORTFOLIO_WARNING': '模型净值进入回撤警戒，降低风险预算',
    'MARKET_WEAK_WARNING': '整体市场偏弱，但不机械否定仍完整的个股趋势',
    'TREND_OPEN': '未触发组合降险条件，仍需逐标的筛选',
    'CONFIRMED_RECOVERY': '连续观察到恢复条件，允许重新评估仓位',
    'RECOVERY_WAIT': '恢复确认不足，暂不放宽风险预算',
    'TREND_EXIT_OR_STALE': '持仓趋势退出或行情不够新鲜',
    'RISK_REDUCTION': '当前模型仓位超过允许风险预算',
    'SCHEDULED_SELECTION': '按统一周期和持仓迟滞规则筛选',
}


def inspection_report(market: Market, cfg: Config, result: Result, names: dict) -> dict:
    """Explain the replay book only; never infer the user's actual holdings.

    Features are rebuilt through the last measured close. A report for an earlier
    end date must not accidentally disclose signals computed from later data.
    """
    effective = market.prefix(result.equity.index[-1])
    features = build_features(effective, cfg)
    units = {symbol: 0. for symbol in effective.symbols}
    for order in result.orders:
        if order['status'] == 'FILLED':
            units[order['symbol']] += order['units'] * (1 if order['side'] == 'BUY' else -1)
    last = result.equity.iloc[-1]
    targets = result.targets.iloc[-1]
    rows = []
    for j, symbol in enumerate(effective.symbols):
        history = effective.frames[symbol].close
        current = units[symbol] * float(history.iloc[-1]) / float(last.nav) if len(history) else 0.
        target = float(targets[symbol])
        ready = bool(features.ready[-1, j])
        entry = bool(features.entry[-1, j])
        exit_signal = bool(features.exit[-1, j])
        if not ready:
            explanation = '有效行情或热身不足；不能新增，已有仓位需检查能否退出'
        elif exit_signal:
            explanation = '趋势退出条件已触发；能否成交仍取决于下一交易日开盘和流动性'
        elif target > current + 1e-8:
            explanation = '模型目标高于回放仓位；并非实际账户买入指令'
        elif target < current - 1e-8:
            explanation = '模型目标低于回放仓位；来源为组合降险、权重约束或重新筛选'
        elif current > 1e-8:
            explanation = '趋势未要求退出，目标变化未达到换仓条件，继续观察'
        else:
            explanation = ('有入场条件，但未获选或风险预算不足' if entry else '未满足入场条件，保持观察')
        score = float(features.score[-1, j])
        rows.append({'symbol': symbol, 'name': names.get(symbol, symbol),
                     'model_weight': current, 'target_weight': target,
                     'fresh_and_ready': ready, 'entry_condition': entry,
                     'exit_condition': exit_signal, 'score': score if abs(score) < float('inf') else None,
                     'explanation': explanation})
    reasons = str(last.reason).split('|')
    return {'status': 'RESEARCH_ONLY_NOT_ACCEPTED', 'asof': str(result.equity.index[-1].date()),
            'earliest_action': '下一经过核验的交易日；仅供人工复核，不生成券商订单',
            'reason': str(last.reason), 'explanations': [_REASON_TEXT.get(r, r) for r in reasons],
            'model_exposure': float(last.exposure), 'risk_budget': float(last.target_cap),
            'research_target_weights': targets.to_dict(), 'securities': rows,
            'warning': '模型回放不是实际账户；经济目标未通过；复权单位不是实际股数。'}

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
            p.add_argument('--config', type=Path, required=True)
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
                cfg = Config(**json.loads(a.config.read_text(encoding='utf-8')))
                symbols = catalog['pools'][a.pool] if a.pool else [s.strip() for s in a.symbols.split(',')]
                # Only explicit members of the reviewed technology catalog are accepted.
                if set(symbols) - set(catalog['sectors']):
                    raise ValueError('symbol outside reviewed technology catalog')
                market = market.subset(symbols)
                result = run(market, cfg, start=a.start, end=a.end,
                             benchmark=getattr(a, 'benchmark', None))
                if a.command == 'backtest':
                    save_result(result, a.output)
                    response = {'status': 'RESEARCH_ONLY_NOT_ACCEPTED', 'output': str(a.output),
                                'metrics': metrics(result), 'config': asdict(cfg)}
                else:
                    response = inspection_report(market, cfg, result, catalog.get('names', {}))
        print(json.dumps(response, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(json.dumps({'status': 'ERROR', 'error': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

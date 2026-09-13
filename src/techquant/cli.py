"""Offline research commands. No brokerage connection or executable order export."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .config import Config
from .data import load_market
from .engine import run
from .evidence import metrics, save_result, verify_evidence


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
                result = run(market.subset(symbols), cfg, start=a.start, end=a.end,
                             benchmark=getattr(a, 'benchmark', None))
                if a.command == 'backtest':
                    save_result(result, a.output)
                    response = {'status': 'RESEARCH_ONLY_NOT_ACCEPTED', 'output': str(a.output),
                                'metrics': metrics(result), 'config': asdict(cfg)}
                else:
                    last = result.equity.iloc[-1]
                    response = {'status': 'RESEARCH_ONLY_NOT_ACCEPTED',
                        'asof': str(result.equity.index[-1].date()),
                        'earliest_action': 'following verified trading session; human review only',
                        'reason': str(last.reason), 'model_exposure': float(last.exposure),
                        'risk_budget': float(last.target_cap),
                        'research_target_weights': result.targets.iloc[-1].to_dict(),
                        'warning': '模型回放不是实际账户；经济目标未通过；不生成券商订单或实盘股数。'}
        print(json.dumps(response, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(json.dumps({'status': 'ERROR', 'error': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

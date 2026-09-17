"""Read-only external comparator replay; never imported by the production package.

Only the input snapshot, universe, dates and starting capital are normalized.
Original policy, execution semantics and fee defaults remain native. Therefore
native results are NOT same-execution economic acceptance of the new strategy.
Run each invocation in its own process to isolate reference module namespaces.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, is_dataclass
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import traceback
import platform
import time

import numpy as np
import pandas as pd

from techquant.data import file_hash, load_market
from techquant.evidence import source_identity

DIRECTORIES = {'chatgpt': 'chatgpt/turtle_dual', 'trae': 'trae/glmcsm',
               'dumate': 'dumate/momentum_rotation', 'workbuddy': 'workbuddy/track_trend'}


def reference_files(root: Path) -> dict[str, str]:
    """Bind executable source and parameter files; never mutate the reference."""
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.suffix in {'.py', '.json', '.yaml', '.yml'}}


def index_files(root: Path) -> dict[str, str]:
    """Index turnover is part of the input, even when an engine ignores it."""
    return {name + '.csv': file_hash(root / (name + '.csv'))
            for name in ('sh000300', 'sh000682', 'sz399808')}


def module(name: str, file: Path):
    spec = importlib.util.spec_from_file_location(name, file)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def clean(value):
    if is_dataclass(value):
        return clean(asdict(value))
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, (float, int)):
        return value if np.isfinite(value) else None
    if isinstance(value, (str, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, value):
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                               allow_nan=False) + '\n', encoding='utf-8')


def native_input_frame(frame: pd.DataFrame, calendar: pd.DatetimeIndex,
                       reference: str) -> pd.DataFrame:
    """Adapt labels only; never invent native-reference price observations.

    WorkBuddy's native engine treats NaN bars as unavailable and leaves the
    corresponding sleeve in cash, but its panel builder intersects index labels
    before that logic. Reindexing to the frozen requested calendar exposes the
    already-native missing-bar behavior while preserving every observed value.
    """
    result = frame[['open', 'high', 'low', 'close', 'volume']].copy()
    result['amount'] = frame.raw_close * frame.volume
    if reference == 'workbuddy':
        result = result.reindex(calendar)
    return result


def install_chatgpt_semantic_noop_optimizations(native) -> dict[str, object]:
    """Remove repeated pure work without changing frozen policy semantics.

    The native ensemble asks for the same pure allocation score up to three
    times per sleeve/session. It also computes those scores for empty pending
    queues and when no new symbol requires portfolio admission. Cache only the
    exact same data-map object/session result and bypass only branches whose
    original score collection is provably unused.
    """
    sleeve_class = native.SleeveBacktestEngine
    ensemble_class = native.BacktestEngine
    original_scores = sleeve_class._allocation_scores
    original_execute = sleeve_class._execute_pending_signals
    original_authorize = ensemble_class._authorize_portfolio_buys

    def cached_scores(self, data_map, date):
        key = (id(data_map), pd.Timestamp(date).value)
        cached = getattr(self, '_native_harness_score_cache', None)
        if cached is not None and cached[0] == key:
            return cached[1]
        result = original_scores(self, data_map, date)
        self._native_harness_score_cache = (key, result)
        return result

    def execute_without_empty_score(
        self, pending, data_map, date, date_to_pos, directions=None
    ):
        if not pending:
            return []
        return original_execute(
            self, pending, data_map, date, date_to_pos, directions
        )

    def authorize_without_empty_ranking(self, states, date):
        held = self._held_portfolio_symbols(states)
        maximum = int(self.cfg['max_positions'])
        if len(held) > maximum:
            raise RuntimeError('portfolio symbol limit was already exceeded')
        has_new_candidate = any(
            signal.direction == 'buy'
            and signal.symbol not in held
            and signal.symbol in state.data_map
            and date in state.data_map[signal.symbol].index
            for state in states
            for signal, _ in state.pending
        )
        if has_new_candidate:
            return original_authorize(self, states, date)

        # This is the exact filtering branch of the frozen method when its
        # score_samples/ranked collections are empty. Order and audit events are
        # retained verbatim; only unused score construction is omitted.
        allowed = held
        date_str = date.strftime('%Y-%m-%d')
        for state in states:
            retained = []
            for signal, strategy in state.pending:
                if signal.direction == 'buy' and signal.symbol not in allowed:
                    state.sleeve._record_order_event(
                        date=date_str,
                        signal=signal,
                        event='rejected_portfolio_symbol_limit',
                        portfolio_max_positions=maximum,
                    )
                    continue
                retained.append((signal, strategy))
            state.pending = retained
        return None

    sleeve_class._allocation_scores = cached_scores
    sleeve_class._execute_pending_signals = execute_without_empty_score
    ensemble_class._authorize_portfolio_buys = authorize_without_empty_ranking
    return {
        'kind': 'semantic_noop_runtime_optimization',
        'allocation_score_cache': 'same sleeve data_map identity and session only',
        'empty_pending_bypass': True,
        'empty_new_candidate_ranking_bypass': True,
    }


def main() -> int:
    started = time.monotonic()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', choices=DIRECTORIES, required=True)
    p.add_argument('--reference-root', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--supplement', type=Path)
    p.add_argument('--indices', type=Path, help='frozen native-reference index CSV directory')
    p.add_argument('--catalog', type=Path, default=Path(__file__).with_name('catalog.json'))
    p.add_argument('--pool', required=True)
    p.add_argument('--end', default='2026-09-11')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    catalog = json.loads(a.catalog.read_text())
    symbols = catalog['pools'][a.pool]
    names = {s[2:]: catalog['names'][s] for s in symbols}
    market = load_market(a.data, supplement=a.supplement, sectors=catalog['sectors']).prefix(a.end)
    reference = a.reference_root / DIRECTORIES[a.reference]
    files = reference_files(reference)
    indices = a.indices or (a.data / 'qfq')
    identity = {'reference': a.reference, 'reference_commit': catalog['reference_commit'],
                'archive_commit': catalog['archive_commit'], 'source_files': files,
                'harness_sha256': file_hash(Path(__file__)), 'pool': a.pool,
                'producer': source_identity(), 'catalog_sha256': file_hash(a.catalog),
                'index_files': index_files(indices),
                'universe': symbols, 'data_sha256': market.subset(symbols).fingerprint(),
                'start': '2023-01-03', 'requested_end': a.end, 'initial_cash': 2_000_000.,
                'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                'seed': 34, 'classification': 'NATIVE_SEMANTICS_NOT_NORMALIZED_EXECUTION',
                'limitations': ['qfq input is not an actual-share corporate-action ledger',
                                'native execution costs and intraday rules differ across projects',
                                'native final liquidation and missing-date handling are preserved']}
    write_json(a.output / 'identity.json', identity)
    np.random.seed(34)
    random.seed(34)
    try:
        with tempfile.TemporaryDirectory(prefix='native-input-') as tmp, (a.output / 'native.log').open('w') as log:
            folder = Path(tmp)
            # Both identifier forms are input adapters only. Prices are not rewritten.
            for symbol, frame in market.frames.items():
                f = native_input_frame(frame, market.calendar, a.reference)
                for code in (symbol, symbol[2:], symbol[2:] + '_' + catalog['names'][symbol]):
                    f.to_csv(folder / f'{code}.csv', index_label='date')
            for symbol in ('sh000300', 'sh000682', 'sz399808'):
                f = pd.read_csv(indices / f'{symbol}.csv')
                f = f.loc[f.date <= a.end]
                for code in (symbol, symbol[2:]):
                    f.to_csv(folder / f'{code}.csv', index=False)
            sys.path.insert(0, str(reference))
            log.write('START_NATIVE_REPLAY\n'); log.flush()
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                if a.reference == 'chatgpt':
                    native = module('quant_fusion', reference / 'quant_fusion.py')
                    identity['harness_optimization'] = (
                        install_chatgpt_semantic_noop_optimizations(native)
                    )
                    write_json(a.output / 'identity.json', identity)
                    log.write('IMPORT_FINISHED\n'); log.flush()
                    engine = native.BacktestEngine(2_000_000.)
                    result = engine.run(names, '2023-01-03', a.end, data_dir=str(folder), indicator_state='cold')
                    frame = result['equity_curve']
                    curve = frame['assets']
                    trades = [vars(t) for t in result.get('trades', [])]
                    settings = {'indicator_state': 'cold', 'policy': result.get('policy'),
                                'effective_policy': result.get('effective_policy')}
                elif a.reference == 'workbuddy':
                    native = module('native_workbuddy', reference / 'quant_ai.py')
                    native.DATA_DIR = str(folder)
                    key = a.pool.removeprefix('workbuddy_')
                    profile = key if key in native.PROFILES else ('b' if len(symbols) == 5 else None)
                    settings = dict(native.DEFAULT_PARAMS)
                    settings.update(native.PROFILES.get(profile, {}))
                    result = native.run_backtest(symbols, '2023-01-03', a.end,
                                                 params=settings, init_cash=2_000_000., verbose=False)
                    curve = result['equity']
                    frame = pd.DataFrame({'assets': curve})
                    trades = [dict(zip(('date', 'symbol', 'side', 'price', 'quantity', 'reason'), t))
                              for t in result['trades']]
                elif a.reference == 'dumate':
                    native = module('native_dumate', reference / 'backtest_engine.py')
                    native.DATA_DIR = str(folder)
                    native.INITIAL_CAPITAL = 2_000_000.
                    settings = native.load_config(str(reference / 'config.yaml'))
                    settings.update(stock_list=list(names), start_date='2023-01-03', end_date=a.end)
                    engine = native.BacktestEngine(settings)
                    result, frame, trades = engine.run()
                    curve = frame['total_value']
                else:
                    from quant.data import BarData, align_dates
                    from quant.main import get_unified_params, POOL_PARAMS
                    from quant.strategy import PureMomentumStrategy
                    from quant.portfolio import PortfolioManager, PortfolioConfig
                    from quant.backtest import Backtester, BacktestConfig
                    pool_key = int(a.pool[5:]) if a.pool.startswith('trae_') else None
                    if pool_key in POOL_PARAMS:
                        sp, pp, bp = (dict(POOL_PARAMS[pool_key][part]) for part in ('strat', 'portfolio', 'backtest'))
                    else:
                        sp, pp, bp = get_unified_params(len(symbols))
                    bp['initial_capital'] = 2_000_000.
                    settings = {'strategy': sp, 'portfolio': pp, 'backtest': bp}
                    bars = {s[2:]: BarData(s[2:], names[s[2:]], pd.read_csv(folder / f'{s}.csv', parse_dates=['date']))
                            for s in symbols}
                    bars = align_dates(bars)
                    signals = PureMomentumStrategy(**sp).generate(bars)
                    weights = PortfolioManager(PortfolioConfig(**pp)).aggregate([signals])
                    result = Backtester(bars, BacktestConfig(**bp)).run(weights)
                    curve = result.equity_curve
                    frame = pd.DataFrame({'assets': curve})
                    trades = result.trades.to_dict('records')
                    # Signal-only diagnostic, not a normalized replay of embedded native risk.
                    weights.to_csv(a.output / 'pre_execution_signal_weights.csv')
                    result.weights.to_csv(a.output / 'native_position_weights.csv')
            curve.index = pd.DatetimeIndex(curve.index)
            if not np.isfinite(curve).all() or (curve <= 0).any():
                raise ValueError('native equity contains nonfinite or nonpositive values')
            expected = market.calendar
            covered = curve.index.equals(expected)
            path = np.r_[2_000_000., curve.to_numpy(dtype=float)]
            summary = {'status': 'NATIVE_REPLAY_MEASURED' if covered else 'INVALID_REQUESTED_COVERAGE',
                       'first': str(curve.index[0].date()), 'last': str(curve.index[-1].date()),
                       'sessions': len(curve), 'expected_sessions': len(expected),
                       'wealth': float(path[-1] / path[0]),
                       'max_drawdown': float((1 - path / np.maximum.accumulate(path)).max()),
                       'orders_native_rows': len(trades),
                       'acceptance': 'NOT_EQUIVALENT_EXECUTION'}
            curve.rename('nav').to_csv(a.output / 'equity.csv', index_label='date')
            frame.to_csv(a.output / 'native_frame.csv', index_label='date')
            pd.DataFrame(clean(trades)).to_csv(a.output / 'trades.csv', index=False)
            write_json(a.output / 'settings.json', settings)
            write_json(a.output / 'summary.json', summary)
    except Exception as exc:
        (a.output / 'failure.txt').write_text(traceback.format_exc())
        summary = {'status': 'NATIVE_REPLAY_FAILED', 'error': type(exc).__name__ + ': ' + str(exc)}
        write_json(a.output / 'summary.json', summary)
    if files != reference_files(reference) or identity['index_files'] != index_files(indices):
        raise RuntimeError('reference source changed during native replay')
    summary['elapsed_seconds'] = time.monotonic() - started
    write_json(a.output / 'summary.json', summary)
    write_json(a.output / 'manifest.json', {f.name: file_hash(f) for f in sorted(a.output.iterdir())})
    print(json.dumps({'reference': a.reference, 'pool': a.pool, **summary}, ensure_ascii=False))
    return 0 if summary['status'] == 'NATIVE_REPLAY_MEASURED' else 1


if __name__ == '__main__':
    raise SystemExit(main())

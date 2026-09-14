"""Finite, resumable expectation study; intentionally excluded from ordinary CI."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

from techquant.config import Config
from techquant.data import file_hash, load_market
from techquant.engine import run
from techquant.evidence import load_result, metrics, save_result, source_identity
from research.expectation import Owner, Parameters, forecast, replay


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def scopes(market, catalog):
    excluded = {'sz300308', 'sz300502', 'sz300394'}
    return {'union': list(market.symbols), 'chatgpt_5': catalog['pools']['chatgpt_5'],
            'joint_optical_leader_removal': [s for s in market.symbols if s not in excluded]}


def saved_run(market, output, cfg, *, params=None, prediction=None, benchmark=None,
              costs=1., delay=1):
    expected = {'config': asdict(cfg), 'universe': list(market.symbols), 'quality': market.quality,
        'data_sha256': market.fingerprint(), 'source': source_identity(), 'provenance': market.provenance,
        'delay': delay, 'cost_multiplier': costs, 'benchmark': benchmark,
        'start': str(market.calendar[0].date()), 'end': str(market.calendar[-1].date()),
        'economic_acceptance': 'UNVERIFIED', 'accounting': 'adjusted economic units, not actual shares',
        'runner_sha256': file_hash(Path(__file__))}
    if params is not None:
        expected['policy'] = Owner(market.symbols, prediction, params).identity()
    if output.exists():
        return load_result(output, expected=expected)
    result = (run(market, cfg, benchmark=benchmark, cost_multiplier=costs, delay=delay)
              if params is None else replay(market, params, prediction=prediction, config=cfg,
                                           cost_multiplier=costs, delay=delay))
    result.metadata['runner_sha256'] = file_hash(Path(__file__))
    if result.metadata != expected:
        raise AssertionError('run identity disagrees with frozen caller identity')
    save_result(result, output)
    return result


def select(market, catalog, output):
    training = market.prefix('2025-12-31')
    grid = [Parameters(h, ridge, slots) for h, ridge, slots in
            itertools.product((10, 20), (.1, 1., 10.), (2, 4))]
    scope_map = scopes(training, catalog)
    plan = {'family': 'adaptive_return_downside_expectation', 'selection_end': '2025-12-31',
        'candidate_count': len(grid), 'candidates': [asdict(p) for p in grid],
        'scopes': scope_map, 'source': source_identity(), 'runner_sha256': file_hash(Path(__file__)),
        'data_sha256': training.fingerprint(),
        'deficit_rule': 'max across scopes of positive log incumbent/candidate wealth, candidate/incumbent drawdown minus one, candidate/incumbent fills minus one',
        'tie_break': 'descending mean original log-relative buy-hold wealth minus .75 drawdown minus .002 annual fills above20; then fewer fills; then declared order'}
    plan_path = output / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('existing study plan belongs to another source/data/runner')
    write_json(plan_path, plan)
    baselines, buy_hold, cache = {}, {}, {}
    cfg, rows, objectives = Config(), [], []
    for scope, symbols in scope_map.items():
        subset = training.subset(symbols)
        baselines[scope] = metrics(saved_run(subset, output / 'runs' / ('incumbent_' + scope), cfg))
        buy_hold[scope] = metrics(saved_run(subset, output / 'runs' / ('buy_hold_' + scope), cfg, benchmark='buy_hold'))
    write_json(output / 'baselines.json', {'incumbent': baselines, 'buy_hold': buy_hold})
    for index, params in enumerate(grid):
        deficits, utility, fills = [], [], 0
        label = f'h{params.horizon}_ridge{params.shrinkage:g}_slots{params.positions}'
        for scope, symbols in scope_map.items():
            subset = training.subset(symbols)
            key = (scope, params.horizon, params.shrinkage)
            if key not in cache:
                cache[key] = forecast(subset, params)
                folder = output / 'forecasts'
                folder.mkdir(parents=True, exist_ok=True)
                name = f'{scope}_h{params.horizon}_ridge{params.shrinkage:g}'
                np.savez_compressed(folder / (name + '.npz'), expected_return=cache[key].expected_return,
                    adverse=cache[key].adverse, utility=cache[key].utility, ready=cache[key].ready)
                write_json(folder / (name + '.json'), {'fits': cache[key].fits,
                    'forecast_sha256': cache[key].fingerprint(), 'data_sha256': subset.fingerprint()})
            result = saved_run(subset, output / 'runs' / (label + '_' + scope), cfg,
                               params=params, prediction=cache[key])
            measured, incumbent = metrics(result), baselines[scope]
            relative = {'wealth': measured['wealth'] / incumbent['wealth'],
                        'drawdown': measured['max_drawdown'] / max(incumbent['max_drawdown'], 1e-12),
                        'fills': measured['orders'] / max(incumbent['orders'], 1)}
            deficit = max(0., -math.log(relative['wealth']), relative['drawdown'] - 1., relative['fills'] - 1.)
            objective = (math.log(measured['wealth'] / buy_hold[scope]['wealth']) -
                         .75 * measured['max_drawdown'] - .002 * max(0., measured['orders_per_year'] - 20.))
            deficits.append(deficit)
            utility.append(objective)
            fills += measured['orders']
            rows.append({'candidate': index, 'scope': scope, **asdict(params), **measured,
                         'worst_normalized_deficit': deficit, 'objective': objective})
            pd.DataFrame(rows).to_csv(output / 'trials.csv', index=False, float_format='%.17g')
        item = {'candidate': index, 'parameters': asdict(params), 'worst_deficit': max(deficits),
                'objective': float(np.mean(utility)), 'fills': fills}
        objectives.append(item)
        write_json(output / 'objectives.json', objectives)
        print(json.dumps(item), flush=True)
    selected = min(objectives, key=lambda r: (r['worst_deficit'], -r['objective'], r['fills'], r['candidate']))
    selected |= {'status': 'CORE_NONREGRESSION_ONLY' if selected['worst_deficit'] <= 1e-12 else 'DIAGNOSTIC_NOT_ACCEPTED',
        'source': source_identity(), 'runner_sha256': file_hash(Path(__file__)),
        'training_data_sha256': training.fingerprint(), 'plan_sha256': file_hash(plan_path),
        'economic_acceptance': 'UNVERIFIED'}
    write_json(output / 'selection.json', selected)
    return selected


def evaluate(market, catalog, selected_path, output):
    selected = json.loads(selected_path.read_text())
    if (selected['source'] != source_identity() or selected['runner_sha256'] != file_hash(Path(__file__))
            or selected['training_data_sha256'] != market.prefix('2025-12-31').fingerprint()
            or selected['plan_sha256'] != file_hash(selected_path.with_name('plan.json'))):
        raise ValueError('selection source/data/runner/plan identity changed')
    params = Parameters(**selected['parameters'])
    rows, cfg = [], Config()
    windows = {'full': (None, None), 'bull': ('2023-01-03', '2026-06-30'),
               'late_june_through_august': ('2026-06-22', '2026-08-31'),
               'july_august': ('2026-07-01', '2026-08-31'),
               'retrospective_2026': ('2026-01-01', None)}
    scope_map = scopes(market, catalog)
    write_json(output / 'plan.json', {'scopes': scope_map, 'windows': windows,
        'parameters': asdict(params), 'selection_sha256': file_hash(selected_path),
        'source': source_identity(), 'runner_sha256': file_hash(Path(__file__))})
    for scope, symbols in scope_map.items():
        subset = market.subset(symbols)
        prediction = forecast(subset, params)
        for policy in ('expectation', 'incumbent', 'buy_hold'):
            result = saved_run(subset, output / 'runs' / (scope + '_' + policy), cfg,
                params=params if policy == 'expectation' else None,
                prediction=prediction if policy == 'expectation' else None,
                benchmark='buy_hold' if policy == 'buy_hold' else None)
            for window, (start, end) in windows.items():
                rows.append({'scope': scope, 'policy': policy, 'window': window, **metrics(result, start, end)})
        print(scope, 'measured', flush=True)
        pd.DataFrame(rows).to_csv(output / 'matrix.csv', index=False, float_format='%.17g')
    write_json(output / 'status.json', {'status': 'MEASURED_NOT_UNIVERSAL_ACCEPTANCE',
        'runs': 9, 'rows': len(rows), 'source': source_identity(), 'runner_sha256': file_hash(Path(__file__))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--supplement', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection', type=Path)
    args = parser.parse_args()
    catalog = json.loads(Path(__file__).with_name('catalog.json').read_text())
    market = load_market(args.data, supplement=args.supplement, sectors=catalog['sectors'])
    before = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.selection:
        evaluate(market, catalog, args.selection, args.output)
    else:
        select(market, catalog, args.output)
    write_json(args.output / 'execution.json', {'elapsed_seconds': time.monotonic() - before,
        'source': source_identity(), 'runner_sha256': file_hash(Path(__file__))})


if __name__ == '__main__':
    main()

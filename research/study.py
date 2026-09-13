"""Finite reproducible retrospective study. Never executed by ordinary CI.

The selection protocol is committed separately before this runner is executed.
All candidates and failures survive; evaluation never changes the selected config.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import itertools
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

from techquant.config import Config
from techquant.data import file_hash, load_market
from techquant.engine import run
from techquant.evidence import load_result, metrics, save_result, source_identity


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def run_selection(market, catalog, protocol, output):
    training = market.prefix(protocol['selection_end'])
    leaders = {'sz300308', 'sz300502', 'sz300394'}
    scopes = {'union': list(market.symbols), 'chatgpt_5': catalog['pools']['chatgpt_5'],
              'joint_optical_leader_removal': [s for s in market.symbols if s not in leaders]}
    if list(scopes) != protocol['selection_scope'] or len(protocol['candidates']) != protocol['candidate_count']:
        raise ValueError('runner and frozen protocol disagree')
    rows, objectives, baselines = [], [], {}
    for name, symbols in scopes.items():
        result = run(training.subset(symbols), Config(**protocol['frozen_parameters_outside_grid']), benchmark='buy_hold')
        save_result(result, output / 'runs' / ('buy_hold_' + name))
        baselines[name] = metrics(result)
    for index, parameters in enumerate(protocol['candidates']):
        cfg = Config(**(protocol['frozen_parameters_outside_grid'] | parameters))
        score, orders = [], 0
        for scope, symbols in scopes.items():
            result = run(training.subset(symbols), cfg)
            save_result(result, output / 'runs' / f'candidate_{index:02}_{scope}')
            measured = metrics(result)
            value = (math.log(measured['wealth'] / baselines[scope]['wealth']) -
                     .75 * measured['max_drawdown'] - .002 * max(0., measured['orders_per_year'] - 20))
            rows.append({'candidate': index, 'scope': scope, 'objective': value,
                         'buy_hold_wealth': baselines[scope]['wealth'], **measured})
            score.append(value)
            orders += measured['orders']
        objectives.append({'candidate': index, 'objective': float(np.mean(score)), 'orders': orders})
        pd.DataFrame(rows).to_csv(output / 'candidates.csv', index=False, float_format='%.17g')
        write_json(output / 'objectives.json', objectives)
        print(json.dumps(objectives[-1]), flush=True)
    best = sorted(objectives, key=lambda x: (-x['objective'], x['orders'], x['candidate']))[0]
    config = asdict(Config(**(protocol['frozen_parameters_outside_grid'] | protocol['candidates'][best['candidate']])))
    selected = {'status': 'SELECTED_NOT_ECONOMICALLY_ACCEPTED', 'candidate': best['candidate'],
                'objective': best['objective'], 'config': config, 'protocol_sha256': file_hash(Path(__file__).with_name('protocol.json')),
                'selection_data_sha256': training.fingerprint(), 'source': source_identity(),
                'runner_sha256': file_hash(Path(__file__))}
    write_json(output / 'selection.json', selected)
    write_json(output / 'selected_config.json', config)
    return selected


def evaluate(market, catalog, protocol, selection, output, batch_size=20):
    if selection.get('protocol_sha256') != file_hash(Path(__file__).with_name('protocol.json')):
        raise ValueError('selection protocol changed')
    if selection.get('source') != source_identity():
        raise ValueError('selection source/runtime identity changed')
    if selection.get('runner_sha256') != file_hash(Path(__file__)):
        raise ValueError('selection runner identity changed')
    if selection.get('selection_data_sha256') != market.prefix(protocol['selection_end']).fingerprint():
        raise ValueError('selection data identity changed')
    index = selection.get('candidate')
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(protocol['candidates']):
        raise ValueError('selection candidate is outside the frozen grid')
    frozen = asdict(Config(**(protocol['frozen_parameters_outside_grid'] | protocol['candidates'][index])))
    if selection.get('config') != frozen:
        raise ValueError('selection configuration differs from the frozen candidate')
    cfg = Config(**frozen)
    pools = dict(catalog['pools'])
    pools['union'] = list(market.symbols)
    cases = [(name, 'original_pool', symbols, 1., 1, cfg) for name, symbols in pools.items()]
    leaders = {'sz300308', 'sz300502', 'sz300394'}
    cases.append(('remove_optical_leaders', 'joint_removal', [s for s in market.symbols if s not in leaders], 1., 1, cfg))
    for s in market.symbols:
        cases.append(('single_' + s, 'single', [s], 1., 1, cfg))
        cases.append(('remove_' + s, 'leave_one_out', [n for n in market.symbols if n != s], 1., 1, cfg))
    for i, sector in enumerate(sorted(set(market.sectors.values()))):
        symbols = [s for s in market.symbols if market.sectors[s] != sector]
        cases.append((f'remove_sector_{i}', 'sector_removal', symbols, 1., 1, cfg))
    rng = np.random.default_rng(protocol['seed'])
    for size in (2, 3, 5, 8, 13, 22, 26, 33):
        for repeat in range(6):
            symbols = sorted(rng.choice(market.symbols, size=size, replace=False).tolist())
            cases.append((f'sample_{size}_{repeat}', 'sampled_subset', symbols, 1., 1, cfg))
    common = catalog['pools'].get('chatgpt_5', [])
    for size in range(1, len(common) + 1):
        for index, symbols in enumerate(itertools.combinations(common, size)):
            cases.append((f'common_exhaustive_{size}_{index}', 'common_five_exhaustive', list(symbols), 1., 1, cfg))
    # Cover every remaining cardinality; keep the original seeded samples unchanged.
    extra_rng = np.random.default_rng(protocol['seed'] + 1)
    for size in range(2, len(market.symbols)):
        if size not in (2, 3, 5, 8, 13, 22, 26, 33):
            symbols = sorted(extra_rng.choice(market.symbols, size=size, replace=False).tolist())
            cases.append((f'cardinality_{size}', 'cardinality_coverage', symbols, 1., 1, cfg))
    for label, costs, delay in (('double_cost', 2., 1), ('triple_cost', 3., 1), ('delayed_open', 1., 2)):
        cases.append((label, 'stress', list(market.symbols), costs, delay, cfg))
    for parameter in ('fast', 'slow', 'rebalance', 'stop_atr', 'risk_drawdown'):
        for factor in (.9, 1.1):
            value = getattr(cfg, parameter) * factor
            if isinstance(getattr(cfg, parameter), int):
                value = max(1, round(value))
            adjusted = Config(**(asdict(cfg) | {parameter: value}))
            cases.append((f'neighbor_{parameter}_{factor}', 'stability_not_reselection', list(market.symbols), 1., 1, adjusted))
    # Scope/case generation is frozen before the first evaluation run.
    write_json(output / 'case_plan.json', [{'name': n, 'group': g, 'symbols': s, 'cost': c,
                                          'delay': d, 'config': asdict(f)} for n,g,s,c,d,f in cases])
    rows = []
    replay_count = 0
    new_cases = 0
    completed = True
    for case_index, (name, group, symbols, costs, delay, config) in enumerate(cases):
        subset = market.subset(symbols)
        created = False
        policies = ('strategy', 'buy_hold', 'equal_weight') if group == 'original_pool' else ('strategy', 'buy_hold')
        for policy in policies:
            replay_count += 1
            destination = output / 'runs' / f'{name}_{policy}'
            benchmark = None if policy == 'strategy' else policy
            expected = {'config': asdict(config), 'universe': list(subset.symbols), 'quality': subset.quality,
                        'data_sha256': subset.fingerprint(), 'source': source_identity(),
                        'provenance': subset.provenance, 'delay': delay, 'cost_multiplier': costs,
                        'benchmark': benchmark, 'start': str(subset.calendar[0].date()),
                        'end': str(subset.calendar[-1].date()), 'economic_acceptance': 'UNVERIFIED',
                        'accounting': 'adjusted economic units, not actual shares'}
            if destination.exists():
                result = load_result(destination, expected=expected)
            else:
                created = True
                result = run(subset, config, cost_multiplier=costs, delay=delay, benchmark=benchmark)
                save_result(result, destination)
            for window, (start, end) in protocol['time_windows'].items():
                rows.append({'case': name, 'group': group, 'policy': policy, 'window': window,
                             'universe_size': len(symbols), **metrics(result, start, end)})
            if name == 'union' and policy == 'strategy':
                # Prefix equivalence uses the frozen full-snapshot prices, not historical vendor revisions.
                cut = subset.calendar[500]
                prefix = run(subset.prefix(cut), config)
                pd.testing.assert_frame_equal(result.equity.loc[:cut], prefix.equity, check_freq=False)
                pd.testing.assert_frame_equal(result.targets.loc[:cut], prefix.targets, check_freq=False)
                write_json(output / 'historical_prefix.json', {'status': 'PASS', 'cut': str(cut.date()),
                    'limitation': 'same snapshot causality, not proof of point-in-time corporate-action inputs'})
                # Economic contribution of actual research units, including terminal marks and cash fees.
                contribution = {s: 0. for s in symbols}
                units = {s: 0. for s in symbols}
                for order in result.orders:
                    if order['status'] != 'FILLED':
                        continue
                    s = order['symbol']; direction = 1 if order['side'] == 'BUY' else -1
                    contribution[s] -= direction * order['notional'] + order['fee']
                    units[s] += direction * order['units']
                for s in symbols:
                    contribution[s] += units[s] * float(subset.frames[s].close.iloc[-1])
                ranked = sorted(contribution, key=lambda s: (-contribution[s], s))
                write_json(output / 'contribution.json', {'pnl': contribution, 'ranked_symbols': ranked})
                # Ex-post leaders are a diagnostic stress, not a new optimization universe.
                cases.append(('remove_realized_top3', 'ex_post_contributor_removal',
                              [s for s in symbols if s not in ranked[:3]], 1., 1, cfg))
        pd.DataFrame(rows).to_csv(output / 'matrix.csv', index=False, float_format='%.17g')
        if case_index % 10 == 0:
            print(f'evaluation {case_index + 1}/{len(cases)} {name}', flush=True)
        new_cases += int(created)
        if new_cases >= batch_size and case_index + 1 < len(cases):
            completed = False
            break
    write_json(output / 'case_plan_resolved.json', [{'name': n, 'group': g, 'symbols': s, 'cost': c,
        'delay': d, 'config': asdict(f)} for n,g,s,c,d,f in cases])
    if not completed:
        progress = {'status':'PARTIAL', 'completed_cases':case_index+1, 'planned_cases':len(cases)}
        write_json(output/'progress.json', progress)
        return progress
    frame = pd.DataFrame(rows)
    full = frame[(frame.window == 'full') & ~frame.group.isin(['stress', 'stability_not_reselection'])]
    comparison = full.pivot(index=['case', 'group'], columns='policy', values=['wealth','max_drawdown','orders'])
    ratio = comparison['wealth']['strategy'] / comparison['wealth']['buy_hold']
    summary = {'status': 'FINITE_RETROSPECTIVE_DIAGNOSTICS_NOT_UNIVERSAL_ACCEPTANCE',
               'case_count': len(cases), 'replay_count': replay_count, 'matched_buy_hold_cases': len(comparison),
               'wealth_at_least_buy_hold': int((ratio >= 1).sum()), 'minimum_wealth_ratio': float(ratio.min()),
               'median_wealth_ratio': float(ratio.median()), 'worst_case': str(ratio.idxmin()),
               'both_return_and_drawdown_dominate': int(((ratio >= 1) &
                   (comparison['max_drawdown']['strategy'] <= comparison['max_drawdown']['buy_hold'])).sum()),
               'economic_acceptance': 'NOT_ESTABLISHED'}
    write_json(output / 'summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['select', 'evaluate'])
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--supplement', type=Path)
    p.add_argument('--selection', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--batch-size', type=int, default=20)
    a = p.parse_args()
    if a.batch_size < 1 or (a.resume and a.mode != 'evaluate'):
        raise ValueError('invalid resume or batch settings')
    a.output.mkdir(parents=True, exist_ok=a.resume)
    started = time.monotonic()
    root = Path(__file__).parent
    catalog = json.loads((root/'catalog.json').read_text())
    protocol = json.loads((root/'protocol.json').read_text())
    market = load_market(a.data, supplement=a.supplement, sectors=catalog['sectors'])
    if str(market.calendar[-1].date()) != protocol['evaluation_end']:
        raise ValueError('snapshot and frozen evaluation end differ')
    run_identity = {'source':source_identity(), 'runner_sha256':file_hash(Path(__file__)),
         'protocol_sha256':file_hash(root/'protocol.json'), 'catalog_sha256':file_hash(root/'catalog.json'),
         'data_sha256':market.fingerprint(), 'provenance':market.provenance, 'mode':a.mode}
    if a.resume:
        previous = json.loads((a.output/'identity.json').read_text())
        if previous != run_identity:
            raise ValueError('resume identity mismatch')
        write_json(a.output/('runner_' + run_identity['runner_sha256'] + '.json'), run_identity)
    else:
        write_json(a.output/'identity.json', run_identity)
    if a.mode == 'select':
        result = run_selection(market, catalog, protocol, a.output)
    else:
        if a.selection is None:
            raise ValueError('evaluation requires the frozen selection file')
        result = evaluate(market, catalog, protocol, json.loads(a.selection.read_text()), a.output, a.batch_size)
    write_json(a.output/'execution.json', {'elapsed_seconds':time.monotonic()-started, 'completed':result.get('status') != 'PARTIAL'})
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

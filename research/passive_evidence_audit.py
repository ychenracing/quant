"""Verify existing passive/comparator accounts; never run or select a strategy.

Reuse is conditional on the exact reviewed adapter and shared package hashes.
It establishes historical account equivalence, not economic/production approval.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from techquant.config import Config
from techquant.data import file_hash
from techquant.evidence import load_result, metrics, source_identity
from techquant.execution import fee, round_quantity
from research.fresh_challenger_formal_validation import build_case_plan, plan_sha256

PASSIVE_SOURCE = '4335d3d7e57585a6b80f024f7f60e9b1b41fc10f'
FORMAL_SOURCE = 'a0705156e82c083f580aa4ebfe52383f5dffdb8d'
CANONICAL_SOURCE = 'e1b0d8c0473c5f3c365d29de9a95fe8280d96765'
ADAPTER_SHA256 = '4c4e3fd3db409292bce3269804f1d9d7e935289691a6e1803c2864bae4efd080'
DATA_SHA256 = 'd9ded1d933f292933aa09145bc36a5696421c43dbf266fce018a34ef0b9f3f9b'
ECONOMIC_FILES = ('equity.csv', 'targets.csv', 'orders.csv', 'metrics.json')
ACCOUNT_FIELDS = ('config', 'universe', 'quality', 'data_sha256', 'provenance',
                  'delay', 'cost_multiplier', 'start', 'end', 'accounting')
CASH_TOLERANCE = 1e-6  # Existing engine's cash invariant; not fitted to differences.


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def source_matches(old: dict, current: dict):
    require(all(current['files'].get(n) == h for n, h in old['files'].items()),
            'economic package changed: historical reuse requires a new proof')
    calculated = hashlib.sha256(json.dumps(old['files'], sort_keys=True).encode()).hexdigest()
    require(calculated == old['package_sha256'], 'source file-map digest mismatch')


def exact_files(left: Path, right: Path):
    for name in ECONOMIC_FILES:
        require((left / name).read_bytes() == (right / name).read_bytes(),
                f'economic file differs: {left.name}/{name}')


def restore(root: Path):
    identity = read(root / 'identity.json')
    return load_result(root, expected=identity)


def verify_manifest(root: Path):
    manifest = read(root / 'MANIFEST.json')
    for name, digest in manifest.items():
        require(file_hash(root / name) == digest, 'archive manifest mismatch: ' + name)
    return len(manifest)


def audit(passive: Path, shards: list[Path], canonical: Path, out: Path):
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).parent
    current = source_identity()
    require(current['files'].get('passive.py') == ADAPTER_SHA256, 'unreviewed passive adapter')
    manifest_count = verify_manifest(passive)
    canonical_manifest_count = verify_manifest(canonical)
    receipt = read(passive / 'receipt.json')
    require(receipt['source_commit'] == PASSIVE_SOURCE and receipt['run_id'] == '35174627146',
            'passive generation identity mismatch')
    require(read(canonical / 'receipt.json')['source_commit'] == CANONICAL_SOURCE,
            'canonical control generation mismatch')
    catalog, protocol = read(root / 'catalog.json'), read(root / 'protocol.json')
    contract = read(root / 'campaign_peak_formal_validation_contract.json')
    windows = contract['time_windows']
    plan = read(shards[0] / 'case_plan.json')
    full_symbols = next(c['symbols'] for c in plan if c['name'] == 'union')
    require(build_case_plan(full_symbols, catalog['sectors'], catalog, protocol) == plan,
            'fixed 199-case inventory differs')
    digest = plan_sha256(plan)
    require(len(plan) == 199 and [c['index'] for c in plan] == list(range(199)), 'case coverage')
    by_index = {}
    for shard in shards:
        identity, status = read(shard / 'identity.json'), read(shard / 'status.json')
        index = identity['shard_index']
        require(index not in by_index and status['status'] == 'SHARD_COMPLETE', 'bad shard')
        require(identity['source']['commit'] == FORMAL_SOURCE, 'formal source mismatch')
        require(identity['data_sha256'] == DATA_SHA256, 'full frozen data mismatch')
        require(identity['plan_sha256'] == digest and status['plan_sha256'] == digest, 'plan hash')
        require(read(shard / 'case_plan.json') == plan, 'shard inventory mismatch')
        require(read(shard / 'shard_plan.json') == [c for c in plan if c['index'] % 4 == index],
                'shard assignment mismatch')
        for key, name in [('catalog_sha256', 'catalog.json'), ('protocol_sha256', 'protocol.json'),
                          ('runner_sha256', 'campaign_peak_formal_validation.py'),
                          ('formal_contract_sha256', 'campaign_peak_formal_validation_contract.json')]:
            require(identity[key] == file_hash(root / name), 'changed formal input: ' + name)
        by_index[index] = shard
    require(set(by_index) == {0, 1, 2, 3}, 'incomplete shards')
    measured_source = read(passive / 'evaluation/runs/union_passive/identity.json')['source']
    source_matches(measured_source, current)
    case_map, window_rows, budget_findings = [], [], []
    max_cash_error, max_nav_identity_error = 0., 0.
    for case in plan:
        shard = by_index[case['index'] % 4]
        account = shard / 'runs' / (case['name'] + '__buy_hold')
        result = restore(account)
        ident = result.metadata
        require(ident['source']['commit'] == FORMAL_SOURCE, 'wrong comparator generation')
        source_matches(ident['source'], current)
        require(all(ident['source'][key] == measured_source[key]
                    for key in ('python', 'platform', 'dependencies')), 'generation runtime differs')
        require(ident['config'] == asdict(Config()), 'configuration mismatch')
        require(ident['universe'] == sorted(case['symbols']) and ident['benchmark'] == 'buy_hold',
                'universe or comparator identity mismatch')
        require(ident['delay'] == case['delay'] and ident['cost_multiplier'] == case['cost_multiplier'],
                'execution case mismatch')
        require((ident['start'], ident['end']) == ('2023-01-03', '2026-09-11'), 'account window')
        eq, orders = result.equity, result.orders
        require(len(eq) == 896 and float(eq.cash.iloc[0]) == 2e6 and float(eq.holdings.iloc[0]) == 0,
                'initial account is not empty cash CNY 2m')
        require(list(result.targets.columns) == ident['universe'] and result.targets.index.equals(eq.index),
                'target account coverage mismatch')
        require(metrics(result) == read(account / 'metrics.json'), 'saved account metrics differ')
        fills = [o for o in orders if o['status'] == 'FILLED']
        daily_cost, spent = {}, {}
        for order in fills:
            require(order['side'] == 'BUY' and order['units'] > 0, 'passive sell or invalid units')
            require(eq.index.get_loc(pd.Timestamp(order['date'])) - eq.index.get_loc(
                pd.Timestamp(order['signal_date'])) == case['delay'], 'execution clock mismatch')
            quantity = order['raw_quantity_equivalent']
            require(quantity == round_quantity(order['symbol'], quantity), 'board-lot mismatch')
            charge = fee(order['notional'], order['side'], order['date'],
                         ident['config']['commission_bps'], case['cost_multiplier'])
            require(abs(charge - order['fee']) <= CASH_TOLERANCE, 'fee ledger mismatch')
            total = order['notional'] + order['fee']
            daily_cost[order['date']] = daily_cost.get(order['date'], 0.) + total
            spent[order['symbol']] = spent.get(order['symbol'], 0.) + total
        reconstructed = 2e6 - pd.Series([daily_cost.get(str(d.date()), 0.) for d in eq.index],
                                        index=eq.index).cumsum()
        cash_error = float((reconstructed - eq.cash).abs().max())
        nav_error = float((eq.nav - eq.cash - eq.holdings).abs().max())
        require(cash_error <= CASH_TOLERANCE and nav_error <= CASH_TOLERANCE, 'account reconciliation')
        require(float(eq.cash.min()) >= -CASH_TOLERANCE, 'cash insolvency')
        max_cash_error, max_nav_identity_error = max(max_cash_error, cash_error), max(max_nav_identity_error, nav_error)
        budget = 2e6 / len(case['symbols'])
        excess = {s: total - budget for s, total in spent.items() if total > budget + CASH_TOLERANCE}
        if excess:
            budget_findings.append({'case': case['name'], 'original_budget': budget, 'excess': excess})
        for name, bounds in windows.items():
            window_rows.append({'case': case['name'], 'window': name, **metrics(result, *bounds)})
        raw = read(account / 'manifest.json')['files']
        case_map.append({**case, 'effective_universe': ident['universe'],
            'run_id': 35122063992, 'attempt': 1, 'source_sha': FORMAL_SOURCE,
            'data_sha256': ident['data_sha256'], 'config': ident['config'],
            'environment': ident['source'], 'identity_sha256': file_hash(account / 'identity.json'),
            'raw_archive_member': str(account.relative_to(shard.parent)), 'raw_files_sha256': raw,
            'start': ident['start'], 'end': ident['end'], 'initial_cash': 2e6, 'initial_holdings': {},
            'initial_per_symbol_budget': budget, 'removed_name_budget': 'CNY_2m_divided_by_supplied_case_universe',
            'window_semantics': 'FULL_ACCOUNT_SLICE_WITH_INHERITED_POSITIONS_AND_PREVIOUS_CLOSE_BASE',
            'cash_reconciliation_max_error': cash_error, 'nav_identity_max_error': nav_error,
            'fill_rows': len(fills), 'order_attempt_rows': len(orders),
            'fill_days': len({o['date'] for o in fills}), 'sells': 0,
            'shared_call_chain_equivalence': 'VERIFIED',
            'strict_all_in_sleeve_budget': 'FAILED' if excess else 'NO_VIOLATION_OBSERVED'})
    comparisons, core, exact_pairs = [], {}, []
    for scope, alias in [('union', 'union'), ('chatgpt_5', 'chatgpt_5'),
                         ('joint_optical_leader_removal', 'remove_optical_leaders')]:
        case = next(c for c in plan if c['name'] == alias)
        historic = by_index[case['index'] % 4] / 'runs' / (alias + '__buy_hold')
        for horizon in ('selection', 'evaluation'):
            b = passive / horizon / 'runs' / (scope + '_buy_hold')
            p = passive / horizon / 'runs' / (scope + '_passive')
            exact_files(b, p)
            pi, bi = read(p / 'identity.json'), read(b / 'identity.json')
            require(all(pi[k] == bi[k] for k in ACCOUNT_FIELDS), 'adapter changes economic identity')
            require(pi['benchmark'] is None and pi['strategy'] == 'passive_ownership' and
                    pi['economic_semantics'] == 'same_engine_buy_hold', 'adapter identity')
            if horizon == 'evaluation':
                exact_files(p, historic)
                hi = read(historic / 'identity.json')
                require(all(pi[k] == hi[k] for k in ACCOUNT_FIELDS), 'cross-run inputs differ')
            exact_pairs.append({'scope': scope, 'horizon': horizon, 'economic_files': 'BYTE_IDENTICAL',
                                'cross_run_historical_exact': horizon == 'evaluation'})
        p = restore(passive / 'selection/runs' / (scope + '_passive'))
        cpath = canonical / 'selection/runs' / (scope + '_treatment')
        control = restore(cpath)
        require(all(control.metadata[k] == p.metadata[k] for k in ACCOUNT_FIELDS), 'canonical pair inputs differ')
        require(control.metadata['policy']['name'] == 'offensive_campaign_peak_authority' and
                control.metadata['policy']['parameters'] == {'enabled': True}, 'wrong canonical control')
        require(control.metadata['policy']['implementation_sha256'] ==
                file_hash(root / 'offensive_campaign_peak_authority.py'), 'canonical policy source differs')
        comparisons.append({'scope': scope, 'passive': metrics(p), 'canonical': metrics(control),
                            'canonical_identity_sha256': file_hash(cpath / 'identity.json')})
        full = restore(passive / 'evaluation/runs' / (scope + '_passive'))
        core[scope] = {window: metrics(full, *bounds) for window, bounds in windows.items()}
    pobj = [r['passive']['wealth'] for r in comparisons]
    cobj = [r['canonical']['wealth'] for r in comparisons]
    pmean, cmean = [math.fsum(math.log(v) for v in values) / 3 for values in (pobj, cobj)]
    screen = {'rows': comparisons, 'passive_mean_log': pmean, 'canonical_mean_log': cmean,
              'passive_minimum': min(pobj), 'canonical_minimum': min(cobj),
              'alpha_promotion_passes': pmean > cmean + 1e-12 and min(pobj) >= min(cobj) - 1e-12,
              'invalidated_claim': '35174627146 active label was dominant_peak, not registered campaign_peak',
              'original_active_account_status': 'VALID_OTHER_CONTROL_NOT_CANONICAL_SCREEN',
              'economic_acceptance': 'NOT_MET'}
    worst_excess = max((v for row in budget_findings for v in row['excess'].values()), default=0.)
    summary = {'status': 'EQUIVALENCE_VERIFIED_NOT_ACCEPTED', 'candidate_source': current,
        'auditor_sha256': file_hash(Path(__file__)), 'historical_passive_source': PASSIVE_SOURCE,
        'historical_formal_source': FORMAL_SOURCE, 'canonical_source': CANONICAL_SOURCE,
        'frozen_data_sha256': DATA_SHA256, 'seed': 34, 'case_plan_sha256': digest,
        'formal_case_plan_file_sha256': file_hash(shards[0] / 'case_plan.json'),
        'passive_manifest_files': manifest_count, 'canonical_manifest_files': canonical_manifest_count,
        'comparator_cases_verified': 199, 'comparator_raw_files_verified': 995,
        'window_metrics_reaggregated': len(window_rows), 'economic_replays': 0,
        'core_exact_pairs': exact_pairs, 'canonical_alpha_screen': screen,
        'cash_reconciliation_max_error': max_cash_error, 'nav_identity_max_error': max_nav_identity_error,
        'strict_budget_violating_cases': len(budget_findings), 'maximum_sleeve_excess_cny': worst_excess,
        'core_passive_metrics': core, 'economic_acceptance': 'NOT_MET', 'merge_eligible': False,
        'not_inherited': ['active treatment/control acceptance', 'old active prefix checks',
            'independent cash-start subwindows', 'native/equivalent-execution superiority',
            'actual-share corporate-action accounting', 'active risk recovery or theme switching'],
        'proof_scope': 'Exact reviewed metadata-only adapter delegates every covered account input to byte-identical shared economic code. Original accounts retain original generating identities.'}
    write(out / 'summary.json', summary)
    write(out / 'case-map.json', case_map)
    write(out / 'canonical-screen-correction.json', screen)
    write(out / 'budget-violations.json', budget_findings)
    pd.DataFrame(window_rows).to_csv(out / 'passive-window-metrics.csv', index=False, float_format='%.17g')
    write(out / 'MANIFEST.json', {p.name: file_hash(p) for p in sorted(out.iterdir())})
    print(json.dumps({k: summary[k] for k in ('status', 'comparator_cases_verified',
        'window_metrics_reaggregated', 'economic_replays', 'strict_budget_violating_cases',
        'maximum_sleeve_excess_cny', 'cash_reconciliation_max_error', 'merge_eligible')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--passive', type=Path, required=True)
    parser.add_argument('--formal', type=Path, nargs=4, required=True)
    parser.add_argument('--canonical', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.passive, args.formal, args.canonical, args.output)

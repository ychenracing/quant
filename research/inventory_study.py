"""Twelve fixed-parameter intent replays, with actual-inventory loss attribution."""
from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path
import tarfile
import time
import numpy as np
import pandas as pd
from techquant.config import Config
from techquant.data import file_hash, load_market
from techquant.engine import run
from techquant.evidence import load_result, metrics, save_result, source_identity
from research.expectation_study import scopes, write_json
from research.inventory_intent import UnitIntent
from research.ledger_attribution import attribute


ROOT = Path(__file__).parent
WINDOWS = {'full': (None, None), 'through_2025': (None, '2025-12-31'),
           'bull': (None, '2026-06-30'),
           'late_june_through_august': ('2026-06-22', '2026-08-31'),
           'july_august': ('2026-07-01', '2026-08-31'),
           'retrospective_2026': ('2026-01-01', None)}


def identity():
    paths = ['inventory_study.py', 'inventory_intent.py', 'ledger_attribution.py',
             'inventory_intent_contract.json', 'expectation_study.py',
             'leadership.py', 'admission.py', 'catalog.json']
    return {'source': source_identity(), 'research': {p: file_hash(ROOT / p) for p in paths}}


def execute(data, supplement, output, historical=None):
    contract = json.loads((ROOT / 'inventory_intent_contract.json').read_text())
    catalog = json.loads((ROOT / 'catalog.json').read_text())
    market = load_market(data, supplement=supplement, sectors=catalog['sectors'])
    if market.fingerprint() != contract['data_sha256']:
        raise ValueError('frozen market identity mismatch')
    plan = {'identity': identity(), 'contract': contract, 'windows': WINDOWS,
            'data_sha256': market.fingerprint(), 'economic_acceptance': 'UNVERIFIED'}
    # JSON round-trip makes tuple/list representation stable across restarts.
    plan = json.loads(json.dumps(plan))
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'plan.json').exists() and json.loads((output / 'plan.json').read_text()) != plan:
        raise ValueError('existing paired study has another identity')
    write_json(output / 'plan.json', plan)
    rows, attribution, equivalence = [], {}, []
    for family, values in contract['parameters'].items():
        module = importlib.import_module('research.' + family)
        parameters = module.Parameters(**values)
        for scope, symbols in scopes(market, catalog).items():
            subset = market.subset(symbols)
            for mode in ('weights', 'units'):
                key = f'{family}_{scope}_{mode}'
                path = output / 'runs' / key
                def factory(m, cfg):
                    owner = module.Owner(m, parameters)
                    return UnitIntent(m, owner) if mode == 'units' else owner
                from dataclasses import asdict
                expected = dict(config=asdict(Config()), universe=list(subset.symbols),
                    quality=subset.quality, data_sha256=subset.fingerprint(), source=source_identity(),
                    provenance=subset.provenance, delay=1, cost_multiplier=1., benchmark=None,
                    start=str(subset.calendar[0].date()), end=str(subset.calendar[-1].date()),
                    economic_acceptance='UNVERIFIED', accounting='adjusted economic units, not actual shares',
                    policy=factory(subset, Config()).identity(), paired_identity=identity())
                if path.exists():
                    result = load_result(path, expected=expected)
                else:
                    result = run(subset, policy_factory=factory)
                    result.metadata['paired_identity'] = identity()
                    if result.metadata != expected:
                        raise AssertionError('unexpected paired replay identity')
                    save_result(result, path)
                for window, (start, end) in WINDOWS.items():
                    rows.append(dict(family=family, scope=scope, mode=mode, window=window,
                                     **metrics(result, start, end)))
                daily, episodes, summary = attribute(subset, result)
                dest = output / 'attribution' / key
                dest.mkdir(parents=True, exist_ok=True)
                daily.to_csv(dest / 'daily.csv', index=False, float_format='%.17g')
                episodes.to_csv(dest / 'episodes.csv', index=False, float_format='%.17g')
                write_json(dest / 'summary.json', summary)
                attribution[key] = summary
                if historical is not None and mode == 'weights':
                    old = historical / family / 'evaluation' / 'runs' / f'{scope}_{family}'
                    expected = json.loads((old / 'identity.json').read_text())
                    past = load_result(old, expected=expected)
                    if (past.metadata['policy'] != result.metadata['policy'] or
                            past.metadata['data_sha256'] != subset.fingerprint()):
                        raise ValueError('historical policy/data differs; cannot claim equivalence')
                    difference = float(np.max(np.abs(result.equity.nav.to_numpy() - past.equity.nav.to_numpy())))
                    equivalence.append(dict(family=family, scope=scope,
                        historical_source=past.metadata['source'], max_nav_difference=difference,
                        same_orders=result.orders == past.orders,
                        equal_curve=bool(difference <= 1e-6)))
                pd.DataFrame(rows).to_csv(output / 'matrix.csv', index=False, float_format='%.17g')
                write_json(output / 'attribution.json', attribution)
                write_json(output / 'historical_equivalence.json', equivalence)
                print(key, json.dumps(metrics(result)), flush=True)
    write_json(output / 'status.json', dict(identity=identity(), runs=12,
        status='PAIRED_MEASUREMENT_NOT_ECONOMIC_ACCEPTANCE', economic_acceptance='UNVERIFIED'))


def package(output):
    files = {str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob('*'))
             if p.is_file() and p.name != 'MANIFEST.json'}
    write_json(output / 'MANIFEST.json', files)
    archive = output.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as target:
        target.add(output, arcname='evidence')
    archive.with_suffix(archive.suffix + '.sha256').write_text(file_hash(archive) + '  ' + archive.name + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--supplement', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--historical', type=Path)
    args = parser.parse_args()
    before = time.monotonic()
    try:
        execute(args.data, args.supplement, args.output, args.historical)
    except Exception as exc:
        write_json(args.output / 'failure.json', {'error': repr(exc), 'identity': identity()})
        raise
    finally:
        write_json(args.output / 'execution.json', {'identity': identity(), 'seconds': time.monotonic()-before})
        package(args.output)


if __name__ == '__main__':
    main()

"""Run the registered ownership comparison using verified issued forecasts only.

The parent archive remains immutable evidence of its original source. Restoring
its arrays is not refitting a model, subsetting a larger universe, or relabelling
an old economic result as a result of the current implementation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tarfile
import time
from types import SimpleNamespace

import numpy as np

from techquant.data import file_hash, load_market
from techquant.evidence import source_identity
from research import coherent
from research.expectation_study import write_json
from research.finite_study import Study
from research.inventory_study import package
from research.pathwise import Prediction

ORIGIN_SOURCE = '0b5626993598d6cbf8bf0af15345e01d18a730e7'
ORIGIN_TREE = 'c9fae2f68fd91b4eb86ef10fb816f4d4b02a8eed'
ORIGIN_RUN = '34809876451'
ARCHIVE_SHA256 = 'cf2eb81f4fb9b8cdba1de8929798883ac01c8cb7025925891b1ccaa529fc61a0'
ARRAYS = ('expected', 'tail', 'ready', 'price', 'ema10', 'ema20', 'ema60',
          'momentum5', 'ret1', 'outcome_probability')


def restore_prediction(market, arrays, receipt):
    dates = [str(d.date()) for d in market.calendar]
    shape = (len(dates), len(market.symbols))
    if (receipt['source']['commit'] != ORIGIN_SOURCE
            or receipt['data_sha256'] != market.fingerprint()
            or receipt['symbols'] != list(market.symbols) or receipt['dates'] != dates):
        raise ValueError('issued forecast source, universe, data or calendar mismatch')
    if set(arrays) != set(ARRAYS) or any(
            arrays[k].shape != (shape + (3,) if k == 'outcome_probability' else shape) for k in ARRAYS):
        raise ValueError('issued forecast array shape mismatch')
    if (not np.isfinite(arrays['expected']).all() or not np.isfinite(arrays['tail']).all()
            or ((arrays['tail'] < 0) | (arrays['tail'] > 1)).any()):
        raise ValueError('issued forecasts must have finite utility and valid probability')
    previous = -1
    for fit in receipt['fits']:
        i = fit['session']
        if (not previous < i < len(dates) or fit['date'] != dates[i]
                or fit['last_label_session'] > i
                or fit['last_feature_session'] + 20 != fit['last_label_session']
                or fit['task'] != 'first_passage' or fit['loss_barrier'] != .12):
            raise ValueError('forecast fit clock or first-passage contract mismatch')
        previous = i
    prediction = Prediction(market.symbols, market.fingerprint(), 20,
                            fits=receipt['fits'], **arrays)
    if prediction.fingerprint() != receipt['forecast_sha256']:
        raise ValueError('issued forecast content identity mismatch')
    for key in ARRAYS:
        getattr(prediction, key).setflags(write=False)
    return prediction


class ForecastStore:
    def __init__(self, archive: Path, destination: Path):
        if file_hash(archive) != ARCHIVE_SHA256:
            raise ValueError('immutable parent archive hash mismatch')
        self.root = destination / 'nonlinear'
        with tarfile.open(archive) as source:
            original_manifest = source.extractfile('nonlinear/MANIFEST.json').read()
            if not self.root.exists():
                destination.mkdir(parents=True, exist_ok=True)
                source.extractall(destination, filter='data')
        if (self.root / 'MANIFEST.json').read_bytes() != original_manifest:
            raise ValueError('extracted parent manifest identity mismatch')
        manifest = json.loads(original_manifest)
        for name, digest in manifest.items():
            path = self.root / name
            if not path.resolve().is_relative_to(self.root.resolve()) or file_hash(path) != digest:
                raise ValueError('parent evidence manifest mismatch: ' + name)
        if ((self.root / 'source-commit.txt').read_text().strip() != ORIGIN_SOURCE
                or (self.root / 'source-tree.txt').read_text().strip() != ORIGIN_TREE):
            raise ValueError('parent source snapshot identity mismatch')
        receipt = json.loads((self.root / 'receipt.json').read_text())
        if receipt['source_commit'] != ORIGIN_SOURCE or str(receipt['run_id']) != ORIGIN_RUN:
            raise ValueError('parent generation identity mismatch')
        self.records, self.cache = {}, {}
        directory = Path(__file__).parent
        for phase in ('selection', 'evaluation'):
            for identity_path in sorted((self.root / phase / 'runs').glob('*/identity.json')):
                identity = json.loads(identity_path.read_text())
                policy = identity.get('policy', {})
                if (policy.get('name') != 'first_passage_expectation'
                        or policy.get('parameters') != {'horizon': 20, 'loss_barrier': .12}):
                    continue
                if identity['source']['commit'] != ORIGIN_SOURCE:
                    raise ValueError('original forecast generator source mismatch')
                # The current owner interprets the same immutable representation.
                # Do not silently reinterpret a receipt using changed parent code.
                dependencies = identity['study']['dependencies']
                for name in ('pathwise.py', 'nonlinear.py', 'expectation.py', 'observed_trend.py'):
                    if file_hash(directory / name) != dependencies[name]:
                        raise ValueError('frozen forecast dependency changed: ' + name)
                data_id, forecast_id = identity['data_sha256'], policy['forecast_sha256']
                path = self.root / phase / 'forecasts' / data_id / (forecast_id + '.npz')
                record = json.loads(path.with_suffix('.json').read_text())
                if (record['source'] != identity['source'] or record['fits'] != policy['fits']
                        or record['forecast_sha256'] != forecast_id
                        or file_hash(path) != record['archive_sha256']):
                    raise ValueError('original forecast receipt/archive mismatch')
                if data_id in self.records:
                    raise ValueError('ambiguous exact-market forecast')
                self.records[data_id] = (path, record)
        self.manifest_count = len(manifest)

    def load(self, market):
        key = market.fingerprint()
        if key not in self.records:
            raise ValueError('no exact issued forecast; subset/refit fallback is forbidden')
        if key not in self.cache:
            path, receipt = self.records[key]
            # Recheck the file at consumption, not just at store construction.
            if file_hash(path) != receipt['archive_sha256']:
                raise ValueError('issued forecast changed before consumption')
            with np.load(path, allow_pickle=False) as saved:
                arrays = {key: saved[key].copy() for key in saved.files}
            self.cache[key] = restore_prediction(market, arrays, receipt)
        return self.cache[key]

    def identity(self):
        return {'source': ORIGIN_SOURCE, 'tree': ORIGIN_TREE, 'run': ORIGIN_RUN,
                'archive_sha256': ARCHIVE_SHA256, 'verified_files': self.manifest_count,
                'mode': 'exact_issued_forecasts_no_refit'}


class CoherentStudy(Study):
    def __init__(self, store):
        self.family, self.store = 'coherent', store
        # Instance-local factory injection reuses the frozen finite runner without
        # altering its selection rule or replacing functions in any global module.
        self.module = SimpleNamespace(Parameters=coherent.Parameters, grid=coherent.grid,
            Owner=lambda market, params: coherent.Owner(market, params, prediction=store.load(market)))

    def saved(self, market, path, parameters=None, benchmark=None, costs=1., delay=1):
        result = super().saved(market, path, parameters, benchmark, costs, delay)
        if parameters is not None:
            prediction = self.store.load(market)
            packet = path.parent.parent / 'forecasts' / market.fingerprint() / (prediction.fingerprint() + '.json')
            record = json.loads(packet.read_text())
            original = self.store.records[market.fingerprint()][1]
            origin = {'source': original['source'], 'archive_sha256': original['archive_sha256'],
                      'parent': self.store.identity()}
            if 'generation_origin' in record and record['generation_origin'] != origin:
                raise ValueError('preserved forecast generation origin changed')
            # The ordinary source field identifies this preservation, not a new
            # fit. Explicitly retain the original generator beside it.
            record['generation_origin'] = origin
            record['mode'] = 'reused_original_issue_not_refitted'
            write_json(packet, record)
        return result

    def identity(self):
        root = Path(__file__).parent
        names = ('coherent.py', 'coherent_contract.json', 'coherent_study.py',
                 'finite_study.py', 'expectation_study.py', 'inventory_study.py',
                 'pathwise.py', 'nonlinear.py', 'expectation.py', 'observed_trend.py',
                 'catalog.json', 'requirements.txt')
        return {'source': source_identity(), 'family': self.family,
                'dependencies': {name: file_hash(root / name) for name in names},
                'forecast_origin': self.store.identity()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--supplement', type=Path, required=True)
    parser.add_argument('--parent-archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started, study = time.monotonic(), None
    try:
        store = ForecastStore(args.parent_archive, args.output.parent / 'issued-forecast-origin')
        study = CoherentStudy(store)
        tree = subprocess.check_output(['git', 'write-tree'], text=True).strip()
        with (args.output / 'source.tar.gz').open('wb') as stream:
            subprocess.run(['git', 'archive', '--format=tar.gz', tree], stdout=stream, check=True)
        write_json(args.output / 'source.json', {'source': source_identity(), 'tree': tree,
                   'study': study.identity()})
        catalog = json.loads(Path(__file__).with_name('catalog.json').read_text())
        market = load_market(args.data, supplement=args.supplement, sectors=catalog['sectors'])
        contract = json.loads(Path(__file__).with_name('coherent_contract.json').read_text())
        if (market.fingerprint() != contract['data_sha256']
                or [{'ranking': p.ranking} for p in coherent.grid()] != contract['candidates']):
            raise ValueError('frozen data or preregistered candidates mismatch')
        study.select(market, catalog, args.output / 'selection')
        study.evaluate(market, catalog, args.output / 'selection/selection.json', args.output / 'evaluation')
        selected = json.loads((args.output / 'selection/selection.json').read_text())
        write_json(args.output / 'receipt.json', {'study': study.identity(),
            'selection_status': selected['status'], 'selected': selected['parameters'],
            'economic_acceptance': 'NOT_PASSED' if selected['worst_deficit'] > 1e-12 else 'UNVERIFIED',
            'status': 'MEASURED_NOT_UNIVERSAL_ACCEPTANCE'})
    except Exception as exc:
        write_json(args.output / 'failure.json', {'error': repr(exc), 'source': source_identity()})
        raise
    finally:
        write_json(args.output / 'execution.json', {'seconds': time.monotonic() - started,
            'source': source_identity(), 'study': study.identity() if study is not None else None})
        package(args.output)


if __name__ == '__main__':
    main()

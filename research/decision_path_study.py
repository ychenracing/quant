"""Measure the declared six path selectors with one exact shadow cache per pool."""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
from techquant.config import Config
from techquant.data import file_hash, load_market
from techquant.engine import run
from techquant.evidence import load_result, save_result, source_identity
from research.decision_paths import (Owner, Parameters, grid, Admission, AdmissionParameters,
                                    Leadership, LeadershipParameters)
from research.finite_study import Study
from research.expectation_study import write_json
from research.inventory_study import package


class DecisionStudy(Study):
    def __init__(self, root: Path):
        self.family = 'decision_paths'
        self.module = SimpleNamespace(Parameters=Parameters, grid=grid)
        self.root, self.cache = root, {}

    def identity(self):
        root = Path(__file__).parent
        return {'source': source_identity(), 'family': self.family, 'dependencies': {
            name: file_hash(root/name) for name in ('decision_paths.py', 'decision_path_study.py',
            'decision_path_contract.json', 'finite_study.py', 'expectation_study.py',
            'inventory_study.py', 'admission.py', 'leadership.py', 'catalog.json')}}

    def expected(self, market, benchmark=None, policy=None):
        result = dict(config=asdict(Config()), universe=list(market.symbols), quality=market.quality,
            data_sha256=market.fingerprint(), source=source_identity(), provenance=market.provenance,
            delay=1, cost_multiplier=1., benchmark=benchmark,
            start=str(market.calendar[0].date()), end=str(market.calendar[-1].date()),
            economic_acceptance='UNVERIFIED', accounting='adjusted economic units, not actual shares')
        if policy is not None:
            result['policy'] = policy
        return result

    def shadows(self, market):
        key = market.fingerprint()
        if key not in self.cache:
            results = {}
            for name in ('incumbent', 'leadership', 'admission', 'buy_hold'):
                factory = None
                if name == 'leadership':
                    factory = lambda m,c: Leadership(m, LeadershipParameters(60,20,4,False))
                if name == 'admission':
                    factory = lambda m,c: Admission(m, AdmissionParameters('responsive',20,.6))
                benchmark = 'buy_hold' if name == 'buy_hold' else None
                expected = self.expected(market, benchmark,
                                         factory(market, Config()).identity() if factory else None)
                path = self.root/'shadows'/key/name
                if path.exists():
                    results[name] = load_result(path, expected=expected)
                else:
                    results[name] = run(market, policy_factory=factory, benchmark=benchmark)
                    if results[name].metadata != expected:
                        raise AssertionError('shadow identity mismatch')
                    save_result(results[name], path)
            self.cache[key] = results
        return self.cache[key]

    def saved(self, market, path, parameters=None, benchmark=None, costs=1., delay=1):
        if costs != 1. or delay != 1:
            raise ValueError('this declared screening study does not change execution stresses')
        shadows = self.shadows(market)
        owner = Owner(market, parameters, shadows) if parameters is not None else None
        expected = self.expected(market, benchmark, owner.identity() if owner else None)
        expected['study'] = self.identity()
        score_path = path.parent.parent/'decisions'/(path.name+'.json')
        if path.exists():
            result = load_result(path, expected=expected)
            if owner is not None:
                receipt = json.loads(score_path.read_text())
                if (receipt['identity'] != expected or len(receipt['observations']) != len(market.calendar)
                    or receipt['observations_sha256'] != self.score_hash(receipt['observations'])):
                    raise ValueError('cached decision receipt identity mismatch')
            return result
        if owner is None:
            result = deepcopy(shadows['buy_hold' if benchmark == 'buy_hold' else 'incumbent'])
        else:
            result = run(market, policy_factory=lambda m,c: owner)
        result.metadata['study'] = self.identity()
        if result.metadata != expected:
            raise AssertionError('path-selector result identity mismatch')
        if owner is not None:
            write_json(score_path, dict(identity=expected, observations=owner.observations,
                       observations_sha256=self.score_hash(owner.observations)))
        save_result(result, path)
        return result

    @staticmethod
    def score_hash(observations):
        import hashlib
        return hashlib.sha256(json.dumps(observations, sort_keys=True,
                              allow_nan=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--supplement', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    study = DecisionStudy(args.output)
    started = time.monotonic()
    try:
        catalog = json.loads(Path(__file__).with_name('catalog.json').read_text())
        market = load_market(args.data, supplement=args.supplement, sectors=catalog['sectors'])
        if market.fingerprint() != 'd9ded1d933f292933aa09145bc36a5696421c43dbf266fce018a34ef0b9f3f9b':
            raise ValueError('frozen input identity mismatch')
        study.select(market, catalog, args.output/'selection')
        study.evaluate(market, catalog, args.output/'selection/selection.json', args.output/'evaluation')
        write_json(args.output/'status.json', dict(identity=study.identity(),
            status='MEASURED_NOT_ECONOMIC_ACCEPTANCE', economic_acceptance='UNVERIFIED'))
    except Exception as exc:
        write_json(args.output/'failure.json', dict(error=repr(exc), identity=study.identity()))
        raise
    finally:
        write_json(args.output/'execution.json', dict(identity=study.identity(), seconds=time.monotonic()-started))
        package(args.output)


if __name__ == '__main__':
    main()

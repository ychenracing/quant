"""Reuse one pinned forecast experiment without refitting or relabelling its source.

A new policy's evidence retains both its own identity and the issuing source.
Predictions for an excluded-symbol run must have been trained on that exact pool.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time
import urllib.error
import urllib.request
import numpy as np
from techquant.data import file_hash
from techquant.evidence import source_identity
from research.nonlinear import Owner as Validator, Parameters, runtime
from research.pathwise import Prediction

SOURCE = '0b5626993598d6cbf8bf0af15345e01d18a730e7'
RUN = 34809876451
MANIFEST_SHA256 = '161210529e6e8aef1044bc6b7a0b8811bf4f1b680afd8f2a9ad2107e4a54bff2'
ARCHIVE_SHA256 = 'cf2eb81f4fb9b8cdba1de8929798883ac01c8cb7025925891b1ccaa529fc61a0'


def retrieve(root: Path) -> Path:
    """Bounded, verified, atomic transport; cached valid archives are not fetched."""
    root.mkdir(parents=True, exist_ok=True)
    archive = root / 'issued.tar.gz'
    if not archive.is_file() or file_hash(archive) != ARCHIVE_SHA256:
        temporary = archive.with_suffix('.partial')
        url = ('https://raw.githubusercontent.com/ychenracing/quant/research/evidence/nonlinear/'
               f'{SOURCE}/{RUN}/nonlinear-evidence.tar.gz')
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=60) as src, temporary.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
                if file_hash(temporary) != ARCHIVE_SHA256:
                    raise ValueError('issued forecast evidence archive mismatch')
                temporary.replace(archive)
                break
            except (OSError, ValueError) as error:
                temporary.unlink(missing_ok=True)
                permanent = isinstance(error, urllib.error.HTTPError) and error.code not in {408,429,500,502,503,504}
                if permanent or attempt == 2:
                    raise
                time.sleep(attempt + 1)
    with tarfile.open(archive) as bundle:
        bundle.extractall(root, filter='data')
    return root / 'nonlinear'


def load_payload(market, path: Path, forecast_id: str) -> Prediction:
    receipt = json.loads(path.with_suffix('.json').read_text())
    if (receipt['data_sha256'] != market.fingerprint()
            or receipt['symbols'] != list(market.symbols)
            or receipt['dates'] != [str(d.date()) for d in market.calendar]
            or receipt['forecast_sha256'] != forecast_id
            or file_hash(path) != receipt['archive_sha256']):
        raise ValueError('issued forecast market/calendar/archive identity mismatch')
    for fit in receipt['fits']:
        if (not 0 <= fit['last_feature_session'] < fit['session'] < len(market.calendar)
                or fit['last_label_session'] != fit['last_feature_session'] + 20
                or fit['last_label_session'] > fit['session']):
            raise ValueError('issued forecast contains an immature training receipt')
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    prediction = Prediction(market.symbols, market.fingerprint(), 20, **arrays, fits=receipt['fits'])
    Validator(market, Parameters(20, .5, 4), prediction=prediction)
    if prediction.fingerprint() != forecast_id:
        raise ValueError('issued forecast contents mismatch')
    for array in arrays.values():
        array.setflags(write=False)
    return prediction


class IssuedForecasts:
    def __init__(self, root: Path):
        self.root = root
        if file_hash(root / 'MANIFEST.json') != MANIFEST_SHA256:
            raise ValueError('pinned parent manifest mismatch')
        manifest = json.loads((root / 'MANIFEST.json').read_text())
        for relative, expected in manifest.items():
            if file_hash(root / relative) != expected:
                raise ValueError('parent evidence file mismatch: ' + relative)
        if (root / 'source-commit.txt').read_text().strip() != SOURCE:
            raise ValueError('parent issuing source mismatch')
        # Issued arrays can be reused only while their feature/learner definitions
        # are unchanged. Policy-only changes do not require a model refit.
        with tarfile.open(root / 'source.tar.gz') as bundle:
            for name in ('research/pathwise.py', 'research/nonlinear.py', 'research/expectation.py',
                         'src/techquant/data.py'):
                member = bundle.extractfile(name)
                if member is None or hashlib.sha256(member.read()).hexdigest() != file_hash(Path(name)):
                    raise ValueError('issuing feature/model source no longer equivalent: ' + name)
        self.cache = {}
        self.records = []
        for section in ('selection', 'evaluation'):
            for path in sorted((root / section / 'runs').glob('*/identity.json')):
                record = json.loads(path.read_text())
                if record.get('policy', {}).get('parameters') == {'horizon':20, 'loss_barrier':.12}:
                    self.records.append((section, record))

    def identity(self):
        return {'issuing_source': SOURCE, 'issuing_run': RUN,
                'parent_manifest_sha256': MANIFEST_SHA256, 'parent_archive_sha256': ARCHIVE_SHA256,
                'model_refitted': False}

    def load(self, market):
        fingerprint = market.fingerprint()
        if fingerprint in self.cache:
            return self.cache[fingerprint]
        matches = [(section, r) for section, r in self.records if r['data_sha256'] == fingerprint
                   and r['universe'] == list(market.symbols)]
        if len(matches) != 1:
            raise ValueError('no unique exact-universe issued forecast; refusing to slice or refit')
        section, record = matches[0]
        own_source = source_identity()
        if (record['source']['commit'] != SOURCE
                or record['source']['python'] != own_source['python']
                or record['source']['dependencies'] != own_source['dependencies']
                or record['policy']['ownership']['inner']['learner_runtime'] != runtime()):
            raise ValueError('issued model runtime identity mismatch')
        forecast_id = record['policy']['forecast_sha256']
        path = self.root / section / 'forecasts' / fingerprint / (forecast_id + '.npz')
        forecast = load_payload(market, path, forecast_id)
        origin = dict(self.identity(), forecast_sha256=forecast_id,
                      payload_sha256=file_hash(path), source=record['source'])
        self.cache[fingerprint] = forecast, origin
        return forecast, origin

"""Attribute a pinned observed-trend run using its issued forecasts, never refits."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request
import numpy as np
import pandas as pd
from techquant.data import file_hash
from research.expectation import matured_labels, _features
from research.expectation_study import write_json

PARENT = 'acd87232cc97518c6258338351a7489fb7ddc1f9/34808183806'
ARCHIVE_SHA256 = '15679c4e9a0e9fa597be057cfe472794fd194456d51bf94cb64c7acd5621dbf6'


def retrieve_parent(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    archive = root / 'parent.tar.gz'
    url = ('https://raw.githubusercontent.com/ychenracing/quant/research/evidence/nonlinear/'
           + PARENT + '/nonlinear-evidence.tar.gz')
    with urllib.request.urlopen(url, timeout=60) as response:
        archive.write_bytes(response.read())
    if file_hash(archive) != ARCHIVE_SHA256:
        raise ValueError('parent archive identity mismatch')
    with tarfile.open(archive) as bundle:
        bundle.extractall(root, filter='data')
    return root / 'nonlinear'


def diagnose(root: Path, market, out: Path):
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    for name, expected in manifest.items():
        if file_hash(root / name) != expected:
            raise ValueError('parent raw evidence integrity mismatch: ' + name)
    selected = json.loads((root / 'selection/selection.json').read_text())['candidate']
    summary = []
    for scope in ('union', 'chatgpt_5', 'joint_optical_leader_removal'):
        identity = json.loads((root / 'selection/runs' / f'candidate{selected}_{scope}' / 'identity.json').read_text())
        m = market.subset(identity['universe']).prefix('2025-12-31')
        if m.fingerprint() != identity['data_sha256']:
            raise ValueError('diagnostic data identity mismatch')
        forecast_id = identity['policy']['inner']['forecast_sha256']
        path = root / 'selection/forecasts' / m.fingerprint() / (forecast_id + '.npz')
        receipt = json.loads(path.with_suffix('.json').read_text())
        if file_hash(path) != receipt['archive_sha256']:
            raise ValueError('issued forecast archive mismatch')
        with np.load(path, allow_pickle=False) as saved:
            f = {key: saved[key].copy() for key in saved.files}
        digest = hashlib.sha256()
        for key in ('expected', 'tail', 'ready'):
            digest.update(np.asarray(f[key], dtype='<f8').tobytes())
        if digest.hexdigest() != forecast_id:
            raise ValueError('issued forecast contents mismatch')
        fits = receipt['fits']
        return_start = min(r['session'] for r in fits if r['task'] == 'return')
        tail_start = min(r['session'] for r in fits if r['task'] == 'tail')
        *_, prior = _features(m)
        labels = matured_labels(m.panel('close').to_numpy(), m.panel('open').to_numpy(), horizon=60, cutoff=len(m.calendar)-1)
        rows = []
        for i in range(len(labels)):
            good = f['ready'][i] & np.isfinite(labels[i, :, 0])
            if good.sum() < 3:
                continue
            y = labels[i, good, 0]
            if np.std(y) < 1e-12:
                continue
            for method, values in (('learner', f['expected']), ('price_prior', prior)):
                x = values[i, good]
                if np.std(x) < 1e-12:
                    continue
                best = np.argsort(-x, kind='stable')[:min(4, len(x))]
                rows.append({'date': str(m.calendar[i].date()), 'method': method, 'model_fitted': i >= return_start,
                    'count': int(good.sum()), 'rank_correlation': float(pd.Series(x).rank().corr(pd.Series(y).rank())),
                    'top4_logreturn': float(y[best].mean()), 'universe_logreturn': float(y.mean())})
        out.mkdir(parents=True, exist_ok=True)
        rows = pd.DataFrame(rows)
        rows.to_csv(out / (scope + '_ranks.csv'), index=False, float_format='%.17g')
        record = {'scope': scope, 'source': identity['source'], 'forecast_sha256': forecast_id,
                  'model_fit_start': return_start, 'rank_metrics': [], 'risk_reliability': []}
        for fitted_only in (False, True):
            for method in ('learner', 'price_prior'):
                subset = rows[(rows.method == method) & (rows.model_fitted | (not fitted_only))]
                record['rank_metrics'].append({'method': method, 'fitted_only': fitted_only, 'dates': len(subset),
                    'mean_daily_rank_correlation': float(subset.rank_correlation.mean()),
                    'mean_top4_excess_log_return': float((subset.top4_logreturn-subset.universe_logreturn).mean())})
        labels = matured_labels(m.panel('close').to_numpy(), m.panel('open').to_numpy(), horizon=5, cutoff=len(m.calendar)-1)
        size = len(labels); good = f['ready'][:size] & np.isfinite(labels).all(axis=-1)
        for fitted_only in (False, True):
            for low, high in ((0., .1), (.1, .25), (.25, .5), (.5, 1.000001)):
                mask = good & (f['tail'][:size] >= low) & (f['tail'][:size] < high)
                if fitted_only:
                    mask[:tail_start] = False
                if mask.any():
                    record['risk_reliability'].append({'fitted_only': fitted_only, 'low': low, 'high': high,
                        'samples': int(mask.sum()), 'predicted': float(f['tail'][:size][mask].mean()),
                        'observed': float((labels[:, :, 1][mask] >= .08).mean())})
        summary.append(record)
    write_json(out / 'summary.json', {'status': 'DIAGNOSTIC_NOT_ECONOMIC_ACCEPTANCE',
        'parent_archive_sha256': ARCHIVE_SHA256, 'parent_manifest_files': len(manifest),
        'method': 'Issued predictions only; overlapping mature labels are descriptive, not independent observations.',
        'scopes': summary})
    return summary

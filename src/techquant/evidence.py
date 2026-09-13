"""Metrics, immutable run output and source/data/configuration binding."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .data import file_hash

if TYPE_CHECKING:
    from .engine import Result


def source_identity() -> dict:
    root = Path(__file__).parent
    files = {str(p.relative_to(root)): file_hash(p) for p in sorted(root.glob('*.py'))}
    return {'commit': os.environ.get('QUANT_SOURCE_COMMIT', 'UNBOUND_LOCAL_SNAPSHOT'),
            'package_sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            'files': files, 'python': platform.python_version(), 'platform': platform.platform(),
            'dependencies': {name: importlib.metadata.version(name) for name in ('numpy', 'pandas')}}


def metrics(result: Result, start: str | None = None, end: str | None = None) -> dict:
    """Positive drawdown magnitude, net mark-to-market wealth, no forced liquidation.

    Subperiod return includes its first day's movement versus the prior measured
    close. Both local-period drawdown and inherited-peak drawdown are reported.
    """
    all_eq = result.equity
    eq = all_eq.loc[start:end]
    if eq.empty:
        return {'status': 'NO_COVERAGE'}
    position = all_eq.index.get_loc(eq.index[0])
    base = (float(all_eq.nav.iloc[position - 1]) if position else
            float(result.metadata['config']['initial_cash']))
    nav = eq.nav.to_numpy(dtype=float)
    path = np.r_[base, nav]
    returns = path[1:] / path[:-1] - 1
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1 / 252)
    wealth = nav[-1] / base
    drawdown = 1 - path / np.maximum.accumulate(path)
    inherited = 1 - eq.nav / all_eq.nav.cummax().loc[eq.index]
    fills = [o for o in result.orders if o['status'] == 'FILLED' and
             str(eq.index[0].date()) <= o['date'] <= str(eq.index[-1].date())]
    fees = sum(o['fee'] for o in fills)
    turnover = sum(o['notional'] / float(all_eq.loc[o['date'], 'nav']) for o in fills)
    sigma = float(np.std(returns, ddof=0))
    return {'status': 'MEASURED', 'start': str(eq.index[0].date()), 'end': str(eq.index[-1].date()),
            'sessions': len(eq), 'wealth': float(wealth), 'total_return': float(wealth - 1),
            'cagr': float(wealth ** (1 / years) - 1), 'max_drawdown': float(drawdown.max()),
            'inherited_drawdown': float(inherited.max()),
            'sharpe_zero_rf': float(np.mean(returns) / sigma * np.sqrt(252)) if sigma > 0 else 0.,
            'orders': len(fills), 'orders_per_year': float(len(fills) / years),
            'gross_turnover': float(turnover), 'fees': float(fees),
            'slippage': float(sum(o['slippage'] for o in fills)),
            'average_exposure': float(eq.exposure.mean()),
            'blocked_attempts': sum(o['status'] == 'BLOCKED' and
                str(eq.index[0].date()) <= o['date'] <= str(eq.index[-1].date()) for o in result.orders)}


def save_result(result: Result, destination: str | Path) -> Path:
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.pending-', dir=destination.parent) as temp:
        root = Path(temp)
        result.equity.to_csv(root / 'equity.csv', float_format='%.12g')
        result.targets.to_csv(root / 'targets.csv', float_format='%.12g', index_label='date')
        pd.DataFrame(result.orders).to_csv(root / 'orders.csv', index=False, float_format='%.12g')
        for name, value in (('identity.json', result.metadata), ('metrics.json', metrics(result))):
            (root / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
        hashes = {p.name: file_hash(p) for p in sorted(root.iterdir())}
        (root / 'manifest.json').write_text(json.dumps({'files': hashes, 'status': 'RESEARCH_ONLY'}, indent=2) + '\n')
        os.rename(root, destination)
    verify_evidence(destination)
    return destination


def verify_evidence(root: str | Path) -> bool:
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    expected = set(manifest['files']) | {'manifest.json'}
    if {p.name for p in root.iterdir()} != expected:
        raise ValueError('unexpected or missing evidence files')
    for name, digest in manifest['files'].items():
        if Path(name).name != name or file_hash(root / name) != digest:
            raise ValueError(f'evidence integrity failure: {name}')
    return True

"""Validated snapshots: shared session calendar, no IPO backfill, no silent drops.

Adjusted OHLC is a research input, not proof of point-in-time corporate-action
accounting. Missing quotes remain missing; forward filling is allowed for marking
held units only, never to make a security eligible or executable.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

_SYMBOL = re.compile(r'^(?:sh|sz|bj)\d{6}$')
_COLUMNS = ('open', 'high', 'low', 'close', 'volume', 'raw_open', 'raw_close')


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Market:
    frames: Mapping[str, pd.DataFrame]
    calendar: pd.DatetimeIndex
    sectors: Mapping[str, str]
    quality: str
    provenance: Mapping[str, object]

    @classmethod
    def from_frames(cls, frames: Mapping[str, pd.DataFrame],
                    calendar: pd.DatetimeIndex, sectors: Mapping[str, str] | None = None,
                    quality: str = 'adjusted_unit_proxy',
                    provenance: Mapping[str, object] | None = None) -> Market:
        dates = pd.DatetimeIndex(calendar)
        if (dates.empty or dates.has_duplicates or not dates.is_monotonic_increasing
                or dates.tz is not None or dates.hasnans
                or not (dates == dates.normalize()).all()
                or (dates < pd.Timestamp('2023-01-01')).any()):
            raise ValueError('require unique increasing timezone-naive sessions from 2023')
        if not frames:
            raise ValueError('universe cannot be empty')
        clean = {}
        for symbol, source in sorted(frames.items()):
            if not _SYMBOL.fullmatch(symbol):
                raise ValueError(f'invalid security identifier: {symbol}')
            f = source.copy(deep=True)
            if not isinstance(f.index, pd.DatetimeIndex):
                raise ValueError(f'{symbol}: DatetimeIndex required')
            if f.index.has_duplicates or not f.index.is_monotonic_increasing:
                raise ValueError(f'{symbol}: duplicate or unordered dates')
            if f.index.tz is not None or f.index.hasnans:
                raise ValueError(f'{symbol}: invalid dates')
            if not set(_COLUMNS).issubset(f.columns):
                raise ValueError(f'{symbol}: required columns are {_COLUMNS}')
            # Validate the input, including rows that would otherwise be filtered out.
            values = f.loc[:, _COLUMNS].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values[:, [0, 1, 2, 3, 5, 6]] <= 0).any():
                raise ValueError(f'{symbol}: nonfinite or nonpositive prices')
            if (values[:, 4] < 0).any():
                raise ValueError(f'{symbol}: negative volume')
            if ((f.high + 1e-7 < f[['open', 'close', 'low']].max(axis=1)).any()
                    or (f.low - 1e-7 > f[['open', 'close', 'high']].min(axis=1)).any()):
                raise ValueError(f'{symbol}: impossible OHLC geometry')
            within = f.loc[(f.index >= dates[0]) & (f.index <= dates[-1]), _COLUMNS]
            if not within.index.isin(dates).all():
                raise ValueError(f'{symbol}: quote outside supplied session calendar')
            clean[symbol] = within
        return cls(clean, dates.copy(), dict(sectors or {}), quality, dict(provenance or {}))

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self.frames))

    def panel(self, column: str) -> pd.DataFrame:
        if column not in _COLUMNS:
            raise ValueError(f'unknown field: {column}')
        return pd.DataFrame({s: self.frames[s][column].reindex(self.calendar)
                             for s in self.symbols}, index=self.calendar)

    def subset(self, symbols: Sequence[str]) -> Market:
        chosen = tuple(symbols)
        if len(chosen) != len(set(chosen)):
            raise ValueError('duplicate symbols are not a second position')
        unknown = set(chosen) - set(self.symbols)
        if unknown:
            raise ValueError(f'unknown symbols: {sorted(unknown)}')
        return Market.from_frames({s: self.frames[s] for s in chosen}, self.calendar,
                                  self.sectors, self.quality, self.provenance)

    def prefix(self, end: str | pd.Timestamp) -> Market:
        dates = self.calendar[self.calendar <= pd.Timestamp(end)]
        return Market.from_frames({s: f.loc[:end] for s, f in self.frames.items()},
                                  dates, self.sectors, self.quality, self.provenance)

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(self.quality.encode())
        h.update(self.calendar.asi8.tobytes())
        for s in self.symbols:
            h.update(s.encode())
            h.update(self.sectors.get(s, 'unknown').encode())
            h.update(self.frames[s].index.asi8.astype('<i8').tobytes())
            h.update(self.frames[s].loc[:, _COLUMNS].to_numpy(dtype='<f8').tobytes())
        return h.hexdigest()


def _member(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('manifest member escapes snapshot root')
    return path


def load_market(root: str | Path, *, supplement: str | Path | None = None,
                sectors: Mapping[str, str] | None = None) -> Market:
    """Load the supplied Tencent manifest, optionally applying the hashed BJ supplement.

    The observation index supplies sessions, not investable assets or hidden
    leader signals. Fail on missing manifest members rather than reducing scope.
    """
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    symbols = [record['symbol'] for record in manifest['records']]
    if len(symbols) != len(set(symbols)):
        raise ValueError('duplicate manifest symbol declarations')
    frames, identities, coverage = {}, {}, {}
    for record in manifest['records']:
        if record.get('role') != 'technology_equity':
            continue
        s = record['symbol']
        if record.get('status') != 'ok':
            raise ValueError(f'{s}: acquisition failed')
        parts = {}
        for mode in ('qfq', 'raw'):
            info = record[mode]
            path = _member(root, info['path'])
            if file_hash(path) != info['sha256']:
                raise ValueError(f'{s}/{mode}: SHA256 mismatch')
            f = pd.read_csv(path, parse_dates=['date']).set_index('date')
            if len(f) != info['rows'] or (len(f) and
                    (str(f.index[0].date()) != info['first'] or
                     str(f.index[-1].date()) != info['last'])):
                raise ValueError(f'{s}/{mode}: manifest coverage mismatch')
            identities[f'{s}/{mode}'] = info['sha256']
            parts[mode] = f
        if not parts['qfq'].index.equals(parts['raw'].index):
            raise ValueError(f'{s}: raw/adjusted session mismatch')
        f = parts['qfq'].copy()
        f['raw_open'], f['raw_close'] = parts['raw'].open, parts['raw'].close
        frames[s] = f
    if supplement is not None:
        folder = Path(supplement)
        m = json.loads((folder / 'supplement_manifest.json').read_text(encoding='utf-8'))
        s = m['symbol']
        if s not in frames or m['status'] != 'ok':
            raise ValueError('invalid supplement identity')
        modes = [row['adjustment'] for row in m['records']]
        if len(modes) != 2 or set(modes) != {'qfq', 'raw'}:
            raise ValueError('supplement requires exactly one qfq and one raw record')
        identities['original_' + s + '/raw'] = identities[s + '/raw']
        identities['original_' + s + '/qfq'] = identities[s + '/qfq']
        parts = {}
        for row in m['records']:
            mode = row['adjustment']
            path = folder / f'{s}_{mode}.csv'
            if file_hash(path) != row['csv_sha256']:
                raise ValueError(f'{s}/{mode}: supplement SHA256 mismatch')
            f = pd.read_csv(path, parse_dates=['date']).set_index('date')
            if len(f) != row['rows'] or (len(f) and
                    (str(f.index[0].date()) != row['first'] or
                     str(f.index[-1].date()) != row['last'])):
                raise ValueError(f'{s}/{mode}: supplement manifest coverage mismatch')
            parts[mode] = f
            identities[f'{s}/{mode}'] = row['csv_sha256']
        if not parts['qfq'].index.equals(parts['raw'].index):
            raise ValueError('supplement raw/adjusted calendar mismatch')
        f = parts['qfq'].copy()
        f['raw_open'], f['raw_close'] = parts['raw'].open, parts['raw'].close
        # The replacement is explicit and its original identity is retained above.
        frames[s] = f
        identities['supplement_manifest'] = file_hash(folder / 'supplement_manifest.json')
    calendar_record = next((r for r in manifest['records'] if r['symbol'] == 'sh000300'), None)
    if calendar_record is None or calendar_record.get('status') != 'ok':
        raise ValueError('calendar observation missing from manifest')
    info = calendar_record['raw']
    calendar_path = _member(root, info['path'])
    if file_hash(calendar_path) != info['sha256']:
        raise ValueError('calendar SHA256 mismatch')
    calendar_frame = pd.read_csv(calendar_path, parse_dates=['date'])
    calendar = pd.DatetimeIndex(calendar_frame.date)
    if len(calendar) != info['rows'] or str(calendar[0].date()) != info['first'] or str(calendar[-1].date()) != info['last']:
        raise ValueError('calendar manifest coverage mismatch')
    identities['calendar'] = file_hash(calendar_path)
    for s, f in frames.items():
        coverage[s] = {'rows': len(f), 'first': str(f.index.min().date()),
                       'last': str(f.index.max().date()),
                       'absent_sessions': int(len(calendar.difference(f.index)))}
    return Market.from_frames(frames, calendar, sectors, 'adjusted_unit_proxy', {
        'provider': manifest['provider'], 'files': identities, 'coverage': coverage,
        'manifest_sha256': file_hash(root / 'manifest.json'),
        'limitation': 'snapshot-adjusted units; not a verified corporate-action/share/tax ledger'})

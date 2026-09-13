"""Snapshot identity and manifest trust boundaries, with tiny local fixtures."""
import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from test_core import sample_market
from techquant.data import Market, file_hash, load_market


class SnapshotTests(unittest.TestCase):
    def test_fingerprint_distinguishes_close_numeric_values(self):
        m = sample_market(1, 90)
        frames = {s: f.copy() for s, f in m.frames.items()}
        frames[m.symbols[0]].iloc[50, 3] += 1e-11
        other = Market.from_frames(frames, m.calendar, quality='synthetic')
        self.assertNotEqual(m.fingerprint(), other.fingerprint())

    def test_calendar_manifest_is_enforced(self):
        m = sample_market(1, 90)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ('qfq', 'raw'):
                (root / folder).mkdir()
            f = next(iter(m.frames.values())).drop(columns=['raw_open', 'raw_close'])
            records = []
            record = {'symbol': m.symbols[0], 'role': 'technology_equity', 'status': 'ok'}
            for mode in ('qfq', 'raw'):
                path = root / mode / (m.symbols[0] + '.csv')
                f.to_csv(path, index_label='date')
                record[mode] = {'path': str(path.relative_to(root)), 'sha256': file_hash(path),
                                'rows': len(f), 'first': str(f.index[0].date()), 'last': str(f.index[-1].date())}
            path = root / 'qfq/sh000300.csv'
            f.to_csv(path, index_label='date')
            records = [record, {'symbol': 'sh000300', 'role': 'observation', 'status': 'ok',
                'raw': {'path': 'qfq/sh000300.csv', 'sha256': '0'*64, 'rows':len(f),
                        'first':str(f.index[0].date()), 'last':str(f.index[-1].date())}}]
            (root/'manifest.json').write_text(json.dumps({'records':records,'provider':'synthetic'}))
            with self.assertRaisesRegex(ValueError, 'calendar.*SHA256'):
                load_market(root)

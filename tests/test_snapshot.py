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


class SnapshotDeclarationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.symbol = 'sz300100'
        self.frame = sample_market(1, 90).frames[self.symbol]
        quotes = self.frame.drop(columns=['raw_open', 'raw_close'])
        record = {'symbol': self.symbol, 'role': 'technology_equity', 'status': 'ok'}
        for mode in ('qfq', 'raw'):
            path = self.root / f'{self.symbol}_{mode}.csv'
            quotes.to_csv(path, index_label='date')
            record[mode] = {'path': path.name, 'sha256': file_hash(path),
                           'rows': len(quotes), 'first': str(quotes.index[0].date()),
                           'last': str(quotes.index[-1].date())}
        self.manifest = {'provider': 'synthetic', 'records': [record,
            {'symbol': 'sh000300', 'role': 'observation', 'status': 'ok',
             'raw': dict(record['raw'])}]}
        self.supplement = self.root / 'supplement'
        self.supplement.mkdir()
        self.short_quotes = quotes.iloc[-30:]
        self.supplement_manifest = {'symbol': self.symbol, 'status': 'ok', 'records': []}
        for mode in ('qfq', 'raw'):
            path = self.supplement / f'{self.symbol}_{mode}.csv'
            self.short_quotes.to_csv(path, index_label='date')
            self.supplement_manifest['records'].append({'adjustment': mode,
                'csv_sha256': file_hash(path), 'rows': len(self.short_quotes),
                'first': str(self.short_quotes.index[0].date()),
                'last': str(self.short_quotes.index[-1].date())})
        self.write_manifests()

    def write_manifests(self):
        (self.root / 'manifest.json').write_text(json.dumps(self.manifest))
        (self.supplement / 'supplement_manifest.json').write_text(
            json.dumps(self.supplement_manifest))

    def test_valid_snapshot_retains_every_quote(self):
        result = load_market(self.root)
        pd.testing.assert_frame_equal(result.frames[self.symbol], self.frame,
                                      check_names=False, check_freq=False)
        self.assertEqual(result.symbols, (self.symbol,))

    def test_duplicate_equity_or_calendar_declaration_is_rejected(self):
        original = list(self.manifest['records'])
        for record in original:
            with self.subTest(symbol=record['symbol']):
                self.manifest['records'] = original + [dict(record)]
                self.write_manifests()
                with self.assertRaisesRegex(ValueError, 'duplicate.*symbol'):
                    load_market(self.root)

    def test_supplement_requires_exactly_one_raw_and_adjusted_record(self):
        original = list(self.supplement_manifest['records'])
        extra = dict(original[0], adjustment='other')
        self.short_quotes.to_csv(self.supplement / f'{self.symbol}_other.csv', index_label='date')
        for records in (original + [dict(original[0])], original[:1], original + [extra]):
            with self.subTest(modes=[r['adjustment'] for r in records]):
                self.supplement_manifest['records'] = records
                self.write_manifests()
                with self.assertRaisesRegex(ValueError, 'supplement.*exactly one'):
                    load_market(self.root, supplement=self.supplement)

    def test_supplement_coverage_is_enforced_even_with_valid_file_hashes(self):
        original = json.dumps(self.supplement_manifest)
        for mode in ('qfq', 'raw'):
            for key, value in (('rows', 31), ('first', '2023-01-03'), ('last', '2023-01-04')):
                with self.subTest(mode=mode, field=key):
                    self.supplement_manifest = json.loads(original)
                    row = next(r for r in self.supplement_manifest['records']
                               if r['adjustment'] == mode)
                    row[key] = value
                    self.write_manifests()
                    with self.assertRaisesRegex(ValueError, 'supplement.*coverage mismatch'):
                        load_market(self.root, supplement=self.supplement)

    def test_valid_supplement_keeps_calendar_and_never_backfills_history(self):
        result = load_market(self.root, supplement=self.supplement)
        self.assertEqual(len(result.calendar), len(self.frame))
        self.assertEqual(len(result.frames[self.symbol]), len(self.short_quotes))
        self.assertTrue(result.panel('close').iloc[:-30, 0].isna().all())
        pd.testing.assert_series_equal(result.frames[self.symbol].close,
                                       self.short_quotes.close, check_names=False, check_freq=False)
        files = result.provenance['files']
        self.assertEqual(files['original_' + self.symbol + '/raw'],
                         self.manifest['records'][0]['raw']['sha256'])
        self.assertEqual(files['supplement_manifest'],
                         file_hash(self.supplement / 'supplement_manifest.json'))

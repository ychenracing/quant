"""An issued forecast is reusable only for its original market and clock."""
import importlib
import importlib.util
import unittest
import numpy as np
from research import test_coherent


class IssuedForecastTests(unittest.TestCase):
    def fixture(self):
        self.assertIsNotNone(importlib.util.find_spec('research.coherent_study'))
        module = importlib.import_module('research.coherent_study')
        owner, f, _ = test_coherent.CoherentTests().owner(2)
        arrays = {k: getattr(f, k).copy() for k in module.ARRAYS}
        receipt = {'data_sha256': owner.market.fingerprint(), 'symbols': list(owner.market.symbols),
            'dates': [str(d.date()) for d in owner.market.calendar], 'fits': [],
            'forecast_sha256': f.fingerprint(), 'source': {'commit': module.ORIGIN_SOURCE}}
        return module, owner.market, arrays, receipt

    def test_exact_packet_restores_readonly_arrays(self):
        mod, market, arrays, receipt = self.fixture()
        f = mod.restore_prediction(market, arrays, receipt)
        self.assertEqual(f.fingerprint(), receipt['forecast_sha256'])
        self.assertFalse(f.expected.flags.writeable)

    def test_same_array_shape_with_a_different_calendar_is_rejected(self):
        mod, market, arrays, receipt = self.fixture()
        receipt['dates'][0] = '2022-12-30'
        with self.assertRaises(ValueError): mod.restore_prediction(market, arrays, receipt)

    def test_future_label_in_a_training_receipt_is_rejected(self):
        mod, market, arrays, receipt = self.fixture()
        receipt['fits'] = [{'session': 30, 'last_feature_session': 20, 'last_label_session': 40,
                            'task': 'first_passage', 'loss_barrier': .12, 'date': str(market.calendar[30].date())}]
        with self.assertRaises(ValueError): mod.restore_prediction(market, arrays, receipt)

    def test_tampered_outcome_probabilities_change_forecast_identity(self):
        mod, market, arrays, receipt = self.fixture()
        arrays['outcome_probability'][0, 0, 0] += .01
        with self.assertRaises(ValueError): mod.restore_prediction(market, arrays, receipt)

    def test_removed_universe_cannot_use_the_parent_pool_forecast(self):
        mod, market, arrays, receipt = self.fixture()
        with self.assertRaises(ValueError):
            mod.restore_prediction(market.subset([market.symbols[0]]), arrays, receipt)

    def test_extracted_manifest_cannot_override_pinned_archive_manifest(self):
        import io, json, tarfile, tempfile
        from pathlib import Path
        from unittest.mock import patch
        mod, _, _, _ = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'parent.tar.gz'
            original = json.dumps({'original.txt': 'original-digest'}).encode()
            with tarfile.open(archive, 'w:gz') as stream:
                member = tarfile.TarInfo('nonlinear/MANIFEST.json')
                member.size = len(original)
                stream.addfile(member, io.BytesIO(original))
            restored = root / 'restore' / 'nonlinear'
            restored.mkdir(parents=True)
            (restored / 'MANIFEST.json').write_text('{}')
            with patch.object(mod, 'ARCHIVE_SHA256', mod.file_hash(archive)):
                with self.assertRaisesRegex(ValueError, 'manifest identity'):
                    mod.ForecastStore(archive, root / 'restore')

    def test_wrong_generation_source_cannot_be_relabelled(self):
        mod, market, arrays, receipt = self.fixture()
        receipt['source']['commit'] = 'other-source'
        with self.assertRaises(ValueError): mod.restore_prediction(market, arrays, receipt)


if __name__ == '__main__': unittest.main()

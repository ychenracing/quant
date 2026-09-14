"""Two compact checks for external-comparator provenance, not economic gates."""
from pathlib import Path
import tempfile
import unittest
from research import native


class ReferenceInputsTests(unittest.TestCase):
    def test_source_manifest_covers_executed_configuration(self):
        self.assertTrue(hasattr(native, 'reference_files'), 'reference configuration audit missing')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'model.py').write_text('x = 1\n')
            (root / 'config.yaml').write_text('budget: 0.5\n')
            first = native.reference_files(root)
            self.assertIn('config.yaml', first)
            (root / 'config.yaml').write_text('budget: 0.8\n')
            self.assertNotEqual(first, native.reference_files(root))

    def test_index_manifest_binds_volume_as_well_as_prices(self):
        self.assertTrue(hasattr(native, 'index_files'), 'index input audit missing')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('sh000300', 'sh000682', 'sz399808'):
                (root / (name + '.csv')).write_text('date,close,volume\n2023-01-03,100,10\n')
            first = native.index_files(root)
            (root / 'sh000300.csv').write_text('date,close,volume\n2023-01-03,100,0\n')
            self.assertNotEqual(first, native.index_files(root))

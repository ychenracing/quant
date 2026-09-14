"""Receipt, calendar and payload integrity for reusing already issued forecasts."""
import importlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from test_core import sample_market
from techquant.data import file_hash
from research.pathwise import Prediction


class IssuedForecastTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module('research.issued_forecasts')
        except ImportError as exc:
            self.fail(f'Issued forecast validator missing: {exc}')

    def payload(self, directory):
        market = sample_market(2, 25)
        shape = (25, 2)
        price = market.panel('close').to_numpy()
        arrays = dict(expected=np.zeros(shape), tail=np.full(shape, .1), ready=np.ones(shape, bool),
                      price=price, ema10=price, ema20=price, ema60=price,
                      momentum5=np.zeros(shape), ret1=np.zeros(shape),
                      outcome_probability=np.full((*shape, 3), np.nan))
        f = Prediction(market.symbols, market.fingerprint(), 20, **arrays, fits=[])
        target = Path(directory) / (f.fingerprint() + '.npz')
        np.savez_compressed(target, **arrays)
        receipt = dict(data_sha256=market.fingerprint(), forecast_sha256=f.fingerprint(),
                       symbols=list(market.symbols), dates=[str(d.date()) for d in market.calendar],
                       fits=[], archive_sha256=file_hash(target))
        target.with_suffix('.json').write_text(json.dumps(receipt))
        return market, target, receipt

    def test_payload_is_exact_and_immutable(self):
        with TemporaryDirectory() as root:
            market, path, receipt = self.payload(root)
            restored = self.module.load_payload(market, path, receipt['forecast_sha256'])
            self.assertEqual(restored.fingerprint(), receipt['forecast_sha256'])
            self.assertFalse(restored.expected.flags.writeable)
            self.assertFalse(restored.outcome_probability.flags.writeable)

    def test_tampered_archive_and_calendar_are_rejected(self):
        with TemporaryDirectory() as root:
            market, path, receipt = self.payload(root)
            path.write_bytes(path.read_bytes() + b'tamper')
            with self.assertRaises(ValueError):
                self.module.load_payload(market, path, receipt['forecast_sha256'])
        with TemporaryDirectory() as root:
            market, path, receipt = self.payload(root)
            receipt['dates'][0] = '2022-12-30'
            path.with_suffix('.json').write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):
                self.module.load_payload(market, path, receipt['forecast_sha256'])

    def test_mature_label_boundary_is_rechecked(self):
        with TemporaryDirectory() as root:
            market, path, receipt = self.payload(root)
            receipt['fits'] = [{'session':20, 'last_feature_session':1, 'last_label_session':21}]
            path.with_suffix('.json').write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):
                self.module.load_payload(market, path, receipt['forecast_sha256'])

    def test_larger_universe_is_not_sliced_into_removed_stock_run(self):
        with TemporaryDirectory() as root:
            market, path, receipt = self.payload(root)
            with self.assertRaises(ValueError):
                self.module.load_payload(market.subset(market.symbols[:1]), path,
                                         receipt['forecast_sha256'])

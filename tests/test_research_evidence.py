"""Small, offline regression set for stale-result reuse and evidence completeness."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_core import sample_market
from techquant.config import Config
from techquant.data import file_hash
from techquant.engine import run
from techquant.evidence import save_result, source_identity, verify_evidence
from research import study


class ResearchEvidenceTests(unittest.TestCase):
    def test_empty_manifest_is_not_valid_research_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'manifest.json').write_text('{"files": {}, "status": "RESEARCH_ONLY"}')
            with self.assertRaisesRegex(ValueError, 'required evidence'):
                verify_evidence(root)

    def test_selection_checks_source_data_and_frozen_config_before_any_replay(self):
        market = sample_market(34, 45)
        protocol = json.loads(Path(study.__file__).with_name('protocol.json').read_text())
        base = {'candidate': 0, 'status': 'SELECTED_NOT_ECONOMICALLY_ACCEPTED',
                'config': asdict(Config(**(protocol['frozen_parameters_outside_grid'] | protocol['candidates'][0]))),
                'source': source_identity(), 'selection_data_sha256': market.fingerprint(),
                'protocol_sha256': file_hash(Path(study.__file__).with_name('protocol.json')),
                'runner_sha256': file_hash(Path(study.__file__))}
        for key, value in [('source', {}), ('selection_data_sha256', 'other data'),
                           ('config', base['config'] | {'fast': base['config']['fast'] + 1}), ('runner_sha256', 'other runner')]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                selection = copy.deepcopy(base)
                selection[key] = value
                with patch.object(study, 'run', side_effect=AssertionError('must reject before replay')):
                    with self.assertRaisesRegex(ValueError, 'selection'):
                        study.evaluate(market, {'pools': {'union': list(market.symbols)}},
                                       protocol, selection, Path(tmp), batch_size=1)

    def test_result_restore_binds_configuration_universe_cost_delay_and_source(self):
        from techquant import evidence
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'result'
            result = run(sample_market(2, 45))
            save_result(result, root)
            self.assertTrue(hasattr(evidence, 'load_result'), 'safe result restore is missing')
            restored = evidence.load_result(root, expected=result.metadata)
            self.assertEqual(restored.metadata, result.metadata)
            for key, value in [('config', {}), ('universe', []), ('cost_multiplier', 2),
                               ('delay', 2), ('source', {}), ('data_sha256', 'different')]:
                bad = dict(result.metadata) | {key: value}
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'identity'):
                    evidence.load_result(root, expected=bad)

    def test_human_report_labels_model_book_and_explains_each_security(self):
        from techquant import cli
        self.assertTrue(hasattr(cli, 'inspection_report'), 'per-security research report is missing')
        from techquant.passive import run_passive_ownership
        market = sample_market(2, 90)
        result = run_passive_ownership(market)
        report = cli.inspection_report(market, result, {})
        self.assertEqual(report['status'], 'RETURN_FIRST_PRODUCTION_MODE')
        self.assertEqual(report['strategy'], 'passive_ownership')
        self.assertEqual({row['symbol'] for row in report['securities']}, set(market.symbols))
        self.assertTrue(all(row['explanation'] for row in report['securities']))
        self.assertEqual(report['sell_candidates'], [])
        self.assertFalse(report['risk']['active_risk_control'])
        self.assertIn('真实账户', report['warning'])

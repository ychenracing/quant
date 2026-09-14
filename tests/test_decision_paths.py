"""Decision-path selection is causal and replays actual funded recommendations."""
import unittest
import numpy as np
from test_core import sample_market
from techquant.engine import run
from research.decision_paths import Owner, Parameters, build_shadows


class DecisionPathsTests(unittest.TestCase):
    def test_prefix_matches_and_removed_universe_has_no_external_shadow(self):
        m = sample_market(3, 100)
        def factory(market, cfg): return Owner(market, Parameters(20, 1.), build_shadows(market))
        full = run(m, policy_factory=factory)
        short = run(m.prefix(str(m.calendar[74].date())), policy_factory=factory)
        np.testing.assert_allclose(full.equity.nav.iloc[:75], short.equity.nav)
        np.testing.assert_allclose(full.targets.iloc[:75], short.targets)
        reduced = m.subset(m.symbols[:2])
        shadows = build_shadows(reduced)
        for shadow in shadows.values():
            self.assertEqual(shadow.metadata['universe'], list(reduced.symbols))
        with self.assertRaisesRegex(ValueError, 'identity'):
            Owner(m, Parameters(), shadows)
        self.assertTrue((full.equity.cash >= 0).all())

    def test_parameters_reject_invalid_values(self):
        for window, penalty in [(0, 1.), (True, 1.), (20, float('nan')), (20, -1.)]:
            with self.assertRaises(ValueError): Parameters(window, penalty)

    def test_shadow_labels_and_target_calendar_cannot_be_substituted(self):
        import copy
        market = sample_market(2, 80)
        shadows = build_shadows(market)
        for change in ('labels', 'calendar'):
            broken = copy.deepcopy(shadows)
            if change == 'labels':
                broken['leadership'], broken['admission'] = broken['admission'], broken['leadership']
            else:
                broken['incumbent'].targets.index = broken['incumbent'].targets.index.shift(1, freq='D')
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'identity'):
                Owner(market, Parameters(), broken)

    def test_study_reuses_exact_shadow_and_preserves_checked_scores(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from research.decision_path_study import DecisionStudy
        market = sample_market(2, 80)
        with TemporaryDirectory() as folder:
            study = DecisionStudy(Path(folder))
            # The study identity includes a declaration, supplied independently of
            # the pricing fixture used in this offline test.
            study.identity = lambda: {'test': 'exact-shadow-and-score-receipt'}
            path = Path(folder)/'selection/runs/path'
            result = study.saved(market, path, Parameters(20, 1.))
            cached = study.saved(market, path, Parameters(20, 1.))
            np.testing.assert_array_equal(result.equity.nav, cached.equity.nav)
            self.assertEqual(len(study.cache), 1)
            score = path.parent.parent/'decisions/path.json'
            receipt = json.loads(score.read_text())
            receipt['observations'][0]['scores']['incumbent'] = 100.
            score.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'receipt identity'):
                study.saved(market, path, Parameters(20, 1.))

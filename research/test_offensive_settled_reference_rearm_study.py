import importlib.util
import unittest


class SettledReferenceRearmStudyTests(unittest.TestCase):
    def test_source_bound_study_registration(self):
        self.assertIsNotNone(
            importlib.util.find_spec('research.offensive_settled_reference_rearm_study'),
            'settled reference rearm source-bound study is absent',
        )
        from research.offensive_settled_reference_rearm_study import Study
        study = Study('offensive_settled_reference_rearm')
        self.assertEqual(study.family, 'offensive_settled_reference_rearm')
        self.assertEqual(study.REGISTRATION_COMMIT, 'bce12d4edb27025537ec08b098901fd6f91073c7')
        self.assertEqual(len(study.module.grid()), 2)
        dependencies = study.identity()['dependencies']
        self.assertIn('offensive_settled_reference_rearm.py', dependencies)
        self.assertIn('offensive_alpha_decay_displacement.py', dependencies)
        self.assertIn('finite_study/__init__.py', dependencies)


if __name__ == '__main__':
    unittest.main()

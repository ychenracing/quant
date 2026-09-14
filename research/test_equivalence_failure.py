"""A refused evidence comparison must preserve its numeric witness."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np
from techquant.policy import CloseDecision
from research import neighborhood_validation as module


class EquivalenceFailureTests(unittest.TestCase):
    def test_exact_cap_guard_is_preserved_with_a_machine_readable_witness(self):
        self.assertTrue(callable(getattr(module, 'check_recorded_decision', None)))
        decision=CloseDecision(np.array([.5]),'HOLD',.5,np.array([1.]))
        previous=SimpleNamespace(target_cap=float(np.nextafter(.5,1.)),reason='HOLD')
        with TemporaryDirectory() as directory:
            destination=Path(directory)/'witness.json'
            with self.assertRaises(ValueError):
                module.check_recorded_decision(decision,np.array([.5]),previous,
                    scope='fixture',day='2023-01-03',destination=destination)
            witness=json.loads(destination.read_text())
            self.assertEqual(witness['maximum_weight_error'],0.)
            self.assertNotEqual(witness['cap_replayed_hex'],witness['cap_recorded_hex'])
            self.assertEqual(witness['reason_replayed'],witness['reason_recorded'])

    def test_weight_and_reason_failures_are_not_hidden(self):
        self.assertTrue(callable(getattr(module,'check_recorded_decision',None)))
        with TemporaryDirectory() as directory:
            for weights,reason in (([.4],'HOLD'),([.5],'SELL')):
                with self.subTest(weights=weights,reason=reason):
                    previous=SimpleNamespace(target_cap=.5,reason=reason)
                    with self.assertRaises(ValueError):
                        module.check_recorded_decision(CloseDecision(np.array([.5]),'HOLD',.5),
                            np.array(weights),previous,scope='fixture',day='2023-01-03',
                            destination=Path(directory)/'witness.json')

    def test_matching_decision_does_not_create_false_failure(self):
        self.assertTrue(callable(getattr(module,'check_recorded_decision',None)))
        with TemporaryDirectory() as directory:
            destination=Path(directory)/'witness.json'
            error=module.check_recorded_decision(CloseDecision(np.array([.5]),'HOLD',.5),
                np.array([.5]),SimpleNamespace(target_cap=.5,reason='HOLD'),
                scope='fixture',day='2023-01-03',destination=destination)
            self.assertEqual(error,0.)
            self.assertFalse(destination.exists())

if __name__=='__main__':unittest.main()

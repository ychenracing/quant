"""Admission may withhold new money, never suppress a protective reduction."""
import unittest
import numpy as np
from test_core import sample_market
from techquant.policy import CloseObservation, CloseDecision


class AdmissionTests(unittest.TestCase):
    def module(self):
        try:
            from research import admission
            return admission
        except ImportError as exc:
            self.fail(f'admission gate missing: {exc}')

    def test_gate_prefix_and_increases_only(self):
        a=self.module();m=sample_market(2,100);p=a.Parameters()
        owner=a.Owner(m,p);short=a.Owner(m.prefix(m.calendar[69]),p)
        np.testing.assert_array_equal(owner.admit[:70],short.admit)
        class Requests:
            def decide(self,o):return CloseDecision(np.array([0.,.8]),'REDUCE_AND_BUY')
        owner.inner=Requests();owner.admit[:]=False
        obs=CloseObservation.from_inventory(40,str(m.calendar[40].date()),100.,20.,np.array([1.,1.]),np.array([.4,.4]))
        x=owner.decide(obs)
        np.testing.assert_allclose(x.weights,[0.,.4])
        self.assertIn('WAIT_FOR_MARKET_ADMISSION',x.reason)

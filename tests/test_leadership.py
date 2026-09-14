"""Protective intent, capital conservation and causal leadership signals."""
import unittest
import numpy as np
from test_core import sample_market
from techquant.policy import CloseObservation


class LeadershipTests(unittest.TestCase):
    def module(self):
        try:
            from research import leadership
            return leadership
        except ImportError as exc:
            self.fail(f'leadership policy is missing: {exc}')

    def test_signal_prefix_is_causal(self):
        m = self.module(); market = sample_market(3, 100); p = m.Parameters()
        a = m.signals(market, p); b = m.signals(market.prefix(market.calendar[69]), p)
        np.testing.assert_allclose(a.score[:70], b.score, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(a.pressure[:70], b.pressure)

    def test_pressure_halving_is_latched_to_actual_units_not_repeated(self):
        m = self.module(); market = sample_market(1, 80); p = m.Parameters(pressure_guard=True)
        s = m.signals(market, p); s.pressure[:] = False; s.pressure[30:33] = True
        s.price[:] = 100.; s.open[:] = 100.; s.ema[:] = 95.; s.ema10[:] = 110.
        s.recovery[:] = False
        owner = m.Owner(market, p, s)
        def obs(i, units, weight, cash):
            return CloseObservation.from_inventory(i, str(market.calendar[i].date()), 1000., cash,
                                                  np.array([units]), np.array([weight]))
        first = owner.decide(obs(30, 8., .8, 200.))
        retry = owner.decide(obs(31, 8., .8, 200.))
        filled = owner.decide(obs(32, 4., .4, 600.))
        self.assertAlmostEqual(first.weights[0], .4)
        self.assertAlmostEqual(retry.weights[0], .4)
        self.assertAlmostEqual(filled.weights[0], .4)
        self.assertIn('PRESSURE_REDUCTION', first.reason)

    def test_protective_exit_cannot_be_cancelled_by_price_rebound(self):
        m = self.module(); market = sample_market(1, 80); p = m.Parameters()
        s = m.signals(market, p); s.price[:] = 100.; s.open[:] = 100.; s.pressure[:] = False
        owner = m.Owner(market, p, s)
        def obs(i):
            return CloseObservation.from_inventory(i, str(market.calendar[i].date()), 1000.,200.,
                                                  np.array([8.]),np.array([.8]))
        owner.decide(obs(30)); s.price[31] = 80.; first=owner.decide(obs(31))
        self.assertEqual(first.weights[0],0.)
        self.assertEqual(owner.decide(obs(32)).weights[0],0.)

    def test_readmission_counts_only_post_liquidation_closes(self):
        m=self.module(); market=sample_market(1,80); p=m.Parameters()
        s=m.signals(market,p); s.price[:]=100.;s.open[:]=100.;s.ema10[:]=90.;s.ema[:]=90.
        s.ready[:]=True;s.score[:]=1.;s.momentum5[:]=.1;s.pressure[:]=False
        owner=m.Owner(market,p,s)
        def obs(i,held):
            return CloseObservation.from_inventory(i,str(market.calendar[i].date()),1000.,200. if held else 1000.,
                np.array([8. if held else 0.]),np.array([.8 if held else 0.]))
        for i in range(25,31): owner.decide(obs(i,True))
        s.price[31]=85.;owner.decide(obs(31,True))
        self.assertEqual(owner.decide(obs(32,False)).weights[0],0.)
        self.assertEqual(owner.decide(obs(33,False)).weights[0],0.)
        self.assertGreater(owner.decide(obs(34,False)).weights[0],0.)

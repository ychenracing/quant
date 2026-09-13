"""Behavioral tests for the registered retention/recovery hypothesis, not return gates."""
from dataclasses import replace
import unittest
import numpy as np
from test_core import sample_market
from techquant.config import Config
from techquant.features import build_features
from techquant.strategy import RiskState, target_weights

class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(fast=10, slow=40, max_positions=2, target_vol=.8)
        self.f = build_features(sample_market(3, 120), self.cfg)
        self.f = replace(self.f, entry=np.ones((120, 3), dtype=bool),
                         exit=np.zeros((120, 3), dtype=bool),
                         score=np.tile([1., .9, 1.3], (120, 1)),
                         market_return=np.full(120, .001), market_vol=np.full(120, .01),
                         breadth=np.full(120, .8), weak=np.zeros(120, dtype=bool),
                         shock_fraction=np.zeros(120), market_dd=np.zeros(120))

    def test_intact_membership_does_not_churn_on_ordinary_score_changes(self):
        current=np.array([.40, .35, 0.])
        target,_=target_weights(100,self.f,current,self.cfg,cap=1.,rebalance=True)
        np.testing.assert_array_equal(target,current)

    def test_dominant_challenger_can_replace_an_incumbent(self):
        f=replace(self.f,score=np.tile([1., .9, 3.],(120,1)))
        target,_=target_weights(100,f,np.array([.40,.35,0.]),self.cfg,cap=1.,rebalance=True)
        self.assertGreater(target[2],0.)
        self.assertEqual(target[1],0.)

    def test_risk_and_concentration_override_retention(self):
        current=np.array([.40,.35,0.])
        reduced,_=target_weights(100,self.f,current,self.cfg,cap=.25,
                                rebalance=True,risk_reduction=True)
        self.assertLessEqual(reduced.sum(),.25+1e-10)
        restored,_=target_weights(100,self.f,current,self.cfg,cap=1.,
                                 rebalance=True,risk_increase=True)
        self.assertGreater(restored.sum(),current.sum())
        f=replace(self.f,sectors=('same','same','other'))
        target,_=target_weights(100,f,np.array([.44,.43,0.]),self.cfg,cap=1.,rebalance=False)
        self.assertLessEqual(target[:2].sum(),self.cfg.sector_cap+1e-10)

    def test_accumulated_loss_alarm_uses_only_completed_returns(self):
        returns=np.zeros(120);returns[98:101]=-.0195
        f=replace(self.f,market_return=returns,breadth=np.full(120,.4))
        state=RiskState(cap=1.)
        cap,reason=state.update(100,f,[100.]*101,self.cfg)
        self.assertEqual(cap,0.)
        self.assertEqual(reason,'ACCUMULATED_MARKET_SHOCK')

    def test_recovery_restores_half_then_full_budget(self):
        state=RiskState()
        for i in range(90,90+self.cfg.recovery):
            state.update(i,self.f,[100.]*(i+1),self.cfg)
        self.assertEqual(state.cap,.5)
        for i in range(90+self.cfg.recovery,90+2*self.cfg.recovery):
            state.update(i,self.f,[100.]*(i+1),self.cfg)
        self.assertEqual(state.cap,1.)

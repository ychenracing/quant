"""Correctness of the separately declared observed-trend policy boundary."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from techquant.policy import CloseObservation
from research import test_nonlinear_ownership as fixtures


class ObservedTrendTests(unittest.TestCase):
    observation = fixtures.OwnershipTests.observation

    def module(self):
        try:
            from research import observed_trend
        except ImportError as error:
            self.fail(f"Declared observed-trend policy is missing: {error}")
        return observed_trend

    def configured(self):
        module=self.module()
        inner, forecast=fixtures.OwnershipTests().owner(size=2,positions=1)
        signals=SimpleNamespace(exit=np.zeros((50,2),dtype=bool),
            entry=np.ones((50,2),dtype=bool),market=np.ones(50,dtype=bool))
        market=SimpleNamespace(symbols=('name0','name1'))
        with patch.object(module,'signals',return_value=signals),patch.object(module,'Learner',return_value=inner):
            owner=module.Owner(market,module.Parameters())
        return owner,signals,forecast

    def test_confirmed_observed_exit_overrides_positive_forecast(self):
        owner,signals,forecast=self.configured()
        forecast.expected[:]=.2
        signals.exit[1,0]=True
        decision=owner.decide(self.observation(1,[100.,0.]))
        np.testing.assert_array_equal(decision.unit_targets,[0.,0.])
        self.assertIn('OBSERVED_TREND_EXIT',decision.reason)
        signals.exit[2,0]=False
        retry=owner.decide(self.observation(2,[100.,0.]))
        self.assertEqual(retry.unit_targets[0],0.)

    def test_failed_admission_retains_existing_units(self):
        owner,signals,_=self.configured()
        signals.entry[1]=False
        before=self.observation(1,[100.,0.])
        decision=owner.decide(before)
        np.testing.assert_array_equal(decision.unit_targets,before.units)

    def test_failed_admission_cannot_fund_new_inventory(self):
        owner,signals,_=self.configured()
        signals.entry[1]=False
        decision=owner.decide(self.observation(1,[0.,0.]))
        np.testing.assert_array_equal(decision.unit_targets,[0.,0.])

    def test_market_gate_only_applies_when_requested(self):
        owner,signals,_=self.configured()
        signals.market[1]=False
        self.assertGreater(owner.decide(self.observation(1,[0.,0.])).weights.sum(),0.)
        owner,signals,_=self.configured()
        owner.params=self.module().Parameters(require_market_trend=True)
        signals.market[1]=False
        np.testing.assert_array_equal(owner.decide(self.observation(1,[0.,0.])).unit_targets,[0.,0.])

    def test_signals_are_prefix_causal(self):
        from test_core import sample_market
        module=self.module();market=sample_market(3,120);p=module.Parameters()
        full=module.signals(market,p);short=module.signals(market.prefix(market.calendar[89]),p)
        for name in ('entry','exit','market'):
            np.testing.assert_array_equal(getattr(full,name)[:90],getattr(short,name))


if __name__=='__main__':unittest.main()

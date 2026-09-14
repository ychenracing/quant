"""Funded risk authority retains warnings without inventing fills or resetting NAV."""
from dataclasses import asdict
import importlib
import importlib.util
import unittest
import numpy as np
import pandas as pd
from test_core import sample_market
from techquant.data import Market
from techquant.engine import run
from techquant.policy import CloseObservation


class FundedRiskTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research.funded_risk'),
                             'registered funded-book authority is not implemented')
        return importlib.import_module('research.funded_risk')

    def owner(self, market=None, cushion=True):
        m = self.module()
        return m.Owner(market or sample_market(2, 140), m.Parameters(cushion))

    def warning(self, owner, cap):
        def update(*args):
            owner.risk.warnings.cap = cap
            return cap, 'CROSS_SECTION_SHOCK' if cap == 0 else 'TREND_OPEN'
        owner.risk.warnings.update = update

    def observation(self, owner, i, units=None, cash=2_000_000.):
        u = np.zeros(len(owner.market.symbols)) if units is None else np.array(units, dtype=float)
        v = np.nan_to_num(u*owner.features.close[i], nan=0.)
        nav = float(cash+v.sum())
        return CloseObservation.from_inventory(i, str(owner.market.calendar[i].date()), nav, cash, u, v/nav)

    def test_two_candidates_bind_the_new_and_parent_contracts(self):
        m = self.module()
        self.assertEqual([asdict(p) for p in m.grid()], [{'use_cushion':False},{'use_cushion':True}])
        for bad in (0, 1, None, 'yes'):
            with self.assertRaises(ValueError): m.Parameters(bad)
        from research.finite_study import Study
        identity = Study('funded_risk').identity()
        self.assertIn('funded_risk_contract.json', identity['dependencies'])
        self.assertIn('support_budget.py', identity['dependencies'])

    def test_warning_freezes_new_risk_but_does_not_invent_a_stock_exit(self):
        owner = self.owner(cushion=False)
        self.warning(owner, 0.)
        o = self.observation(owner, 40, [10000., 0.])
        d = owner.decide(o)
        np.testing.assert_array_equal(d.unit_targets, o.units)
        self.assertIn('CROSS_SECTION_SHOCK', d.reason)
        self.assertEqual(owner.risk.warnings.cap, 0.)
        self.assertGreater(owner.risk.cap, 0.)
        self.assertEqual(owner._risk_limit(o, d.cap), 0.)

    def test_a_small_loss_budget_breach_is_not_granted_an_eight_percent_drift_band(self):
        owner = self.owner(cushion=True)
        self.warning(owner, 1.)
        o = self.observation(owner, 40, [40000., 0.], 1200000.)
        owner.risk.cap = .1  # recovering notional cap must not waive a live risk breach
        price = owner.features.close[40]
        distance = np.maximum(price-owner.admission_stop(40), .02*price)
        prior_risk = float(o.units@distance)
        budget = .99*prior_risk
        self.assertGreater(budget, .01*o.nav)  # exercise a breach above the declared exploration floor
        owner.risk.global_peak = (o.nav-budget)/(1-owner.config.risk_drawdown)
        d = owner.decide(o)
        risk = float(d.unit_targets@np.maximum(price-owner.stop,.02*price))
        self.assertLess(d.unit_targets[0], o.units[0])
        self.assertLessEqual(risk, budget+1e-7)
        self.assertLess(float((o.units-d.unit_targets)@price)/o.nav, owner.config.trade_band)
        self.assertEqual(d.unit_targets[1], 0.)

    def test_global_peak_and_full_actual_nav_history_survive_recovery(self):
        owner = self.owner(cushion=True)
        self.warning(owner, 0.)
        for i,nav in [(40,3_000_000.),(41,2_000_000.),(42,2_500_000.)]:
            owner.decide(self.observation(owner,i,cash=nav))
            self.assertEqual(owner.risk.global_peak,3_000_000.)
        self.assertEqual(owner.history,[3_000_000.,2_000_000.,2_500_000.])
        expected = min(.1*2_500_000., max(.01*2_500_000.,2_500_000.-.82*3_000_000.))
        self.assertAlmostEqual(owner.risk.budget, expected)

    def test_loss_budget_projection_does_not_spend_expected_full_exit_proceeds(self):
        owner = self.owner(cushion=False)
        self.warning(owner, 1.)
        owner.decide(self.observation(owner,40,[10000.,0.]))
        owner.stop[0] = owner.features.close[41,0]*1.02
        d = owner.decide(self.observation(owner,41,[10000.,0.]))
        np.testing.assert_array_equal(d.unit_targets,[0.,0.])
        retry = owner.decide(self.observation(owner,42,[10000.,0.]))
        np.testing.assert_array_equal(retry.unit_targets,[0.,0.])
        self.assertTrue(owner.exit_pending[0])

    def test_partial_protection_remains_latched_when_current_risk_recovers(self):
        owner = self.owner(cushion=True)
        self.warning(owner,1.)
        o = self.observation(owner,40,[50000.,0.],1200000.)
        owner.risk.global_peak = o.nav/(1-owner.config.risk_drawdown)
        d = owner.decide(o)
        self.assertLess(d.unit_targets[0], o.units[0])
        retry = owner.decide(self.observation(owner,41,[49000.,0.],1230000.))
        self.assertLessEqual(retry.unit_targets[0],d.unit_targets[0]+1e-9)
        self.assertEqual(retry.unit_targets[1],0.)

    def test_actual_execution_is_prefix_causal_exclusion_isolated_and_self_funded(self):
        m = self.module()
        market = sample_market(4,145)
        frames = {s:f.copy() for s,f in market.frames.items()}
        for f in frames.values():
            f.loc[market.calendar[65:80],['open','high','low','close','raw_open','raw_close']] *= .82
        market = Market.from_frames(frames,market.calendar,quality='synthetic')
        def execute(current, delay=1):
            return run(current,delay=delay,policy_factory=lambda x,c:m.Owner(x,m.Parameters(True)))
        full=execute(market);cut=market.calendar[110];short=execute(market.prefix(cut))
        pd.testing.assert_frame_equal(full.equity.loc[:cut],short.equity)
        pd.testing.assert_frame_equal(full.targets.loc[:cut],short.targets)
        removed=market.symbols[-1]
        frames[removed].loc[:,['open','high','low','close','raw_open','raw_close']] *= 100.
        changed=Market.from_frames(frames,market.calendar,quality='synthetic')
        a,b=execute(market.subset(market.symbols[:-1])),execute(changed.subset(market.symbols[:-1]))
        pd.testing.assert_frame_equal(a.equity,b.equity);self.assertEqual(a.orders,b.orders)
        from research.ledger_attribution import attribute
        for r in (full,execute(market,2)):
            self.assertTrue((r.equity.cash>=0).all());self.assertTrue((r.equity.exposure<=1+1e-10).all())
            self.assertTrue(all(o['signal_date']<o['date'] for o in r.orders if o['status']=='FILLED'))
            self.assertLess(attribute(market,r)[2]['max_reconciliation_error'],1e-6)


if __name__=='__main__':unittest.main()

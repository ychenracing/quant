"""Trend confirmation cannot bypass current entry, actual liquidation or risk."""
import importlib
import importlib.util
import unittest
import numpy as np
import pandas as pd
from test_core import sample_market
from techquant.data import Market
from techquant.engine import run
from techquant.policy import CloseObservation


class ReadmissionTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research.readmission'))
        return importlib.import_module('research.readmission')

    def owner(self, n=1):
        module=self.module()
        owner=module.Owner(sample_market(n,145),module.Parameters())
        self.cap(owner,1.)
        return owner

    def cap(self,owner,cap):
        def update(*args):
            owner.risk.cap=cap
            return cap,'CONTROLLED_CAP'
        owner.risk.update=update

    def observation(self,owner,i,units=None):
        units=np.zeros(len(owner.market.symbols)) if units is None else np.array(units,dtype=float)
        price=owner.features.close[i]
        cash=2_000_000.-float(units@price)
        return CloseObservation.from_inventory(i,str(owner.market.calendar[i].date()),
            2_000_000.,cash,units,units*price/2_000_000.)

    def test_one_fixed_candidate_and_exact_parent_identity(self):
        from dataclasses import asdict
        m=self.module()
        self.assertEqual([asdict(p) for p in m.grid()],[{}])
        with self.assertRaises(TypeError):m.Parameters(2)
        owner=self.owner()
        self.assertEqual(asdict(owner.params),{'risk_budget':.10,'positions':2})
        identity=m.Study('readmission').identity()
        self.assertIn('readmission_contract.json',identity['dependencies'])
        self.assertIn('support_budget.py',identity['dependencies'])
        with self.assertRaises(ValueError):m.Study('other')

    def test_healthy_trend_can_precede_trigger_but_cannot_trade_without_it(self):
        owner=self.owner()
        owner.previous_units[:]=1000.
        owner.features.entry[40:42]=False
        for i in (40,41):
            d=owner.decide(self.observation(owner,i))
            self.assertEqual(d.unit_targets[0],0.)
            self.assertTrue(owner.readmit[0])
        d=owner.decide(self.observation(owner,42))
        self.assertEqual(owner.healthy[0],3)
        self.assertFalse(owner.readmit[0])
        self.assertGreater(d.unit_targets[0],0.)

    def test_broken_or_unfresh_trend_restarts_confirmation(self):
        for condition in ('unfresh','below_fast','exit'):
            with self.subTest(condition=condition):
                owner=self.owner()
                owner.previous_units[:]=1000.
                owner.features.entry[40:43]=False
                if condition=='unfresh':owner.ready[41]=False
                elif condition=='below_fast':owner.fast[41]=owner.features.close[41]+1.
                else:owner.features.exit[41]=True
                owner.decide(self.observation(owner,40))
                owner.decide(self.observation(owner,41))
                self.assertEqual(owner.healthy[0],0)
                for i in (42,43):
                    d=owner.decide(self.observation(owner,i))
                    self.assertEqual(d.unit_targets[0],0.)
                self.assertGreater(owner.decide(self.observation(owner,44)).unit_targets[0],0.)

    def test_blocked_exit_cannot_start_a_reentry_episode(self):
        owner=self.owner()
        owner.features.exit[41]=True
        owner.decide(self.observation(owner,40,[1000.]))
        for i in (41,42,43,44):
            d=owner.decide(self.observation(owner,i,[1000.]))
            self.assertEqual(d.unit_targets[0],0.)
            self.assertTrue(owner.exit_pending[0])
        for i in (45,46):
            d=owner.decide(self.observation(owner,i))
            self.assertEqual(d.unit_targets[0],0.)
            self.assertTrue(owner.readmit[0])
        self.assertGreater(owner.decide(self.observation(owner,47)).unit_targets[0],0.)

    def test_recovered_health_never_overrides_closed_account_authority(self):
        owner=self.owner()
        self.cap(owner,0.)
        owner.previous_units[:]=1000.
        for i in (40,41,42,43):
            self.assertEqual(owner.decide(self.observation(owner,i)).unit_targets[0],0.)
        self.assertFalse(owner.readmit[0])
        self.assertEqual(len(owner.history),4)

    def test_prefix_exclusion_and_delayed_cash_accounting(self):
        m=self.module()
        market=sample_market(4,145)
        frames={s:f.copy() for s,f in market.frames.items()}
        for f in frames.values():
            f.loc[market.calendar[65:80],['open','high','low','close','raw_open','raw_close']]*=.82
        market=Market.from_frames(frames,market.calendar,quality='synthetic')
        def execute(current,delay=1):
            return run(current,delay=delay,policy_factory=lambda x,c:m.Owner(x,m.Parameters()))
        full=execute(market);cut=market.calendar[110];short=execute(market.prefix(cut))
        pd.testing.assert_frame_equal(full.equity.loc[:cut],short.equity)
        pd.testing.assert_frame_equal(full.targets.loc[:cut],short.targets)
        frames[market.symbols[-1]].loc[:,['open','high','low','close','raw_open','raw_close']]*=100.
        changed=Market.from_frames(frames,market.calendar,quality='synthetic')
        a,b=execute(market.subset(market.symbols[:-1])),execute(changed.subset(market.symbols[:-1]))
        pd.testing.assert_frame_equal(a.equity,b.equity);self.assertEqual(a.orders,b.orders)
        from research.ledger_attribution import attribute
        for result in (full,execute(market,2)):
            self.assertTrue((result.equity.cash>=0).all())
            self.assertTrue((result.equity.exposure<=1+1e-10).all())
            self.assertTrue(all(o['signal_date']<o['date'] for o in result.orders if o['status']=='FILLED'))
            self.assertLess(attribute(market,result)[2]['max_reconciliation_error'],1e-6)


if __name__=='__main__':unittest.main()

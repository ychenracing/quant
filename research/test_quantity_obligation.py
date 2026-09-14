"""Regression contracts: execution rounding cannot create fresh risk authority."""
from dataclasses import asdict
import importlib
import importlib.util
import unittest
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from test_core import sample_market
from techquant.data import Market
from techquant.engine import run
from techquant.policy import CloseObservation
from research.support_budget import Owner as OldOwner, Parameters as OldParameters


def observe(owner, i, units, cash=2_000_000.):
    units = np.array(units, dtype=float)
    values = np.nan_to_num(units*owner.features.close[i], nan=0.)
    nav = float(cash+values.sum())
    return CloseObservation.from_inventory(i, str(owner.market.calendar[i].date()),
                                          nav, cash, units, values/nav)


def control(owner, cap=1.):
    owner.breakout[:] = False
    def update(*args):
        owner.risk.cap = cap
        return cap, 'CONTROLLED_RISK_AUTHORITY'
    owner.risk.update = update


class QuantityObligationTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research.quantity_obligation'),
                             'independent obligation and declaration adapter is absent')
        return importlib.import_module('research.quantity_obligation')

    def test_original_owner_reproduces_rounding_feedback(self):
        owner = OldOwner(sample_market(1, 100), OldParameters(.10, 2))
        control(owner)
        owner.reduction_ceiling[0] = 999.95
        first = owner.decide(observe(owner, 40, [1000.05]))
        self.assertAlmostEqual(first.unit_targets[0], 900.05)
        self.assertAlmostEqual(owner.reduction_ceiling[0], 900.05)
        # A slightly incomplete conversion leaves inventory just above the
        # rounded declaration, but already below the original risk ceiling.
        owner.decide(observe(owner, 41, [900.05001]))
        self.assertLess(owner.reduction_ceiling[0], 900.05)

    def test_three_fixed_parents_and_no_undeclared_parameter(self):
        module = self.module()
        self.assertEqual([asdict(p) for p in module.grid()],
                         [{'authority': x} for x in ('support','funded','cushion')])
        for bad in ('other', True, None, 1):
            with self.assertRaises(ValueError): module.Parameters(bad)
        from research.finite_study import Study
        identity = Study('quantity_obligation').identity()
        for name in ('quantity_obligation_contract.json','support_budget.py','funded_risk.py'):
            self.assertIn(name, identity['dependencies'])

    def test_blocked_retry_preserves_economic_ceiling_and_rounds_only_declaration(self):
        module = self.module()
        adapter = module.Owner(sample_market(1,100), module.Parameters('support'))
        owner = adapter.inner; control(owner)
        owner.reduction_ceiling[0] = 999.95
        first = adapter.decide(observe(owner,40,[1000.05]))
        self.assertAlmostEqual(first.unit_targets[0],900.05)
        self.assertEqual(owner.reduction_ceiling[0],999.95)
        retry = adapter.decide(observe(owner,41,[1000.05]))
        np.testing.assert_array_equal(first.unit_targets,retry.unit_targets)
        self.assertEqual(owner.reduction_ceiling[0],999.95)
        self.assertEqual(adapter.trace[-1]['economic_ceiling'],999.95)
        self.assertFalse(adapter.trace[-1]['obligation_complete'])

    def test_partial_fill_above_obligation_retries_below_it_without_ratchet(self):
        module = self.module()
        adapter = module.Owner(sample_market(2,100), module.Parameters('support'))
        owner = adapter.inner; control(owner)
        owner.reduction_ceiling[0] = 9900.95
        adapter.decide(observe(owner,40,[10000.05,0.]))
        d = adapter.decide(observe(owner,41,[9950.05,0.]))
        self.assertLessEqual(d.unit_targets[0],9900.95)
        self.assertEqual(owner.reduction_ceiling[0],9900.95)
        self.assertEqual(d.unit_targets[1],0.)

    def test_actual_completion_does_not_chase_previous_rounded_declaration(self):
        module = self.module()
        adapter = module.Owner(sample_market(1,100), module.Parameters('support'))
        owner = adapter.inner; control(owner)
        owner.reduction_ceiling[0] = 999.95
        adapter.decide(observe(owner,40,[1000.05]))
        completed = adapter.decide(observe(owner,41,[900.05001]))
        np.testing.assert_array_equal(completed.unit_targets,[900.05001])
        self.assertTrue(np.isinf(owner.reduction_ceiling[0]))
        self.assertTrue(adapter.trace[-1]['obligation_complete'])

    def test_fresh_risk_can_tighten_ceiling_and_full_exit_still_latches(self):
        module = self.module()
        adapter = module.Owner(sample_market(1,100), module.Parameters('support'))
        owner = adapter.inner; control(owner,.5)
        o = observe(owner,40,[100000.05],cash=20000.)
        expected = .5*o.nav/owner.features.close[40,0]
        first = adapter.decide(o)
        self.assertAlmostEqual(owner.reduction_ceiling[0],expected,places=8)
        self.assertLessEqual(first.unit_targets[0],expected)
        control(owner,.25)
        adapter.decide(observe(owner,41,[100000.05],cash=20000.))
        self.assertLess(owner.reduction_ceiling[0],expected)
        control(owner,1.)
        owner.stop[0] = owner.features.close[42,0]*1.1
        d = adapter.decide(observe(owner,42,[100000.05],cash=20000.))
        self.assertEqual(d.unit_targets[0],0.)
        self.assertTrue(owner.exit_pending[0])
        self.assertEqual(adapter.decide(observe(owner,43,[100000.05])).unit_targets[0],0.)

    def test_study_cache_preserves_trace_integrity_and_benchmarks_remain_loadable(self):
        module = self.module()
        from research.finite_study import Study
        study = Study('quantity_obligation');market = sample_market(2,100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Original strict six-file replay integrity must not be relaxed.
            for name,parameters,benchmark in [('owned',module.Parameters(),None),
                                               ('hold',None,'buy_hold'),('incumbent',None,None)]:
                path = root/'runs'/name
                original = study.saved(market,path,parameters,benchmark)
                cached = study.saved(market,path,parameters,benchmark)
                pd.testing.assert_frame_equal(original.targets.rename_axis('date'),cached.targets,check_freq=False)
            trace = root/'intents'/'owned.json'
            self.assertTrue(trace.is_file())
            saved = json.loads(trace.read_text());saved['identity']['delay'] = 50
            trace.write_text(json.dumps(saved))
            with self.assertRaises(ValueError):
                study.saved(market,root/'runs'/'owned',module.Parameters())

    def test_prefix_removal_and_delay_reconcile_actual_cash_for_each_fixed_parent(self):
        module = self.module()
        market = sample_market(3,140)
        frames = {s:f.copy() for s,f in market.frames.items()}
        for f in frames.values():
            f.loc[market.calendar[65:80],['open','high','low','close','raw_open','raw_close']] *= .82
            # A causal raw/adjusted conversion change must not require future
            # information to express the same economic inventory obligation.
            f.loc[:,['raw_open','raw_close']] *= np.where(np.arange(len(f))%2,1.00001,1.)[:,None]
        market = Market.from_frames(frames,market.calendar,quality='synthetic')
        from research.ledger_attribution import attribute
        for parameters in module.grid():
            def execute(m, delay=1):
                return run(m,delay=delay,policy_factory=lambda current,c:module.Owner(current,parameters))
            full = execute(market);cut = market.calendar[110]
            prefix = execute(market.prefix(cut))
            pd.testing.assert_frame_equal(full.equity.loc[:cut],prefix.equity)
            pd.testing.assert_frame_equal(full.targets.loc[:cut],prefix.targets)
            other = {s:f.copy() for s,f in market.frames.items()}
            other[market.symbols[-1]].loc[:,['open','high','low','close','raw_open','raw_close']] *= 70.
            changed = Market.from_frames(other,market.calendar,quality='synthetic')
            a,b = execute(market.subset(market.symbols[:-1])),execute(changed.subset(market.symbols[:-1]))
            pd.testing.assert_frame_equal(a.equity,b.equity);self.assertEqual(a.orders,b.orders)
            for result in (full,execute(market,2)):
                self.assertTrue((result.equity.cash>=0).all())
                self.assertTrue(all(x['signal_date']<x['date'] for x in result.orders if x['status']=='FILLED'))
                self.assertLess(attribute(market,result)[2]['max_reconciliation_error'],1e-6)


if __name__=='__main__': unittest.main()

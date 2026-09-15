"""A fresh-admission filter cannot relax existing protection or invent inputs."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from techquant.config import Config
from techquant.engine import run
from techquant.policy import CloseObservation
from research.test_allocation_auction import market
from research.observed_admission_completion import Owner as Parent, Parameters as ParentParameters
from research.admission_structure import Owner, Parameters, grid, fresh_admission_mask


class AdmissionStructureTests(unittest.TestCase):
    def test_registered_booleans_only(self):
        self.assertEqual([p.require_stable_support for p in grid()], [False, True])
        for value in (0, 1, None, 'true'):
            with self.assertRaises(ValueError):
                Parameters(value)

    def test_falling_equal_rising_and_missing_history(self):
        current = np.array([9., 10., 11., 10., np.nan])
        previous = np.array([10., 10., 10., np.nan, 10.])
        np.testing.assert_array_equal(fresh_admission_mask(current, previous, np.zeros(5, bool)),
                                      [False, True, True, True, True])
        np.testing.assert_array_equal(fresh_admission_mask(current, previous, np.ones(5, bool)),
                                      np.ones(5, bool))
        with self.assertRaises(ValueError):
            fresh_admission_mask(current, previous[:-1], np.zeros(5, bool))

    def test_filter_only_removes_original_unheld_admissions(self):
        m = market()
        owner = Owner(m, Parameters(True)); p = owner.inner
        i = 40; price = p.features.close[i]; units = np.zeros(len(price))
        p.support[i] = 10.; p.support[i-p.config.fast] = 11.
        obs = CloseObservation.from_inventory(i, str(m.calendar[i].date()), 2e6, 2e6,
                                              units, units)
        wanted = p._allocate(obs, units.copy(), price, np.full(len(price), 9.),
                             np.array([True, False, True]), 1.)
        np.testing.assert_array_equal(wanted, units)
        event = owner.trace[-1]
        self.assertEqual(event['withheld'], [True, False, True])
        self.assertEqual(event['original_allowed'], [True, False, True])
        # A rising base cannot make an originally disallowed stock eligible.
        p.support[i] = 12.
        result = p._allocate(obs, units.copy(), price, np.full(len(price), 9.),
                             np.zeros(len(price), bool), 1.)
        np.testing.assert_array_equal(result, units)

    def test_held_inventory_and_parent_breakout_authority_unchanged(self):
        m = market(names=1); control = Owner(m, Parameters(False)); treatment = Owner(m, Parameters(True))
        i = 40; price = treatment.inner.features.close[i]; units = np.array([10000.])
        obs = CloseObservation.from_inventory(i, str(m.calendar[i].date()), 2e6,
                  float(2e6-units@price), units, units*price/2e6)
        for owner in (control, treatment):
            p = owner.inner; p.support[i] = 10.; p.support[i-p.config.fast] = 11.
            p.stop[:] = price*.95; p.breakout[i] = True
        values = [o.inner._allocate(obs, units.copy(), price, price*.95, np.ones(1,bool), 1.)
                  for o in (control,treatment)]
        np.testing.assert_array_equal(*values)
        self.assertFalse(treatment.trace)

    def test_control_is_exact_parent_account(self):
        m = market(days=120)
        a = run(m, policy_factory=lambda m,c: Parent(m, ParentParameters(), config=c))
        b = run(m, policy_factory=lambda m,c: Owner(m, Parameters(False), config=c))
        pd.testing.assert_frame_equal(a.equity, b.equity)
        pd.testing.assert_frame_equal(a.targets, b.targets)
        self.assertEqual(a.orders, b.orders)

    def test_protective_exit_and_partial_fill_retry_take_precedence(self):
        m = market(); o = Owner(m, Parameters(True)); p = o.inner
        units = np.array([1000., 0., 0.]); p.previous_units = units.copy()
        p.stop[0] = 1e6; p.risk.cap = 1.
        for i,remaining in ((40,1000.),(41,500.)):
            held = np.array([remaining,0.,0.]); price = p.features.close[i]
            obs = CloseObservation.from_inventory(i,str(m.calendar[i].date()),2e6,
                   float(2e6-held@price),held,held*price/2e6)
            d = o.decide(obs)
            self.assertEqual(d.unit_targets[0],0.)
            self.assertTrue(p.exit_pending[0])
            self.assertTrue(np.all(d.unit_targets <= held+1e-10))
        self.assertFalse(any(e.get('kind')=='FRESH_SUPPORT_STRUCTURE' for e in o.trace))

    def test_configuration_controls_lag_without_changing_windows(self):
        m = market(); cfg = replace(Config(),fast=5,slow=20)
        o = Owner(m,Parameters(True),config=cfg); p=o.inner; i=40
        self.assertIs(p.config,cfg)
        p.support[i]=10.;p.support[i-cfg.fast]=11.;p.support[i-10]=9.
        units=np.zeros(3);price=p.features.close[i]
        obs=CloseObservation.from_inventory(i,str(m.calendar[i].date()),2e6,2e6,units,units)
        d=p._allocate(obs,units.copy(),price,price*.95,np.ones(3,bool),1.)
        np.testing.assert_array_equal(d,units)
        self.assertEqual(o.trace[-1]['lag'],5)

    def test_causal_prefix_and_removed_symbol_isolation(self):
        m=market(days=110);factory=lambda m,c: Owner(m,Parameters(True),config=c)
        full=run(m,policy_factory=factory,delay=2);cut=m.calendar[75]
        short=run(m.prefix(cut),policy_factory=factory,delay=2)
        pd.testing.assert_frame_equal(full.equity.loc[:cut],short.equity)
        pd.testing.assert_frame_equal(full.targets.loc[:cut],short.targets)
        self.assertEqual([o for o in full.orders if o['date']<=str(cut.date())],short.orders)
        subset=m.subset(m.symbols[:2]);a=run(subset,policy_factory=factory)
        frames={s:f.copy() for s,f in m.frames.items()};frames[m.symbols[-1]].loc[:,['open','high','low','close','raw_open','raw_close']]*=4
        from techquant.data import Market
        altered=Market.from_frames(frames,m.calendar,sectors=m.sectors,quality=m.quality)
        b=run(altered.subset(m.symbols[:2]),policy_factory=factory)
        pd.testing.assert_frame_equal(a.equity,b.equity);self.assertEqual(a.orders,b.orders)

    def test_trace_cache_integrity_and_configuration_registration(self):
        from research.finite_study import Study
        import json
        study=Study('admission_structure');m=market();cfg=replace(Config(),fast=5,slow=20)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'runs'/'candidate';a=study.saved(m,path,Parameters(True),configuration=cfg)
            b=study.saved(m,path,Parameters(True),configuration=cfg)
            pd.testing.assert_frame_equal(a.equity,b.equity,check_dtype=False,check_exact=True)
            trace=path.parent.parent/'intents'/'candidate.json'
            saved=json.loads(trace.read_text());saved['trace'].append({'tampered':True})
            trace.write_text(json.dumps(saved))
            with self.assertRaises(ValueError):study.saved(m,path,Parameters(True),configuration=cfg)

if __name__=='__main__':unittest.main()

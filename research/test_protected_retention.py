"""Reallocate a new aggregate reduction without relaxing any existing ceiling."""
from dataclasses import replace
from pathlib import Path
import tempfile,unittest,json
import numpy as np
import pandas as pd
from techquant.config import Config
from techquant.engine import run
from techquant.policy import CloseObservation
from research.test_allocation_auction import market
from research.observed_admission_completion import Owner as Parent, Parameters as ParentParameters
from research.protected_retention import Owner, Parameters, grid, retain_supported_strength


class ProtectedRetentionTests(unittest.TestCase):
    def test_declared_boolean_comparison(self):
        self.assertEqual([p.prefer_supported_strength for p in grid()],[False,True])
        for value in (0,1,None,'true'):
            with self.assertRaises(ValueError):Parameters(value)

    def test_same_notional_retains_stronger_lower_risk_inventory(self):
        upper=np.array([40.,40.]);proportional=np.array([25.,25.]);price=np.array([10.,10.])
        risk=np.array([1.,2.]);score=np.array([2.,1.])
        retained=retain_supported_strength(upper,proportional,price,risk,score)
        np.testing.assert_allclose(retained,[40.,10.])
        self.assertAlmostEqual(float(retained@price),float(proportional@price))
        self.assertLess(float(retained@risk),float(proportional@risk))
        self.assertTrue(np.all(retained<=upper))
        np.testing.assert_array_equal(proportional,[25.,25.])

    def test_individual_ceiling_is_not_relaxed(self):
        upper=np.array([28.,40.]);proportional=upper*50/68
        retained=retain_supported_strength(upper,proportional,np.ones(2)*10,np.array([1.,2.]),np.array([2.,1.]))
        np.testing.assert_allclose(retained,[28.,22.])
        self.assertTrue(np.all(retained<=upper))

    def test_higher_risk_tied_score_missing_score_and_capacity_fallback(self):
        for upper,price,distance,score in [
            ([40.,40.],[10.,10.],[2.,1.],[2.,1.]),
            ([40.,40.],[10.,10.],[1.,2.],[1.,1.]),
            ([40.,40.],[10.,10.],[1.,2.],[np.nan,1.]),
            ([40.],[10.],[1.],[2.]),
            ([40.,40.,40.],[10.,10.,10.],[1.,2.,3.],[3.,2.,1.]),
            ([0.,0.],[0.,0.],[0.,0.],[0.,0.])]:
            upper=np.array(upper);proportional=upper*.5
            np.testing.assert_array_equal(retain_supported_strength(upper,proportional,price,distance,score),proportional)
        np.testing.assert_array_equal(retain_supported_strength([40.,40.],[0.,0.],[10.,10.],[1.,2.],[2.,1.]),[0.,0.])
        with self.assertRaises(ValueError):retain_supported_strength([1.],[2.],[10.],[1.],[1.])
        with self.assertRaises(ValueError):retain_supported_strength([1.],[.5],[0.],[1.],[1.])

    def test_feasible_transfer_properties_and_symbol_order_invariance(self):
        rng=np.random.default_rng(34)
        for _ in range(250):
            upper=rng.uniform(1,10000,2);price=rng.uniform(1,300,2)
            distance=price*rng.uniform(.02,.3,2);score=rng.normal(size=2)
            proportional=upper*rng.uniform(0,1)
            retained=retain_supported_strength(upper,proportional,price,distance,score)
            self.assertTrue(np.all(retained>=0) and np.all(retained<=upper))
            self.assertAlmostEqual(float(retained@price)/float(upper@price),float(proportional@price)/float(upper@price),places=13)
            self.assertLessEqual(float(retained@distance),float(proportional@distance)+1e-8)
            self.assertGreaterEqual(float((retained*price)@score),float((proportional*price)@score)-1e-8)
            reversed_result=retain_supported_strength(upper[::-1],proportional[::-1],price[::-1],distance[::-1],score[::-1])
            np.testing.assert_array_equal(retained,reversed_result[::-1])

    def test_original_default_reduction_arithmetic(self):
        parent=Parent(market(),ParentParameters()).inner
        desired=np.array([1.,2.,3.]);price=np.array([10.,20.,30.]);o=CloseObservation.from_inventory(40,'2023-03-01',1000.,860.,desired,desired*price/1000.)
        original=desired.copy();original*=.08/.14
        np.testing.assert_array_equal(parent._reduce_exposure(o,desired,price,.08,.14),original)

    def test_control_account_is_exact_parent(self):
        m=market(days=120)
        control=run(m,policy_factory=lambda m,c: Owner(m,Parameters(False),config=c))
        parent=run(m,policy_factory=lambda m,c: Parent(m,ParentParameters(),config=c))
        pd.testing.assert_frame_equal(control.equity,parent.equity)
        pd.testing.assert_frame_equal(control.targets,parent.targets)
        self.assertEqual(control.orders,parent.orders)

    def test_new_reduction_trace_and_prior_ceiling(self):
        m=market();owner=Owner(m,Parameters(True));p=owner.inner;i=40
        p.features.close[i]=[10.,10.,10.];p.features.score[i]=[2.,1.,0.];p.stop[:]=[9.,8.,0.]
        p.reduction_ceiling[:]=[28.,np.inf,np.inf]
        actual=np.array([40.,40.,0.]);upper=np.array([28.,40.,0.]);o=CloseObservation.from_inventory(i,str(m.calendar[i].date()),1000.,200.,actual,actual*.01)
        selected=p._reduce_exposure(o,upper,np.ones(3)*10,.5,.68)
        np.testing.assert_allclose(selected,[28.,22.,0.])
        self.assertEqual(p.reduction_ceiling[0],28.)
        record=owner.trace[-1];self.assertTrue(record['applied'])
        self.assertLessEqual(record['selected_risk'],record['proportional_risk'])
        self.assertAlmostEqual(record['selected_notional'],record['proportional_notional'])

    def test_pending_full_exit_partial_fills_and_no_new_cash(self):
        m=market();owner=Owner(m,Parameters(True));p=owner.inner
        p.stop[0]=1e6;p.previous_units=np.array([1000.,0.,0.]);p.risk.cap=1.
        for i,left in ((40,1000.),(41,500.)):
            units=np.array([left,0.,0.]);price=p.features.close[i]
            o=CloseObservation.from_inventory(i,str(m.calendar[i].date()),2e6,float(2e6-units@price),units,units*price/2e6)
            decision=owner.decide(o)
            self.assertEqual(decision.unit_targets[0],0.)
            self.assertTrue(p.exit_pending[0]);self.assertTrue(np.all(decision.unit_targets<=units+1e-10))

    def test_configuration_prefix_and_removed_symbol_isolation(self):
        m=market(days=115);cfg=replace(Config(),fast=5,slow=20)
        self.assertIs(Owner(m,Parameters(True),config=cfg).inner.config,cfg)
        factory=lambda m,c: Owner(m,Parameters(True),config=c)
        full=run(m,cfg,policy_factory=factory,delay=2);cut=m.calendar[75]
        short=run(m.prefix(cut),cfg,policy_factory=factory,delay=2)
        pd.testing.assert_frame_equal(full.equity.loc[:cut],short.equity)
        self.assertEqual([o for o in full.orders if o['date']<=str(cut.date())],short.orders)
        frames={s:f.copy() for s,f in m.frames.items()};frames[m.symbols[-1]].loc[:,['open','high','low','close','raw_open','raw_close']]*=2
        from techquant.data import Market
        changed=Market.from_frames(frames,m.calendar,sectors=m.sectors,quality=m.quality)
        a=run(m.subset(m.symbols[:2]),cfg,policy_factory=factory);b=run(changed.subset(m.symbols[:2]),cfg,policy_factory=factory)
        pd.testing.assert_frame_equal(a.equity,b.equity);self.assertEqual(a.orders,b.orders)

    def test_configuration_and_trace_cache_identity(self):
        from research.finite_study import Study
        study=Study('protected_retention');m=market();cfg=replace(Config(),fast=5,slow=20)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'runs'/'candidate';a=study.saved(m,path,Parameters(True),configuration=cfg);b=study.saved(m,path,Parameters(True),configuration=cfg)
            pd.testing.assert_frame_equal(a.equity,b.equity,check_dtype=False,check_exact=True)
            trace=path.parent.parent/'intents'/'candidate.json';saved=json.loads(trace.read_text());saved['trace'].append({'tampered':True});trace.write_text(json.dumps(saved))
            with self.assertRaises(ValueError):study.saved(m,path,Parameters(True),configuration=cfg)

if __name__=='__main__':unittest.main()

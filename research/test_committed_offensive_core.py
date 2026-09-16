"""Contracts for sparse event-driven offensive ownership."""
from dataclasses import asdict, replace
import importlib, importlib.util, unittest
import numpy as np, pandas as pd
from test_core import sample_market
from techquant.engine import run
from techquant.policy import CloseObservation
from research.trend_book import Owner as ParentOwner, Parameters as ParentParameters


def obs(owner, session, units, cash):
    units=np.asarray(units,dtype=float)
    marks=owner.price_signals.price[session]
    values=units*marks
    nav=float(cash+values.sum())
    return CloseObservation.from_inventory(session,str(owner.market.calendar[session].date()),nav,cash,units,values/nav)


class CommittedOffensiveCoreTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research.committed_offensive_core'),
                             'committed offensive core implementation is absent')
        return importlib.import_module('research.committed_offensive_core')

    def test_control_is_exact_parent(self):
        m=self.module(); self.assertEqual([asdict(p) for p in m.grid()],[{'enabled':False},{'enabled':True}])
        market=sample_market(5,145)
        parent=run(market,policy_factory=lambda current,cfg:ParentOwner(current,ParentParameters(2)))
        control=run(market,policy_factory=lambda current,cfg:m.Owner(current,m.Parameters(False)))
        pd.testing.assert_frame_equal(parent.equity,control.equity); pd.testing.assert_frame_equal(parent.targets,control.targets)
        self.assertEqual(parent.orders,control.orders)

    def test_registered_with_existing_paired_overlay(self):
        self.module()
        from research.finite_study import Study
        study=Study('committed_offensive_core')
        self.assertEqual(study.family,'committed_offensive_core')
        self.assertEqual(len(study.module.grid()),2)
        self.assertIn('committed_offensive_core.py',study.identity()['dependencies'])

    def test_initial_cash_selects_two_and_uses_full_observed_cash(self):
        m=self.module(); owner=m.Owner(sample_market(5,145),m.Parameters(True))
        d=owner.decide(obs(owner,100,np.zeros(5),2_000_000.))
        self.assertEqual(np.count_nonzero(d.weights),2); self.assertAlmostEqual(float(d.weights.sum()),1.0,12)
        self.assertIn('COMMITTED_OFFENSIVE_FILL',d.reason)

    def test_intact_owned_names_cannot_be_displaced_by_new_higher_rank(self):
        m=self.module(); owner=m.Owner(sample_market(3,145),m.Parameters(True))
        units=np.array([10_000.,10_000.,0.]); current=obs(owner,101,units,200_000.)
        score=owner.features.score.copy(); score[101]=np.array([1.,2.,100.]); owner.features=replace(owner.features,score=score)
        d=owner.decide(current)
        np.testing.assert_allclose(d.weights,current.weights,rtol=0,atol=1e-12)
        self.assertIn('COMMITTED_OFFENSIVE_HOLD',d.reason)

    def test_security_break_sells_without_same_close_replacement(self):
        m=self.module(); owner=m.Owner(sample_market(3,145),m.Parameters(True))
        units=np.array([10_000.,10_000.,0.]); session=101
        ex=owner.trend.exit.copy(); en=owner.trend.entry.copy(); ex[session,0]=True; en[session,0]=True
        owner.trend=replace(owner.trend,exit=ex,entry=en)
        score=owner.features.score.copy(); score[session]=np.array([100.,2.,50.]); owner.features=replace(owner.features,score=score)
        current=obs(owner,session,units,100_000.); d=owner.decide(current)
        self.assertEqual(float(d.weights[0]),0.0); self.assertEqual(float(d.weights[2]),0.0)
        self.assertAlmostEqual(float(d.weights[1]),float(current.weights[1]),12)
        self.assertIn('SECURITY_EXIT',d.reason)

    def test_real_cash_after_exit_can_fill_vacancy_without_trimming_survivor(self):
        m=self.module(); owner=m.Owner(sample_market(3,145),m.Parameters(True)); session=102
        units=np.array([0.,10_000.,0.]); current=obs(owner,session,units,900_000.)
        score=owner.features.score.copy(); score[session]=np.array([1.,2.,50.]); owner.features=replace(owner.features,score=score)
        d=owner.decide(current)
        self.assertAlmostEqual(float(d.weights[1]),float(current.weights[1]),12)
        self.assertGreater(float(d.weights[2]),0.0)
        self.assertLessEqual(float(d.weights.sum()),1.0+1e-12)
        self.assertIn('COMMITTED_OFFENSIVE_FILL',d.reason)

    def test_vacancy_fill_with_zero_marks_keeps_unit_targets_finite(self):
        m=self.module(); owner=m.Owner(sample_market(3,145),m.Parameters(True)); session=101
        price=owner.price_signals.price.copy(); ready=owner.price_signals.ready.copy()
        ema20=owner.price_signals.ema20.copy(); mom=owner.price_signals.momentum5.copy()
        price[session]=np.array([592.1984985685666,638.0019293754939,0.0])
        ready[session]=np.array([True,True,False]); ema20[session]=np.array([500.,500.,0.]); mom[session]=np.array([.1,.1,0.])
        owner.price_signals=replace(owner.price_signals,price=price,ready=ready,ema20=ema20,momentum5=mom)
        entry=owner.trend.entry.copy(); exit_=owner.trend.exit.copy(); market=owner.trend.market.copy()
        entry[session]=np.array([True,True,False]); exit_[session]=False; market[session]=True
        owner.trend=replace(owner.trend,entry=entry,exit=exit_,market=market)
        score=owner.features.score.copy(); score[session]=np.array([1.,2.,np.nan]); owner.features=replace(owner.features,score=score)
        units=np.array([30851.32349796713,0.,0.]); current=obs(owner,session,units,1330820.257558683)
        d=owner.decide(current)
        self.assertTrue(np.isfinite(d.unit_targets).all())
        d.validated_unit_targets(owner.price_signals.price[session],current.nav)

    def test_weak_market_gates_new_cash_but_not_intact_inventory(self):
        m=self.module(); owner=m.Owner(sample_market(2,145),m.Parameters(True)); session=101
        units=np.array([10_000.,0.]); current=obs(owner,session,units,900_000.)
        state=owner.trend.market.copy(); state[session]=False; owner.trend=replace(owner.trend,market=state)
        d=owner.decide(current)
        np.testing.assert_allclose(d.weights,current.weights,rtol=0,atol=1e-12)
        self.assertIn('COMMITTED_OFFENSIVE_HOLD',d.reason)

if __name__=='__main__': unittest.main()

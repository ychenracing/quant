"""Joint funding must respect the original actual-account constraints."""
import unittest
from dataclasses import replace
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from techquant.config import Config
from techquant.data import Market
from techquant.engine import run
from research.allocation_auction import Owner, Parameters, purchase_plan
from research.observed_admission_completion import Owner as Parent, Parameters as ParentParameters
from research.quantity_obligation import preserve_trace, verify_trace


def market(days=100, names=3):
    dates = pd.bdate_range('2023-01-03', periods=days)
    frames = {}
    for j in range(names):
        close = 20*np.exp(np.arange(days)*(.004+j*.0005)+.02*np.sin(np.arange(days)/5+j))
        frames[f'sz300{100+j:03}'] = pd.DataFrame(dict(open=close*.999,high=close*1.01,
            low=close*.99,close=close,raw_open=close*.999,raw_close=close,
            volume=np.full(days,2e7)), index=dates)
    return Market.from_frames(frames, dates, sectors={s:'tech' for s in frames}, quality='synthetic')


class AuctionTests(unittest.TestCase):
    def plan(self, **changes):
        inputs=dict(scores=np.array([10.,9.,8.]),fractions=np.array([.5,.1,.1]),
            upper=np.array([.55,.55,.55]),minimum=np.array([.01,.01,.01]),
            sectors=('a','b','c'),sector_slack={'a':1.,'b':1.,'c':1.},
            cash=1.,risk=.1,held=np.zeros(3,dtype=bool),capacity=2,
            symbols=('sz300100','sz300101','sz300102'))
        inputs.update(changes)
        return purchase_plan(**inputs)

    def test_joint_feasible_choice_beats_greedy_high_score(self):
        actual=self.plan()
        np.testing.assert_allclose(actual,[0.,.55,.45],rtol=0,atol=1e-12)
        self.assertAlmostEqual(float(actual@np.array([10.,9.,8.])),8.55)
        self.assertLessEqual(float(actual@np.array([.5,.1,.1])),.1+1e-12)

    def test_same_sector_and_minimum_leg_constraints(self):
        a=self.plan(sectors=('x','x','x'),sector_slack={'x':.75})
        self.assertLessEqual(float(a.sum()),.75+1e-12)
        np.testing.assert_allclose(a,[0.,.55,.2],rtol=0,atol=1e-12)
        a=self.plan(cash=.015,minimum=np.full(3,.01))
        self.assertEqual(np.count_nonzero(a),1)
        self.assertTrue(np.all(a[a>0]>=.01-1e-12))
        self.assertFalse(self.plan(cash=.005).any())
        self.assertFalse(self.plan(risk=0.).any())

    def test_actual_occupied_slots_are_not_assumed_sold(self):
        a=self.plan(held=np.array([True,True,False]))
        self.assertEqual(a[2],0.)
        a=self.plan(held=np.array([True,False,False]))
        self.assertLessEqual(np.count_nonzero(a[1:]),1)

    def test_singleton_and_no_risk_use_nonnegative_real_resources(self):
        a=purchase_plan(scores=np.array([2.]),fractions=np.array([.2]),
            upper=np.array([1.]),minimum=np.array([.01]),sectors=('x',),
            sector_slack={'x':1.},cash=.4,risk=.05,held=np.array([False]),
            capacity=1,symbols=('sz300100',))
        np.testing.assert_allclose(a,[.25],rtol=0,atol=1e-12)
        with self.assertRaises(ValueError):self.plan(cash=float('nan'))
        with self.assertRaises(ValueError):Parameters(joint='yes')

    def test_feasible_solution_dominates_an_independent_discrete_grid(self):
        # Independent feasible-grid lower bound, not the solver's own vertices.
        for risk in (.025, .05, .1):
            for cash in (.1, .5, 1.):
                answer=self.plan(risk=risk,cash=cash)
                scores=np.array([10.,9.,8.]);rho=np.array([.5,.1,.1])
                best=0.
                for a in np.arange(0.,.551,.05):
                    for b in np.arange(0.,.551,.05):
                        for c in np.arange(0.,.551,.05):
                            x=np.array([a,b,c])
                            if np.count_nonzero(x)>2 or x.sum()>cash+1e-12 or x@rho>risk+1e-12:
                                continue
                            best=max(best,float(x@scores))
                self.assertGreaterEqual(float(answer@scores)+1e-12,best)
                self.assertLessEqual(float(answer.sum()),cash+1e-12)
                self.assertLessEqual(float(answer@rho),risk+1e-12)

    def test_study_identity_and_trace_dependency_coverage(self):
        from research.finite_study import Study
        study=Study('allocation_auction')
        self.assertEqual([p.joint for p in study.module.grid()],[False,True])
        deps=study.identity()['dependencies']
        for name in ('allocation_auction.py','support_budget.py','quantity_obligation.py',
                     'observed_readiness.py','admission_budget_completion.py',
                     'observed_admission_completion.py'):
            self.assertIn(name,deps)
        with tempfile.TemporaryDirectory() as directory:
            m=market(50,2);path=Path(directory)/'runs'/'candidate'
            result=study.saved(m,path,Parameters(True))
            study.saved(m,path,Parameters(True))
            trace=path.parent.parent/'intents'/(path.name+'.json')
            trace.write_text(trace.read_text().replace('JOINT_FUNDED_ALLOCATION','ALTERED'))
            with self.assertRaises(ValueError):study.saved(m,path,Parameters(True))

    def test_control_is_the_unmodified_parent_account(self):
        m=market()
        a=run(m,policy_factory=lambda m,c:Owner(m,Parameters(False),config=c))
        b=run(m,policy_factory=lambda m,c:Parent(m,ParentParameters(),config=c))
        pd.testing.assert_frame_equal(a.equity,b.equity,check_exact=True)
        pd.testing.assert_frame_equal(a.targets,b.targets,check_exact=True)
        self.assertEqual(a.orders,b.orders)

    def test_prefix_configuration_and_removed_input_isolation(self):
        m=market(100,4);cfg=replace(Config(),fast=9,slow=44)
        full=run(m,cfg,policy_factory=lambda m,c:Owner(m,Parameters(True),config=c))
        cut=m.calendar[65]
        short=run(m.prefix(cut),cfg,policy_factory=lambda m,c:Owner(m,Parameters(True),config=c))
        pd.testing.assert_frame_equal(full.equity.loc[:cut],short.equity,check_exact=True)
        pd.testing.assert_frame_equal(full.targets.loc[:cut],short.targets,check_exact=True)
        self.assertEqual([o for o in full.orders if o['date']<=str(cut.date())],short.orders)
        selected=list(m.symbols[:2]);base=m.subset(selected)
        frames={s:f.copy() for s,f in m.frames.items()}
        frames[m.symbols[-1]][['open','high','low','close','raw_open','raw_close']]*=2
        modified=Market.from_frames(frames,m.calendar,sectors=m.sectors,quality='synthetic').subset(selected)
        a=Owner(base,Parameters(True),config=cfg);b=Owner(modified,Parameters(True),config=cfg)
        self.assertEqual(a.inner.config,cfg)
        np.testing.assert_array_equal(a.inner.features.score,b.inner.features.score)
        self.assertEqual(a.inner.config,Parent(base,ParentParameters(),config=cfg).inner.config)

    def test_protection_has_priority_and_trace_integrity(self):
        m=market(100,2)
        owners=[]
        def factory(m,c):
            owner=Owner(m,Parameters(True),config=c);owners.append(owner);return owner
        frames={s:f.copy() for s,f in m.frames.items()}
        for frame in frames.values():
            frame.loc[m.calendar[60]:,['open','high','low','close','raw_open','raw_close']]*=.5
        crashed=Market.from_frames(frames,m.calendar,sectors=m.sectors,quality='synthetic')
        r=run(crashed,policy_factory=factory);owner=owners[0]
        self.assertTrue(any(o['side']=='SELL' and o['status']=='FILLED' for o in r.orders))
        auction=[t for t in owner.trace if t.get('kind')=='JOINT_FUNDED_ALLOCATION']
        self.assertTrue(auction)
        for t in auction:
            self.assertGreater(t['risk_cap'],0.)
            self.assertTrue(all(x>=y-1e-10 for x,y in zip(t['proposed_units'],t['actual_units'])))
            purchases=np.array(t['purchase_weights']);rho=np.array(t['risk_fractions'])
            self.assertLessEqual(float(purchases.sum()),t['cash_fraction']+1e-12)
            self.assertLessEqual(float(purchases@rho),t['risk_fraction']+1e-12)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'trace.json';identity=owner.identity()
            preserve_trace(path,identity,owner.trace);verify_trace(path,identity)
            path.write_text(path.read_text().replace('JOINT_FUNDED_ALLOCATION','ALTERED'))
            with self.assertRaises(ValueError):verify_trace(path,identity)


if __name__=='__main__':unittest.main()

from dataclasses import asdict, replace
import importlib, unittest
import numpy as np
import pandas as pd
from test_core import sample_market
from techquant.engine import run
from techquant.policy import CloseObservation
from research.trend_book import Owner as ParentOwner, Parameters as ParentParameters


def obs(owner, session, units, cash):
    units=np.asarray(units,dtype=float)
    marks=owner.base.price_signals.price[session]
    values=units*marks; nav=float(cash+values.sum())
    return CloseObservation.from_inventory(session,str(owner.market.calendar[session].date()),nav,cash,units,values/nav)


def force(owner, session, scores, entries):
    b=owner.base
    f=b.features.score.copy(); f[session]=np.asarray(scores,float); b.features=replace(b.features,score=f)
    p=b.price_signals
    ready=p.ready.copy(); ready[session]=True
    price=p.price.copy(); ema20=p.ema20.copy(); mom=p.momentum5.copy(); ret1=p.ret1.copy()
    ema20[session]=np.where(np.isfinite(price[session]),price[session]*.9,0); mom[session]=.1; ret1[session]=0
    b.price_signals=replace(p,ready=ready,ema20=ema20,momentum5=mom,ret1=ret1)
    t=b.trend; entry=t.entry.copy(); exit_=t.exit.copy(); market=t.market.copy()
    entry[session]=np.asarray(entries,bool); exit_[session]=False; market[session]=True
    b.trend=replace(t,entry=entry,exit=exit_,market=market)


class DeployableVacancyTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('research.offensive_deployable_vacancy_displacement')

    def test_control_is_exact_parent(self):
        m=self.module(); self.assertEqual([asdict(p) for p in m.grid()],[{'enabled':False},{'enabled':True}])
        market=sample_market(6,145)
        parent=run(market,policy_factory=lambda current,cfg:ParentOwner(current,ParentParameters(2)))
        control=run(market,policy_factory=lambda current,cfg:m.Owner(current,m.Parameters(False)))
        pd.testing.assert_frame_equal(parent.equity,control.equity)
        pd.testing.assert_frame_equal(parent.targets,control.targets)
        self.assertEqual(parent.orders,control.orders)

    def prepared(self, cash_fraction):
        m=self.module(); owner=m.Owner(sample_market(6,145),m.Parameters(True)); n=6; session=100
        scores=np.array([1.,3.,.5,.4,.3,.2]); entries=np.array([True,True,False,False,False,False])
        force(owner,session,scores,entries)
        marks=owner.base.price_signals.price[session]
        units=np.zeros(n); units[0]=1000.0
        holdings=float(units[0]*marks[0]); cash=holdings*cash_fraction/(1-cash_fraction)
        owner.base.was_held[0]=True; owner.base.owned_alpha_reference[0]=2.0; owner.base.last_session=session-1
        return owner,obs(owner,session,units,cash)

    def test_nondeployable_nominal_vacancy_allows_displacement(self):
        owner,o=self.prepared(.005)
        d=owner.decide(o)
        self.assertEqual(d.unit_targets[0],0.0)
        self.assertEqual(d.unit_targets[1],0.0)
        self.assertTrue(owner.base.retired[0])
        self.assertIn('CAPITAL_FULL_DISPLACEMENT',d.reason)
        events=[r for r in owner.trace if r.get('action')=='CAPITAL_FULL_DISPLACEMENT']
        self.assertEqual(len(events),1); self.assertEqual(events[0]['challenger'],owner.market.symbols[1])

    def test_deployable_vacancy_keeps_incumbent_and_uses_observed_cash(self):
        owner,o=self.prepared(.02)
        d=owner.decide(o)
        self.assertGreater(d.unit_targets[0],0.0)
        self.assertGreater(d.unit_targets[1],0.0)
        self.assertFalse(owner.base.retired[0])
        self.assertNotIn('CAPITAL_FULL_DISPLACEMENT',d.reason)

    def test_nondeployable_vacancy_without_alpha_decay_does_not_force_sale(self):
        owner,o=self.prepared(.005)
        owner.base.owned_alpha_reference[0]=.5
        d=owner.decide(o)
        self.assertGreater(d.unit_targets[0],0.0)
        self.assertFalse(owner.base.retired[0])


if __name__=='__main__': unittest.main()

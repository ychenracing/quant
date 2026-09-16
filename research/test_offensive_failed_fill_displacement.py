from dataclasses import replace
import importlib, importlib.util, unittest
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


class FailedFillDisplacementTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research.offensive_failed_fill_displacement'), 'failed-fill implementation is absent')
        return importlib.import_module('research.offensive_failed_fill_displacement')

    def test_control_is_exact_parent(self):
        m=self.module(); market=sample_market(6,145)
        parent=run(market,policy_factory=lambda current,cfg:ParentOwner(current,ParentParameters(2)))
        control=run(market,policy_factory=lambda current,cfg:m.Owner(current,m.Parameters(False)))
        pd.testing.assert_frame_equal(parent.equity,control.equity); pd.testing.assert_frame_equal(parent.targets,control.targets); self.assertEqual(parent.orders,control.orders)

    def prepared(self):
        m=self.module(); owner=m.Owner(sample_market(6,145),m.Parameters(True)); n=6
        units=np.zeros(n); session=100
        force(owner,session,[1.,3.,.5,.4,.3,.2],[True,True,False,False,False,False])
        marks=owner.base.price_signals.price[session]; units[0]=1000.; holdings=float(units[0]*marks[0]); cash=holdings*.005/.995
        owner.base.was_held[0]=True; owner.base.owned_alpha_reference[0]=2.; owner.base.last_session=session-1
        return owner,units,cash

    def test_first_nondeployable_vacancy_attempt_does_not_sell(self):
        owner,units,cash=self.prepared(); o=obs(owner,100,units,cash)
        d=owner.decide(o)
        self.assertGreater(d.unit_targets[0],0.)
        self.assertGreater(d.unit_targets[1],0.)
        self.assertFalse(owner.base.retired[0])
        self.assertEqual(owner.failed_fill_attempt_session,100)

    def test_next_close_without_new_holding_confirms_and_displaces(self):
        owner,units,cash=self.prepared(); owner.decide(obs(owner,100,units,cash))
        force(owner,101,[1.,3.,.5,.4,.3,.2],[True,True,False,False,False,False])
        d=owner.decide(obs(owner,101,units,cash))
        self.assertEqual(d.unit_targets[0],0.)
        self.assertEqual(d.unit_targets[1],0.)
        self.assertTrue(owner.base.retired[0])
        self.assertIn('FAILED_FILL_DISPLACEMENT',d.reason)

    def test_successful_new_holding_cancels_failed_fill_confirmation(self):
        owner,units,cash=self.prepared(); owner.decide(obs(owner,100,units,cash))
        units2=units.copy(); units2[1]=10.
        force(owner,101,[1.,3.,.5,.4,.3,.2],[True,True,False,False,False,False])
        d=owner.decide(obs(owner,101,units2,0.0))
        self.assertGreater(d.unit_targets[0],0.)
        self.assertFalse(any(r.get('action')=='FAILED_FILL_DISPLACEMENT' for r in owner.trace))


if __name__=='__main__': unittest.main()

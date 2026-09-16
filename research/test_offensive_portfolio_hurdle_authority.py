from dataclasses import replace
import importlib
import importlib.util
import unittest

import numpy as np
import pandas as pd

from test_core import sample_market
from techquant.engine import run
from techquant.policy import CloseObservation
from research.trend_book import Owner as TrendOwner, Parameters as TrendParameters


def obs(owner, session, units, cash):
    units=np.asarray(units,dtype=float)
    marks=owner.parent.base.price_signals.price[session]
    values=units*marks; nav=float(cash+values.sum())
    return CloseObservation.from_inventory(session,str(owner.market.calendar[session].date()),nav,cash,units,values/nav)


def force(owner, session, scores, entries):
    b=owner.parent.base
    f=b.features.score.copy(); f[session]=np.asarray(scores,float); b.features=replace(b.features,score=f)
    p=b.price_signals; ready=p.ready.copy(); ready[session]=True
    ema=p.ema20.copy(); ema[session]=np.where(np.isfinite(p.price[session]),p.price[session]*.9,0)
    mom=p.momentum5.copy(); mom[session]=.1
    ret1=p.ret1.copy(); ret1[session]=0
    b.price_signals=replace(p,ready=ready,ema20=ema,momentum5=mom,ret1=ret1)
    t=b.trend; entry=t.entry.copy(); entry[session]=np.asarray(entries,bool)
    exit_=t.exit.copy(); exit_[session]=False; market=t.market.copy(); market[session]=True
    b.trend=replace(t,entry=entry,exit=exit_,market=market)


class PortfolioHurdleAuthorityTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research.offensive_portfolio_hurdle_authority'),
                             'portfolio hurdle authority implementation is absent')
        return importlib.import_module('research.offensive_portfolio_hurdle_authority')

    def prepared(self, challenger_score=3.0):
        m=self.module(); owner=m.Owner(sample_market(6,145),m.Parameters(True)); units=np.zeros(6)
        force(owner,100,[1.0,challenger_score,1.5,3.4,.3,.2],[True,True,True,False,False,False])
        units[0]=1000.; units[3]=1000.
        b=owner.parent.base; b.was_held[[0,3]]=True
        b.owned_alpha_reference[0]=2.0; b.owned_alpha_reference[3]=3.5; b.last_session=99
        return owner,units

    def test_control_is_exact_trend_book(self):
        m=self.module(); market=sample_market(6,145)
        parent=run(market,policy_factory=lambda current,cfg:TrendOwner(current,TrendParameters(2)))
        control=run(market,policy_factory=lambda current,cfg:m.Owner(current,m.Parameters(False)))
        pd.testing.assert_frame_equal(parent.equity,control.equity)
        pd.testing.assert_frame_equal(parent.targets,control.targets)
        self.assertEqual(parent.orders,control.orders)

    def test_marginal_challenger_cannot_clear_stronger_campaign_hurdle(self):
        owner,units=self.prepared(3.0)
        decision=owner.decide(obs(owner,100,units,0.0))
        self.assertGreater(decision.unit_targets[0],0.0)
        self.assertGreater(decision.unit_targets[3],0.0)
        self.assertTrue(any(r.get('action')=='PORTFOLIO_HURDLE_BLOCK' for r in owner.trace))

    def test_dominant_challenger_can_displace_weakest_incumbent_same_close(self):
        owner,units=self.prepared(4.0)
        decision=owner.decide(obs(owner,100,units,0.0))
        self.assertEqual(decision.unit_targets[0],0.0)
        self.assertGreater(decision.unit_targets[3],0.0)
        self.assertTrue(any(r.get('action')=='PORTFOLIO_HURDLE_AUTHORIZED' for r in owner.trace))

    def test_real_vacancy_remains_parent_authoritative(self):
        m=self.module(); owner=m.Owner(sample_market(6,145),m.Parameters(True)); units=np.zeros(6); units[0]=1000.
        force(owner,100,[1.,4.,2.,.5,.3,.2],[True,True,True,False,False,False])
        b=owner.parent.base; b.was_held[0]=True; b.owned_alpha_reference[0]=2.; b.last_session=99
        decision=owner.decide(obs(owner,100,units,1_000_000.0))
        self.assertGreater(decision.unit_targets[1],0.0)
        self.assertFalse(any(r.get('kind')=='PORTFOLIO_HURDLE_AUTHORITY_EVENT' for r in owner.trace))


if __name__=='__main__': unittest.main()

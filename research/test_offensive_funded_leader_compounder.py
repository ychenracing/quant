from dataclasses import replace
import importlib
import importlib.util
import unittest

import numpy as np
import pandas as pd

from test_core import sample_market
from techquant.engine import run
from techquant.policy import CloseObservation
from research.offensive_campaign_peak_authority import Owner as ChampionOwner, Parameters as ChampionParameters


def base(owner):
    return owner.parent.parent.base


def fresh(owner):
    return owner.parent.parent


def observe(owner, session, units, cash=0.0):
    units=np.asarray(units,float); marks=base(owner).price_signals.price[session]; values=units*marks; nav=float(cash+values.sum())
    return CloseObservation.from_inventory(session,str(owner.market.calendar[session].date()),nav,cash,units,np.divide(values,nav,out=np.zeros_like(values),where=nav>0))


def force(owner,session,scores,entries,*,exits=None,ret1=None,ready=None):
    b=base(owner); f=b.features.score.copy(); f[session]=np.asarray(scores,float); b.features=replace(b.features,score=f)
    p=b.price_signals; rd=p.ready.copy(); rd[session]=True if ready is None else np.asarray(ready,bool); ema=p.ema20.copy(); ema[session]=np.where(np.isfinite(p.price[session]),p.price[session]*.9,0); mom=p.momentum5.copy(); mom[session]=.1; r=p.ret1.copy(); r[session]=0 if ret1 is None else np.asarray(ret1,float); b.price_signals=replace(p,ready=rd,ema20=ema,momentum5=mom,ret1=r)
    t=b.trend; en=t.entry.copy(); en[session]=np.asarray(entries,bool); ex=t.exit.copy(); ex[session]=False if exits is None else np.asarray(exits,bool); mk=t.market.copy(); mk[session]=True; b.trend=replace(t,entry=en,exit=ex,market=mk)


class FundedLeaderCompounderTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research.offensive_funded_leader_compounder'),'funded leader compounder implementation is absent')
        return importlib.import_module('research.offensive_funded_leader_compounder')

    def prepared(self):
        m=self.module(); o=m.Owner(sample_market(6,145),m.Parameters(True)); units=np.zeros(6); units[0]=1000.; units[3]=1000.; o.observed_units=units.copy(); b=base(o); b.was_held[[0,3]]=True; b.owned_alpha_reference[0]=2.; b.owned_alpha_reference[3]=.5; b.last_session=99; o.parent.actual_was_held[[0,3]]=True; closes=b.price_signals.price[100]; o.parent.campaign_peak_close[[0,3]]=closes[[0,3]]; o.acquisition_basis[0]=closes[0]*.9; o.acquisition_basis[3]=closes[3]*.9; return o,units

    def test_disabled_is_exact_campaign_peak_champion(self):
        m=self.module(); market=sample_market(6,145); champion=run(market,policy_factory=lambda current,cfg:ChampionOwner(current,ChampionParameters(True))); control=run(market,policy_factory=lambda current,cfg:m.Owner(current,m.Parameters(False))); pd.testing.assert_frame_equal(champion.equity,control.equity); pd.testing.assert_frame_equal(champion.targets,control.targets); self.assertEqual(champion.orders,control.orders)

    def test_top_one_funded_profitable_alpha_improver_latches_leader(self):
        o,units=self.prepared(); force(o,100,[3.,2.5,.2,.6,.1,.1],[True,True,False,True,False,False]); o.decide(observe(o,100,units)); self.assertTrue(o.leader_compounder[0]); self.assertTrue(any(r.get('action')=='FUNDED_LEADER_PROVEN' and o.market.symbols[0] in r.get('symbols',[]) for r in o.trace))

    def test_latched_leader_ignores_generic_slow_trend_exit(self):
        o,units=self.prepared(); o.leader_compounder[0]=True; force(o,100,[1.5,.2,.1,.6,.1,.1],[True,False,False,True,False,False],exits=[True,False,False,False,False,False]); d=o.decide(observe(o,100,units)); self.assertGreater(d.unit_targets[0],0.); self.assertTrue(any(r.get('action')=='FUNDED_LEADER_SLOW_EXIT_IGNORED' for r in o.trace))

    def test_nonleader_does_not_earn_slow_exit_immunity(self):
        o,units=self.prepared(); force(o,100,[2.2,3.,.2,.6,.1,.1],[True,True,False,True,False,False],exits=[True,False,False,False,False,False]); d=o.decide(observe(o,100,units)); self.assertFalse(o.leader_compounder[0]); self.assertEqual(d.unit_targets[0],0.)

    def test_acute_loss_remains_authoritative_for_latched_leader(self):
        o,units=self.prepared(); o.leader_compounder[0]=True; force(o,100,[3.,.2,.1,.6,.1,.1],[True,False,False,True,False,False],ret1=[-.09,0,0,0,0,0]); d=o.decide(observe(o,100,units)); self.assertEqual(d.unit_targets[0],0.)

    def test_actual_addition_updates_weighted_basis_and_flat_clears_latch(self):
        o,_=self.prepared(); o.observed_units[:]=0.; o.acquisition_basis[:]=np.nan; units=np.zeros(6); units[0]=1000.; force(o,100,[3.,.2,.1,.6,.1,.1],[True,False,False,True,False,False]); o.decide(observe(o,100,units)); first=o.execution_open[100,0]; self.assertAlmostEqual(o.acquisition_basis[0],first); prior=o.acquisition_basis[0]; units[0]=1500.; base(o).last_session=100; force(o,101,[3.,.2,.1,.6,.1,.1],[True,False,False,True,False,False]); o.decide(observe(o,101,units)); expected=(1000.*prior+500.*o.execution_open[101,0])/1500.; self.assertAlmostEqual(o.acquisition_basis[0],expected); o.leader_compounder[0]=True; units[0]=0.; base(o).last_session=101; force(o,102,[3.,.2,.1,.6,.1,.1],[True,False,False,True,False,False]); o.decide(observe(o,102,units)); self.assertTrue(np.isnan(o.acquisition_basis[0])); self.assertFalse(o.leader_compounder[0])


if __name__=='__main__': unittest.main()

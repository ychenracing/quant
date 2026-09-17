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


def base(owner): return owner.parent.base

def observe(owner,session,units,cash=0.0):
    units=np.asarray(units,float); marks=base(owner).price_signals.price[session]; values=units*marks; nav=float(cash+values.sum()); return CloseObservation.from_inventory(session,str(owner.market.calendar[session].date()),nav,cash,units,np.divide(values,nav,out=np.zeros_like(values),where=nav>0))

def force(owner,session,scores,entries,*,ret1=None):
    b=base(owner); f=b.features.score.copy(); f[session]=np.asarray(scores,float); b.features=replace(b.features,score=f); p=b.price_signals; ready=p.ready.copy(); ready[session]=True; ema=p.ema20.copy(); ema[session]=np.where(np.isfinite(p.price[session]),p.price[session]*.9,0); mom=p.momentum5.copy(); mom[session]=.1; r=p.ret1.copy(); r[session]=0 if ret1 is None else np.asarray(ret1,float); b.price_signals=replace(p,ready=ready,ema20=ema,momentum5=mom,ret1=r); t=b.trend; en=t.entry.copy(); en[session]=np.asarray(entries,bool); ex=t.exit.copy(); ex[session]=False; mk=t.market.copy(); mk[session]=True; b.trend=replace(t,entry=en,exit=ex,market=mk)

class DominantPeakAuthorityTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research.offensive_dominant_peak_authority'),'dominant peak authority implementation is absent'); return importlib.import_module('research.offensive_dominant_peak_authority')
    def prepared(self):
        m=self.module(); o=m.Owner(sample_market(6,145),m.Parameters(True)); units=np.zeros(6); units[0]=1000.; units[3]=1000.; b=base(o); b.was_held[[0,3]]=True; b.owned_alpha_reference[0]=2.; b.owned_alpha_reference[3]=.5; b.last_session=99; o.actual_was_held[[0,3]]=True; close=b.price_signals.price[100]; o.campaign_peak_close[[0,3]]=close[[0,3]]; return o,units
    def test_disabled_is_exact_campaign_peak_champion(self):
        m=self.module(); market=sample_market(6,145); champion=run(market,policy_factory=lambda current,cfg:ChampionOwner(current,ChampionParameters(True))); control=run(market,policy_factory=lambda current,cfg:m.Owner(current,m.Parameters(False))); pd.testing.assert_frame_equal(champion.equity,control.equity); pd.testing.assert_frame_equal(champion.targets,control.targets); self.assertEqual(champion.orders,control.orders)
    def test_dominant_peak_incumbent_blocks_fresh_challenger(self):
        o,units=self.prepared(); force(o,100,[1.0,3.0,.2,.7,.1,.1],[True,True,False,True,False,False]); base(o).trend.entry[99,1]=False; d=o.decide(observe(o,100,units)); self.assertGreater(d.unit_targets[0],0.); self.assertFalse(base(o).retired[0]); self.assertTrue(any(r.get('action')=='DOMINANT_FUNDED_PEAK_DISPLACEMENT_BLOCK' for r in o.trace))
    def test_weaker_peak_incumbent_does_not_block_fresh_challenger(self):
        o,units=self.prepared(); force(o,100,[.4,3.0,.2,1.2,.1,.1],[True,True,False,True,False,False]); base(o).trend.entry[99,1]=False; d=o.decide(observe(o,100,units)); self.assertEqual(d.unit_targets[0],0.); self.assertTrue(base(o).retired[0]); self.assertTrue(any(r.get('action')=='WEAKER_PEAK_RELEASE' for r in o.trace))
    def test_nonpeak_incumbent_keeps_parent_displacement(self):
        o,units=self.prepared(); o.campaign_peak_close[0]=base(o).price_signals.price[100,0]*1.1; force(o,100,[1.0,3.0,.2,.7,.1,.1],[True,True,False,True,False,False]); base(o).trend.entry[99,1]=False; d=o.decide(observe(o,100,units)); self.assertEqual(d.unit_targets[0],0.); self.assertTrue(base(o).retired[0])
    def test_acute_exit_is_never_peak_blocked(self):
        o,units=self.prepared(); force(o,100,[1.0,3.0,.2,.7,.1,.1],[True,True,False,True,False,False],ret1=[-.09,0,0,0,0,0]); d=o.decide(observe(o,100,units)); self.assertEqual(d.unit_targets[0],0.); self.assertFalse(any(r.get('action')=='DOMINANT_FUNDED_PEAK_DISPLACEMENT_BLOCK' for r in o.trace))

if __name__=='__main__': unittest.main()

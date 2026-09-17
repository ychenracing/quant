from dataclasses import replace
import importlib
import importlib.util
import unittest
import numpy as np
import pandas as pd
from test_core import sample_market
from techquant.engine import run
from techquant.policy import CloseObservation
from research.offensive_dominant_peak_authority import Owner as DominantOwner, Parameters as DominantParameters

def base(o): return o.parent.parent.base

def obs(o,s,u,c=0.):
    u=np.asarray(u,float); m=base(o).price_signals.price[s]; v=u*m; nav=float(c+v.sum()); return CloseObservation.from_inventory(s,str(o.market.calendar[s].date()),nav,c,u,np.divide(v,nav,out=np.zeros_like(v),where=nav>0))
def force(o,s,scores,entries,ret1=None):
    b=base(o); f=b.features.score.copy(); f[s]=np.asarray(scores,float); b.features=replace(b.features,score=f); p=b.price_signals; rd=p.ready.copy(); rd[s]=True; ema=p.ema20.copy(); ema[s]=np.where(np.isfinite(p.price[s]),p.price[s]*.9,0); mom=p.momentum5.copy(); mom[s]=.1; r=p.ret1.copy(); r[s]=0 if ret1 is None else np.asarray(ret1,float); b.price_signals=replace(p,ready=rd,ema20=ema,momentum5=mom,ret1=r); t=b.trend; en=t.entry.copy(); en[s]=np.asarray(entries,bool); ex=t.exit.copy(); ex[s]=False; mk=t.market.copy(); mk[s]=True; b.trend=replace(t,entry=en,exit=ex,market=mk)
class PortfolioLeaderPeakAuthorityTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research.offensive_portfolio_leader_peak_authority'),'implementation absent'); return importlib.import_module('research.offensive_portfolio_leader_peak_authority')
    def prepared(self):
        m=self.module(); o=m.Owner(sample_market(6,145),m.Parameters(True)); u=np.zeros(6); u[0]=1000.; u[3]=1000.; b=base(o); b.was_held[[0,3]]=True; b.owned_alpha_reference[0]=.9; b.owned_alpha_reference[3]=.5; b.last_session=99; o.parent.actual_was_held[[0,3]]=True; close=b.price_signals.price[100]; o.parent.campaign_peak_close[[0,3]]=close[[0,3]]; return o,u
    def test_disabled_is_exact_dominant_peak(self):
        m=self.module(); market=sample_market(6,145); a=run(market,policy_factory=lambda x,c:DominantOwner(x,DominantParameters(True))); b=run(market,policy_factory=lambda x,c:m.Owner(x,m.Parameters(False))); pd.testing.assert_frame_equal(a.equity,b.equity); pd.testing.assert_frame_equal(a.targets,b.targets); self.assertEqual(a.orders,b.orders)
    def test_new_portfolio_leader_can_release_weaker_peak(self):
        o,u=self.prepared(); force(o,100,[.4,3.,.2,1.2,.1,.1],[True,True,False,True,False,False]); base(o).trend.entry[99,1]=False; d=o.decide(obs(o,100,u)); self.assertEqual(d.unit_targets[0],0.); self.assertTrue(base(o).retired[0]); self.assertTrue(any(r.get('action')=='PORTFOLIO_LEADER_WEAKER_PEAK_RELEASE' for r in o.trace))
    def test_nonleader_challenger_cannot_release_weaker_peak(self):
        o,u=self.prepared(); force(o,100,[.4,1.0,.2,1.2,.1,.1],[True,True,False,True,False,False]); base(o).trend.entry[99,1]=False; d=o.decide(obs(o,100,u)); self.assertGreater(d.unit_targets[0],0.); self.assertFalse(base(o).retired[0]); self.assertTrue(any(r.get('action')=='NONLEADER_CHALLENGER_PEAK_BLOCK' for r in o.trace))
    def test_dominant_incumbent_still_uses_parent_peak_block(self):
        o,u=self.prepared(); base(o).owned_alpha_reference[0]=2.; force(o,100,[1.,3.,.2,.7,.1,.1],[True,True,False,True,False,False]); base(o).trend.entry[99,1]=False; d=o.decide(obs(o,100,u)); self.assertGreater(d.unit_targets[0],0.); self.assertTrue(any(r.get('action')=='DOMINANT_FUNDED_PEAK_DISPLACEMENT_BLOCK' for r in o.trace))
    def test_acute_exit_remains_authoritative(self):
        o,u=self.prepared(); force(o,100,[.4,1.,.2,1.2,.1,.1],[True,True,False,True,False,False],ret1=[-.09,0,0,0,0,0]); d=o.decide(obs(o,100,u)); self.assertEqual(d.unit_targets[0],0.)
if __name__=='__main__': unittest.main()

from dataclasses import replace
import importlib, importlib.util, unittest
import numpy as np
import pandas as pd
from test_core import sample_market
from techquant.engine import run
from techquant.policy import CloseObservation
from research.trend_book import Owner as ParentOwner, Parameters as ParentParameters

def obs(owner,session,units,cash):
    units=np.asarray(units,dtype=float); marks=owner.base.price_signals.price[session]; values=units*marks; nav=float(cash+values.sum())
    return CloseObservation.from_inventory(session,str(owner.market.calendar[session].date()),nav,cash,units,values/nav)

def force(owner,session,scores,entries):
    b=owner.base; f=b.features.score.copy(); f[session]=np.asarray(scores,float); b.features=replace(b.features,score=f)
    p=b.price_signals; ready=p.ready.copy(); ready[session]=True; price=p.price.copy(); ema20=p.ema20.copy(); mom=p.momentum5.copy(); ret1=p.ret1.copy(); ema20[session]=np.where(np.isfinite(price[session]),price[session]*.9,0); mom[session]=.1; ret1[session]=0
    b.price_signals=replace(p,ready=ready,ema20=ema20,momentum5=mom,ret1=ret1); t=b.trend; entry=t.entry.copy(); exit_=t.exit.copy(); market=t.market.copy(); entry[session]=np.asarray(entries,bool); exit_[session]=False; market[session]=True; b.trend=replace(t,entry=entry,exit=exit_,market=market)

class ChallengerFillContinuityTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research.offensive_challenger_fill_continuity'),'challenger continuity implementation is absent'); return importlib.import_module('research.offensive_challenger_fill_continuity')
    def test_control_is_exact_parent(self):
        m=self.module(); market=sample_market(6,145); parent=run(market,policy_factory=lambda current,cfg:ParentOwner(current,ParentParameters(2))); control=run(market,policy_factory=lambda current,cfg:m.Owner(current,m.Parameters(False))); pd.testing.assert_frame_equal(parent.equity,control.equity); pd.testing.assert_frame_equal(parent.targets,control.targets); self.assertEqual(parent.orders,control.orders)
    def prepared(self):
        m=self.module(); owner=m.Owner(sample_market(6,145),m.Parameters(True)); units=np.zeros(6); force(owner,100,[1.,3.,2.,.4,.3,.2],[True,True,True,False,False,False]); marks=owner.base.price_signals.price[100]; units[0]=1000.; holdings=float(units[0]*marks[0]); cash=holdings*.005/.995; owner.base.was_held[0]=True; owner.base.owned_alpha_reference[0]=2.; owner.base.last_session=99; return owner,units,cash
    def test_same_failed_challenger_still_best_can_displace(self):
        owner,units,cash=self.prepared(); first=owner.decide(obs(owner,100,units,cash)); self.assertGreater(first.unit_targets[1],0.); self.assertEqual(owner.failed_fill_attempt_challenger,1); force(owner,101,[1.,3.2,2.5,.4,.3,.2],[True,True,True,False,False,False]); d=owner.decide(obs(owner,101,units,cash)); self.assertEqual(d.unit_targets[0],0.); self.assertEqual(d.unit_targets[1],0.); self.assertIn('CHALLENGER_FILL_CONTINUITY',d.reason); ev=[r for r in owner.trace if r.get('action')=='IDENTITY_CONTINUOUS_DISPLACEMENT']; self.assertEqual(len(ev),1); self.assertEqual(ev[0]['challenger'],owner.market.symbols[1])
    def test_changed_best_challenger_cannot_use_previous_failed_fill(self):
        owner,units,cash=self.prepared(); owner.decide(obs(owner,100,units,cash)); force(owner,101,[1.,2.5,4.,.4,.3,.2],[True,True,True,False,False,False]); d=owner.decide(obs(owner,101,units,cash)); self.assertGreater(d.unit_targets[0],0.); self.assertFalse(owner.base.retired[0]); self.assertFalse(any(r.get('action')=='IDENTITY_CONTINUOUS_DISPLACEMENT' for r in owner.trace)); self.assertTrue(any(r.get('action')=='CHALLENGER_IDENTITY_CHANGED' for r in owner.trace))
    def test_successful_prior_fill_cancels_continuity_confirmation(self):
        owner,units,cash=self.prepared(); owner.decide(obs(owner,100,units,cash)); units2=units.copy(); units2[1]=10.; force(owner,101,[1.,3.2,2.5,.4,.3,.2],[True,True,True,False,False,False]); owner.decide(obs(owner,101,units2,0.)); self.assertFalse(any(r.get('action')=='IDENTITY_CONTINUOUS_DISPLACEMENT' for r in owner.trace))

if __name__=='__main__': unittest.main()

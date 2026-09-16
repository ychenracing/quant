from dataclasses import replace
import importlib, importlib.util, unittest
import numpy as np
import pandas as pd
from test_core import sample_market
from techquant.engine import run
from techquant.policy import CloseObservation
from research.trend_book import Owner as ParentOwner, Parameters as ParentParameters

def obs(owner,session,units,cash):
    units=np.asarray(units,float); marks=owner.base.price_signals.price[session]; vals=units*marks; nav=float(cash+vals.sum()); return CloseObservation.from_inventory(session,str(owner.market.calendar[session].date()),nav,cash,units,vals/nav)
def force(owner,session,scores,entries):
    b=owner.base; f=b.features.score.copy(); f[session]=np.asarray(scores,float); b.features=replace(b.features,score=f); p=b.price_signals; ready=p.ready.copy(); ready[session]=True; price=p.price.copy(); ema=p.ema20.copy(); mom=p.momentum5.copy(); ret=p.ret1.copy(); ema[session]=np.where(np.isfinite(price[session]),price[session]*.9,0); mom[session]=.1; ret[session]=0; b.price_signals=replace(p,ready=ready,ema20=ema,momentum5=mom,ret1=ret); t=b.trend; en=t.entry.copy(); ex=t.exit.copy(); mk=t.market.copy(); en[session]=np.asarray(entries,bool); ex[session]=False; mk[session]=True; b.trend=replace(t,entry=en,exit=ex,market=mk)
class FreshChallengerAuthorityTests(unittest.TestCase):
    def module(self): self.assertIsNotNone(importlib.util.find_spec('research.offensive_fresh_challenger_authority'),'fresh challenger authority implementation is absent'); return importlib.import_module('research.offensive_fresh_challenger_authority')
    def prepared(self):
        m=self.module(); o=m.Owner(sample_market(6,145),m.Parameters(True)); units=np.zeros(6); force(o,100,[1.,3.,2.,.4,.3,.2],[True,True,True,False,False,False]); marks=o.base.price_signals.price[100]; units[0]=1000.; h=float(units[0]*marks[0]); cash=h*.005/.995; o.base.was_held[0]=True; o.base.owned_alpha_reference[0]=2.; o.base.last_session=99; return o,units,cash
    def test_control_is_exact_parent(self):
        m=self.module(); market=sample_market(6,145); p=run(market,policy_factory=lambda x,c:ParentOwner(x,ParentParameters(2))); q=run(market,policy_factory=lambda x,c:m.Owner(x,m.Parameters(False))); pd.testing.assert_frame_equal(p.equity,q.equity); pd.testing.assert_frame_equal(p.targets,q.targets); self.assertEqual(p.orders,q.orders)
    def test_reference_rearm_marks_campaign_but_does_not_remove_ordinary_eligibility(self):
        o,units,cash=self.prepared(); o.invalidated[1]=True; o.invalidated_since[1]=90; o.failed_reference[1]=1.; o.base.was_held[1]=False; d=o.decide(obs(o,100,units,cash)); self.assertTrue(o.reference_rearmed_without_fresh_epoch[1]); self.assertGreater(d.unit_targets[1],0.)
    def test_marked_same_challenger_cannot_force_incumbent_sale(self):
        o,units,cash=self.prepared(); o.reference_rearmed_without_fresh_epoch[1]=True; o.failed_fill_attempt_session=99; o.failed_fill_attempt_held_count=1; o.failed_fill_attempt_challenger=1; d=o.decide(obs(o,100,units,cash)); self.assertGreater(d.unit_targets[0],0.); self.assertFalse(o.base.retired[0]); self.assertTrue(any(r.get('action')=='RECOVERED_CHALLENGER_AUTHORITY_BLOCK' for r in o.trace)); self.assertFalse(any(r.get('action')=='FRESH_AUTHORITY_DISPLACEMENT' for r in o.trace))
    def test_actual_inventory_clears_recovered_campaign_provenance(self):
        o,units,cash=self.prepared(); o.reference_rearmed_without_fresh_epoch[1]=True; units[1]=10.; force(o,100,[1.,3.,2.,.4,.3,.2],[True,True,True,False,False,False]); o.decide(obs(o,100,units,0.)); self.assertFalse(o.reference_rearmed_without_fresh_epoch[1])
    def test_fresh_false_then_true_epoch_clears_recovered_campaign_provenance(self):
        o,units,cash=self.prepared(); o.reference_rearmed_without_fresh_epoch[1]=True; force(o,100,[1.,3.,2.,.4,.3,.2],[True,False,True,False,False,False]); o.decide(obs(o,100,units,cash)); self.assertTrue(o.recovered_saw_nonentry[1]); self.assertTrue(o.reference_rearmed_without_fresh_epoch[1]); force(o,101,[1.,3.,2.,.4,.3,.2],[True,True,True,False,False,False]); o.decide(obs(o,101,units,cash)); self.assertFalse(o.reference_rearmed_without_fresh_epoch[1])
if __name__=='__main__': unittest.main()

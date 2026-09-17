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


def observe(owner, session, units, cash):
    units=np.asarray(units,float); marks=base(owner).price_signals.price[session]; values=units*marks; nav=float(cash+values.sum())
    return CloseObservation.from_inventory(session,str(owner.market.calendar[session].date()),nav,cash,units,np.divide(values,nav,out=np.zeros_like(values),where=nav>0))


def force(owner,session,scores,entries,*,exits=None,ret1=None):
    b=base(owner); f=b.features.score.copy(); f[session]=np.asarray(scores,float); b.features=replace(b.features,score=f)
    p=b.price_signals; ready=p.ready.copy(); ready[session]=True; ema=p.ema20.copy(); ema[session]=np.where(np.isfinite(p.price[session]),p.price[session]*.9,0); mom=p.momentum5.copy(); mom[session]=.1; r=p.ret1.copy(); r[session]=0 if ret1 is None else np.asarray(ret1,float); b.price_signals=replace(p,ready=ready,ema20=ema,momentum5=mom,ret1=r)
    t=b.trend; en=t.entry.copy(); en[session]=np.asarray(entries,bool); ex=t.exit.copy(); ex[session]=False if exits is None else np.asarray(exits,bool); mk=t.market.copy(); mk[session]=True; b.trend=replace(t,entry=en,exit=ex,market=mk)


class SingleDominantCampaignTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research.offensive_single_dominant_campaign'),'single dominant campaign implementation is absent')
        return importlib.import_module('research.offensive_single_dominant_campaign')

    def test_disabled_is_exact_campaign_peak_champion(self):
        m=self.module(); market=sample_market(6,145); champion=run(market,policy_factory=lambda current,cfg:ChampionOwner(current,ChampionParameters(True))); control=run(market,policy_factory=lambda current,cfg:m.Owner(current,m.Parameters(False))); pd.testing.assert_frame_equal(champion.equity,control.equity); pd.testing.assert_frame_equal(champion.targets,control.targets); self.assertEqual(champion.orders,control.orders)

    def test_enabled_changes_native_offensive_capacity_to_one(self):
        m=self.module(); owner=m.Owner(sample_market(6,145),m.Parameters(True)); self.assertEqual(base(owner).config.max_positions,1)

    def test_flat_book_funds_only_highest_score_campaign(self):
        m=self.module(); owner=m.Owner(sample_market(6,145),m.Parameters(True)); units=np.zeros(6); force(owner,100,[3.,2.,1.,.5,.4,.3],[True,True,True,False,False,False]); base(owner).last_session=99; decision=owner.decide(observe(owner,100,units,2_000_000.)); positive=np.flatnonzero(decision.unit_targets>1e-10); self.assertEqual(positive.tolist(),[0]); self.assertGreater(decision.unit_targets[0],0.)

    def test_healthy_funded_campaign_prevents_second_slot_even_with_cash(self):
        m=self.module(); owner=m.Owner(sample_market(6,145),m.Parameters(True)); units=np.zeros(6); units[0]=1000.; b=base(owner); b.was_held[0]=True; b.owned_alpha_reference[0]=1.; b.last_session=99; owner.parent.actual_was_held[0]=True; close=b.price_signals.price[100,0]; owner.parent.campaign_peak_close[0]=close*1.1; force(owner,100,[1.2,3.,2.,.5,.4,.3],[True,True,True,False,False,False]); decision=owner.decide(observe(owner,100,units,1_000_000.)); self.assertGreater(decision.unit_targets[0],0.); self.assertEqual(decision.unit_targets[1],0.); self.assertEqual(np.count_nonzero(decision.unit_targets>1e-10),1)

    def test_decayed_campaign_can_be_displaced_by_stronger_challenger(self):
        m=self.module(); owner=m.Owner(sample_market(6,145),m.Parameters(True)); units=np.zeros(6); units[0]=1000.; b=base(owner); b.was_held[0]=True; b.owned_alpha_reference[0]=1.; b.last_session=99; owner.parent.actual_was_held[0]=True; close=b.price_signals.price[100,0]; owner.parent.campaign_peak_close[0]=close*1.1; force(owner,100,[.5,3.,2.,.4,.3,.2],[True,True,True,False,False,False]); decision=owner.decide(observe(owner,100,units,0.)); self.assertEqual(decision.unit_targets[0],0.); self.assertTrue(b.retired[0]); self.assertTrue(any(r.get('action')=='ALPHA_DECAY_DISPLACEMENT' for r in owner.trace))


if __name__=='__main__': unittest.main()

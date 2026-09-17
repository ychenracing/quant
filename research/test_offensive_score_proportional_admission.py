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


def force(owner,session,scores,entries):
    b=base(owner); f=b.features.score.copy(); f[session]=np.asarray(scores,float); b.features=replace(b.features,score=f)
    p=b.price_signals; ready=p.ready.copy(); ready[session]=True; ema=p.ema20.copy(); ema[session]=np.where(np.isfinite(p.price[session]),p.price[session]*.9,0); mom=p.momentum5.copy(); mom[session]=.1; ret=p.ret1.copy(); ret[session]=0; b.price_signals=replace(p,ready=ready,ema20=ema,momentum5=mom,ret1=ret)
    t=b.trend; en=t.entry.copy(); en[session]=np.asarray(entries,bool); ex=t.exit.copy(); ex[session]=False; mk=t.market.copy(); mk[session]=True; b.trend=replace(t,entry=en,exit=ex,market=mk)


class ScoreProportionalAdmissionTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research.offensive_score_proportional_admission'),'score proportional admission implementation is absent')
        return importlib.import_module('research.offensive_score_proportional_admission')

    def test_disabled_is_exact_campaign_peak_champion(self):
        m=self.module(); market=sample_market(6,145); champion=run(market,policy_factory=lambda current,cfg:ChampionOwner(current,ChampionParameters(True))); control=run(market,policy_factory=lambda current,cfg:m.Owner(current,m.Parameters(False))); pd.testing.assert_frame_equal(champion.equity,control.equity); pd.testing.assert_frame_equal(champion.targets,control.targets); self.assertEqual(champion.orders,control.orders)

    def test_two_simultaneous_vacancies_split_cash_in_score_proportion(self):
        m=self.module(); owner=m.Owner(sample_market(6,145),m.Parameters(True)); units=np.zeros(6); force(owner,100,[3.,1.,.2,.1,.1,.1],[True,True,False,False,False,False]); base(owner).last_session=99; decision=owner.decide(observe(owner,100,units,2_000_000.)); marks=base(owner).price_signals.price[100]; values=decision.unit_targets*marks; self.assertAlmostEqual(values[0]/values[1],3.,places=9); self.assertAlmostEqual(values[0]+values[1],2_000_000.,places=5); self.assertEqual(np.flatnonzero(decision.unit_targets>1e-10).tolist(),[0,1]); self.assertTrue(any(r.get('action')=='SCORE_PROPORTIONAL_ADMISSION' for r in owner.trace))

    def test_single_vacancy_is_economically_unchanged(self):
        m=self.module(); market=sample_market(6,145); treatment=m.Owner(market,m.Parameters(True)); champion=ChampionOwner(market,ChampionParameters(True)); units=np.zeros(6); units[3]=1000.
        for owner in (treatment,champion):
            b=owner.parent.parent.base if hasattr(owner,'parent') and hasattr(owner.parent,'parent') else None
        # Configure treatment and champion separately with the same state.
        force(treatment,100,[3.,1.,.2,.5,.1,.1],[True,False,False,True,False,False]); tb=base(treatment); tb.was_held[3]=True; tb.owned_alpha_reference[3]=.5; tb.last_session=99; treatment.parent.actual_was_held[3]=True; treatment.parent.campaign_peak_close[3]=tb.price_signals.price[100,3]*1.1
        cb=champion.parent.base; f=cb.features.score.copy(); f[100]=np.array([3.,1.,.2,.5,.1,.1]); cb.features=replace(cb.features,score=f); p=cb.price_signals; ready=p.ready.copy(); ready[100]=True; ema=p.ema20.copy(); ema[100]=np.where(np.isfinite(p.price[100]),p.price[100]*.9,0); mom=p.momentum5.copy(); mom[100]=.1; ret=p.ret1.copy(); ret[100]=0; cb.price_signals=replace(p,ready=ready,ema20=ema,momentum5=mom,ret1=ret); t=cb.trend; en=t.entry.copy(); en[100]=np.array([True,False,False,True,False,False]); ex=t.exit.copy(); ex[100]=False; mk=t.market.copy(); mk[100]=True; cb.trend=replace(t,entry=en,exit=ex,market=mk); cb.was_held[3]=True; cb.owned_alpha_reference[3]=.5; cb.last_session=99; champion.actual_was_held[3]=True; champion.campaign_peak_close[3]=cb.price_signals.price[100,3]*1.1
        obs_t=observe(treatment,100,units,1_000_000.); marks=cb.price_signals.price[100]; values=units*marks; nav=float(1_000_000.+values.sum()); obs_c=CloseObservation.from_inventory(100,str(market.calendar[100].date()),nav,1_000_000.,units,values/nav)
        td=treatment.decide(obs_t); cd=champion.decide(obs_c); np.testing.assert_allclose(td.unit_targets,cd.unit_targets); np.testing.assert_allclose(td.weights,cd.weights)

    def test_existing_funded_units_are_not_rebalanced_by_admission_rule(self):
        m=self.module(); owner=m.Owner(sample_market(6,145),m.Parameters(True)); units=np.zeros(6); units[3]=1000.; b=base(owner); b.was_held[3]=True; b.owned_alpha_reference[3]=.5; b.last_session=99; owner.parent.actual_was_held[3]=True; owner.parent.campaign_peak_close[3]=b.price_signals.price[100,3]*1.1; force(owner,100,[3.,2.,1.,.8,.1,.1],[True,True,True,True,False,False]); decision=owner.decide(observe(owner,100,units,1_000_000.)); self.assertAlmostEqual(decision.unit_targets[3],1000.); self.assertEqual(np.count_nonzero(decision.unit_targets>1e-10),2)


if __name__=='__main__': unittest.main()

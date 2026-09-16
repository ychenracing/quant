"""Contracts for settlement-aware campaign reference rearm."""
from dataclasses import asdict, replace
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


def force(owner, session, *, entries, acute=None, scores=None, market=True):
    b=owner.base
    if scores is not None:
        f=b.features.score.copy(); f[session]=np.asarray(scores,float); b.features=replace(b.features,score=f)
    p=b.price_signals
    ready=p.ready.copy(); ready[session]=True
    price=p.price.copy(); ema20=p.ema20.copy(); mom=p.momentum5.copy(); ret1=p.ret1.copy()
    ema20[session]=np.where(np.isfinite(price[session]),price[session]*.9,0); mom[session]=.1; ret1[session]=0
    if acute is not None:
        for j in acute: ret1[session,j]=-.1
    b.price_signals=replace(p,ready=ready,ema20=ema20,momentum5=mom,ret1=ret1)
    t=b.trend; entry=t.entry.copy(); exit_=t.exit.copy(); ms=t.market.copy()
    entry[session]=np.asarray(entries,bool); exit_[session]=False; ms[session]=market
    b.trend=replace(t,entry=entry,exit=exit_,market=ms)


class SettledReferenceRearmTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research.offensive_settled_reference_rearm'),
                             'settled reference rearm implementation is absent')
        return importlib.import_module('research.offensive_settled_reference_rearm')

    def test_control_is_exact_parent(self):
        m=self.module(); self.assertEqual([asdict(p) for p in m.grid()],[{'enabled':False},{'enabled':True}])
        market=sample_market(6,145)
        parent=run(market,policy_factory=lambda current,cfg:ParentOwner(current,ParentParameters(2)))
        control=run(market,policy_factory=lambda current,cfg:m.Owner(current,m.Parameters(False)))
        pd.testing.assert_frame_equal(parent.equity,control.equity)
        pd.testing.assert_frame_equal(parent.targets,control.targets)
        self.assertEqual(parent.orders,control.orders)

    def test_exit_settlement_close_cannot_rearm_same_campaign(self):
        m=self.module(); owner=m.Owner(sample_market(6,145),m.Parameters(True)); n=6
        entries=np.zeros(n,bool); entries[0]=True
        force(owner,100,entries=entries,scores=np.array([5.,1.,1.,1.,1.,1.]))
        first=owner.decide(obs(owner,100,np.zeros(n),2_000_000.0)); units=first.unit_targets.copy()
        owner.decide(obs(owner,101,units,0.0))
        self.assertAlmostEqual(owner.base.owned_alpha_reference[0],5.)
        force(owner,102,entries=entries,acute=[0],scores=np.array([4.,1.,1.,1.,1.,1.]))
        exited=owner.decide(obs(owner,102,units,0.0)); self.assertEqual(exited.unit_targets[0],0.)
        force(owner,103,entries=entries,scores=np.array([6.,1.,1.,1.,1.,1.]))
        settlement=owner.decide(obs(owner,103,np.zeros(n),2_000_000.0))
        self.assertEqual(settlement.unit_targets[0],0.)
        self.assertTrue(owner.invalidated[0])
        self.assertTrue(any(r.get('action')=='SETTLEMENT_REARM_BLOCKED' for r in owner.trace))
        force(owner,104,entries=entries,scores=np.array([6.,1.,1.,1.,1.,1.]))
        rearmed=owner.decide(obs(owner,104,np.zeros(n),2_000_000.0))
        self.assertGreater(rearmed.unit_targets[0],0.)
        self.assertFalse(owner.invalidated[0])

    def test_weaker_score_remains_blocked_after_settlement(self):
        m=self.module(); owner=m.Owner(sample_market(6,145),m.Parameters(True)); n=6
        entries=np.zeros(n,bool); entries[0]=True
        force(owner,100,entries=entries,scores=np.array([5.,1.,1.,1.,1.,1.]))
        first=owner.decide(obs(owner,100,np.zeros(n),2_000_000.0)); units=first.unit_targets.copy(); owner.decide(obs(owner,101,units,0.0))
        force(owner,102,entries=entries,acute=[0],scores=np.array([4.,1.,1.,1.,1.,1.])); owner.decide(obs(owner,102,units,0.0))
        force(owner,103,entries=entries,scores=np.array([6.,1.,1.,1.,1.,1.])); owner.decide(obs(owner,103,np.zeros(n),2_000_000.0))
        force(owner,104,entries=entries,scores=np.array([4.9,1.,1.,1.,1.,1.]))
        weak=owner.decide(obs(owner,104,np.zeros(n),2_000_000.0))
        self.assertEqual(weak.unit_targets[0],0.)
        self.assertTrue(owner.invalidated[0])

    def test_fresh_symbol_is_never_subject_to_settlement_rule(self):
        m=self.module(); owner=m.Owner(sample_market(6,145),m.Parameters(True)); n=6
        entries=np.zeros(n,bool); entries[1]=True; scores=np.arange(n,dtype=float)+1
        force(owner,100,entries=entries,acute=[1],scores=scores)
        acute_close=owner.decide(obs(owner,100,np.zeros(n),2_000_000.0))
        self.assertEqual(acute_close.unit_targets[1],0.)
        self.assertFalse(owner.invalidated[1])
        force(owner,101,entries=entries,scores=scores)
        next_close=owner.decide(obs(owner,101,np.zeros(n),2_000_000.0))
        self.assertGreater(next_close.unit_targets[1],0.)
        self.assertFalse(owner.invalidated[1])


if __name__=='__main__': unittest.main()

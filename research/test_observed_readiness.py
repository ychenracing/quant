"""Expand history causally without changing the selected parent's other rules."""
from dataclasses import asdict
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from test_core import sample_market
from techquant.data import Market
from techquant.engine import run
from research.quantity_obligation import Owner as Parent, Parameters as ParentParameters
from test_quantity_obligation import observe, control


class ObservedReadinessTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research.observed_readiness'),
                             'the registered expanding-observation owner is absent')
        return importlib.import_module('research.observed_readiness')

    def test_one_fixed_candidate_and_source_bound_parent(self):
        m = self.module()
        self.assertEqual([asdict(p) for p in m.grid()], [{}])
        with self.assertRaises(TypeError): m.Parameters(minimum=5)
        from research.finite_study import Study
        dependencies = Study('observed_readiness').identity()['dependencies']
        for name in ('observed_readiness_contract.json','quantity_obligation.py',
                     'quantity_obligation_contract.json','support_budget.py','support_budget_contract.json'):
            self.assertIn(name, dependencies)

    def test_actual_history_minimum_and_no_invented_first_true_range(self):
        m = self.module(); market = sample_market(2,100)
        owner = m.Owner(market,m.Parameters()).inner
        minimum = owner.config.fast
        self.assertFalse(owner.ready[:minimum].any())
        self.assertFalse(np.isfinite(owner.atr[:minimum]).any())
        self.assertTrue(owner.ready[minimum:20].all())
        self.assertTrue(np.isfinite(owner.support[minimum:20]).all())
        old = Parent(market,ParentParameters('support')).inner
        self.assertFalse(old.ready[:20].any())

    def test_twenty_observation_estimates_are_identical_to_parent(self):
        m = self.module(); market = sample_market(3,130)
        owner = m.Owner(market,m.Parameters()).inner
        old = Parent(market,ParentParameters('support')).inner
        for field in ('atr','support','ready'):
            np.testing.assert_array_equal(getattr(owner,field)[20:],getattr(old,field)[20:])
        np.testing.assert_array_equal(owner.features.entry,old.features.entry)
        np.testing.assert_array_equal(owner.features.exit,old.features.exit)
        np.testing.assert_array_equal(owner.features.score,old.features.score)
        self.assertEqual(owner.config,old.config); self.assertEqual(owner.params,old.params)

    def test_new_listing_counts_quotes_not_global_calendar_or_zero_volume(self):
        m = self.module(); market = sample_market(2,100)
        frames = {s:f.copy() for s,f in market.frames.items()}
        symbol = market.symbols[0];frame = frames[symbol].iloc[25:].copy()
        frames[symbol] = frame
        frame.loc[market.calendar[27:30],'volume'] = 0.
        frame.loc[market.calendar[40:43],'volume'] = 0.
        market = Market.from_frames(frames,market.calendar,quality='synthetic')
        owner = m.Owner(market,m.Parameters()).inner
        active = (market.panel('close').notna() & market.panel('volume').gt(0)).cumsum().iloc[:,0].to_numpy()
        self.assertFalse(owner.ready[active<owner.config.fast,0].any())
        self.assertFalse(owner.ready[40:43,0].any())
        self.assertTrue(owner.ready[38,0])

    def test_early_admission_uses_actual_cash_and_same_support_loss_budget(self):
        m = self.module(); adapter = m.Owner(sample_market(3,100),m.Parameters());owner=adapter.inner
        control(owner,1.)
        observation = observe(owner,15,[0.,0.,0.],cash=2_000_000.)
        decision=adapter.decide(observation);price=owner.features.close[15]
        self.assertGreater(decision.unit_targets.sum(),0.)
        self.assertLessEqual(np.count_nonzero(decision.unit_targets),2)
        self.assertLessEqual(float(decision.unit_targets@price),.99*observation.cash+1e-7)
        distance=np.maximum(price-owner.admission_stop(15),.02*price)
        self.assertLessEqual(float(decision.unit_targets@distance),.10*observation.nav+1e-7)

    def test_unfilled_economic_ceiling_still_blocks_purchases_and_does_not_ratchet(self):
        m=self.module();adapter=m.Owner(sample_market(2,100),m.Parameters());owner=adapter.inner
        control(owner);owner.reduction_ceiling[0]=999.95
        first=adapter.decide(observe(owner,40,[1000.05,0.]))
        second=adapter.decide(observe(owner,41,[1000.05,0.]))
        self.assertEqual(owner.reduction_ceiling[0],999.95)
        np.testing.assert_array_equal(first.unit_targets,second.unit_targets)
        self.assertEqual(second.unit_targets[1],0.)

    def test_future_prefix_removed_quotes_and_delayed_execution(self):
        m=self.module(); market=sample_market(3,130)
        def execute(data,delay=1):
            return run(data,delay=delay,policy_factory=lambda current,c:m.Owner(current,m.Parameters()))
        full=execute(market);cut=market.calendar[85];prefix=execute(market.prefix(cut))
        pd.testing.assert_frame_equal(full.equity.loc[:cut],prefix.equity)
        pd.testing.assert_frame_equal(full.targets.loc[:cut],prefix.targets)
        frames={s:f.copy() for s,f in market.frames.items()}
        frames[market.symbols[-1]].loc[:,['open','high','low','close','raw_open','raw_close']] *= 7
        changed=Market.from_frames(frames,market.calendar,quality='synthetic')
        names=market.symbols[:-1]
        a,b=execute(market.subset(names)),execute(changed.subset(names))
        self.assertEqual(a.orders,b.orders);pd.testing.assert_frame_equal(a.equity,b.equity)
        from research.ledger_attribution import attribute
        for result in (full,execute(market,2)):
            self.assertTrue((result.equity.cash>=0).all())
            self.assertLess(attribute(market,result)[2]['max_reconciliation_error'],1e-6)

    def test_study_trace_and_cache_identity_remain_strict(self):
        m=self.module()
        from research.finite_study import Study
        study=Study('observed_readiness');market=sample_market(2,100)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'runs'/'candidate'
            first=study.saved(market,path,m.Parameters());second=study.saved(market,path,m.Parameters())
            pd.testing.assert_frame_equal(first.targets.rename_axis('date'),second.targets,check_freq=False)
            trace=root/'intents/candidate.json';self.assertTrue(trace.is_file())
            saved=json.loads(trace.read_text());saved['identity']['delay']=9
            trace.write_text(json.dumps(saved))
            with self.assertRaises(ValueError):study.saved(market,path,m.Parameters())


if __name__=='__main__':unittest.main()

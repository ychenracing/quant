"""Original configuration neighborhoods must configure the entire owner."""
from dataclasses import asdict, replace
import importlib
import importlib.util
import inspect
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from test_core import sample_market
from techquant.config import Config
from techquant.data import Market
from techquant.engine import run
from techquant.features import build_features
from research.observed_admission_completion import Owner, Parameters


class NeighborhoodValidationTests(unittest.TestCase):
    def configured(self, market, config):
        self.assertIn('config', inspect.signature(Owner).parameters,
                      'configuration must enter before feature construction')
        return Owner(market, Parameters(), config=config)

    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research.neighborhood_validation'),
                             'original-neighborhood runner is missing')
        return importlib.import_module('research.neighborhood_validation')

    def test_config_reaches_all_feature_and_support_construction(self):
        market=sample_market(3,110)
        for config in (replace(Config(),fast=9),replace(Config(),fast=11),replace(Config(),slow=36)):
            owner=self.configured(market,config);expected=build_features(market,config)
            self.assertIs(owner.inner.config,config)
            for name in ('ready','score','entry','exit','breadth','market_vol','weak'):
                np.testing.assert_array_equal(getattr(owner.inner.features,name),getattr(expected,name))
            # Expanding support's first usable prior-close window uses this fast minimum.
            first=np.flatnonzero(np.isfinite(owner.inner.support[:,0]))[0]
            self.assertEqual(first,config.fast)
            self.assertEqual(owner.identity()['readiness_origin']['minimum'],config.fast)
            self.assertEqual(owner.inner.identity()['configuration'],asdict(config))

    def test_default_and_explicit_default_have_identical_executed_paths(self):
        market=sample_market(3,140);self.configured(market,Config())
        a=run(market,policy_factory=lambda m,c:Owner(m,Parameters()))
        b=run(market,policy_factory=lambda m,c:Owner(m,Parameters(),config=c))
        pd.testing.assert_frame_equal(a.equity,b.equity,check_exact=True)
        pd.testing.assert_frame_equal(a.targets,b.targets,check_exact=True)
        self.assertEqual(a.orders,b.orders)

    def test_owners_are_isolated_and_invalid_config_is_not_ignored(self):
        market=sample_market(2,100)
        a=self.configured(market,replace(Config(),fast=9));b=self.configured(market,replace(Config(),fast=11))
        self.assertEqual(a.inner.config.fast,9);self.assertEqual(b.inner.config.fast,11)
        for bad in ({'fast':9},True,9):
            with self.assertRaises((ValueError,TypeError)):
                self.configured(market,bad)
        from research.quantity_obligation import Owner as Q, Parameters as P
        with self.assertRaisesRegex(ValueError,'support'):
            Q(market,P('funded'),config=Config())

    def test_changed_configuration_preserves_prefix_and_excluded_input_isolation(self):
        market=sample_market(3,130);cfg=replace(Config(),fast=9,risk_drawdown=.162)
        self.configured(market,cfg)
        factory=lambda m,c:Owner(m,Parameters(),config=c)
        a=run(market,cfg,policy_factory=factory,delay=2);cut=market.calendar[80]
        b=run(market.prefix(cut),cfg,policy_factory=factory,delay=2)
        pd.testing.assert_frame_equal(a.equity.loc[:cut],b.equity,check_exact=True)
        self.assertEqual([o for o in a.orders if o['date']<=str(cut.date())],b.orders)
        changed={s:f.copy() for s,f in market.frames.items()}
        changed[market.symbols[-1]].loc[:,['open','high','low','close','raw_open','raw_close']]*=10
        other=Market.from_frames(changed,market.calendar,quality='synthetic')
        a=run(market.subset(market.symbols[:-1]),cfg,policy_factory=factory)
        b=run(other.subset(market.symbols[:-1]),cfg,policy_factory=factory)
        pd.testing.assert_frame_equal(a.equity,b.equity,check_exact=True);self.assertEqual(a.orders,b.orders)

    def test_original_neighborhoods_are_not_regenerated_or_reselected(self):
        module=self.module();market=sample_market(2,80)
        case={'name':'neighbor_fast_0.9','group':'stability_not_reselection',
              'symbols':list(market.symbols),'cost':1.,'delay':1,'config':asdict(replace(Config(),fast=9))}
        self.assertEqual(module.neighborhood_cases([case],market),[case])
        with self.assertRaises(ValueError):module.neighborhood_cases([case,case],market)
        bad=case|{'config':case['config']|{'fast':True}}
        with self.assertRaises(ValueError):module.neighborhood_cases([bad],market)
        bad=case|{'symbols':[market.symbols[0],market.symbols[0]]}
        with self.assertRaises(ValueError):module.neighborhood_cases([bad],market)

    def test_cache_binds_configuration_and_preserves_intent_integrity(self):
        module=self.module();market=sample_market(2,90)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'runs'/'candidate';cfg=replace(Config(),fast=9)
            a=module.saved(market,path,cfg);b=module.saved(market,path,cfg)
            self.assertEqual(a.metadata,b.metadata);self.assertEqual(a.orders,b.orders)
            with self.assertRaises(ValueError):module.saved(market,path,replace(Config(),fast=11))
            trace=Path(d)/'intents'/'candidate.json';trace.write_text('{}')
            with self.assertRaises(ValueError):module.saved(market,path,cfg)

    def test_default_frozen_comparison_must_not_skip_a_missing_account(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises((ValueError,FileNotFoundError)):
                module.default_equivalence(Path(d),sample_market(2,80),Path(d)/'report.json')


if __name__=='__main__':unittest.main()

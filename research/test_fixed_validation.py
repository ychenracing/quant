"""Fixed-candidate validation must preserve identities, floors and missing gates."""
from dataclasses import asdict
import copy
import importlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from techquant.config import Config
from techquant.engine import run
from test_core import sample_market
from test_fill_materiality import constant_market


class FixedValidationTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research.fixed_validation'),
                             'fixed-candidate validation implementation is absent')
        return importlib.import_module('research.fixed_validation')

    def case(self, symbols, name='example', **changes):
        return dict(name=name,group='original_pool',symbols=list(symbols),cost=1.,delay=1,
                    config=asdict(Config())) | changes

    def test_aliases_share_accounts_but_not_scenario_names(self):
        module=self.module();market=sample_market(2,90)
        one=self.case(market.symbols);two=self.case(reversed(market.symbols),'alias')
        cases,missing=module.resolve_cases([one,two],market)
        self.assertEqual(len(cases),2);self.assertFalse(missing)
        self.assertEqual(module.case_key(cases[0]),module.case_key(cases[1]))
        self.assertNotEqual(cases[0]['name'],cases[1]['name'])

    def test_neighborhoods_remain_explicitly_unverified(self):
        module=self.module();market=sample_market(2,90)
        neighbor=self.case(market.symbols,'neighbor',group='stability_not_reselection',
                           config=asdict(Config())|{'slow':44})
        cases,missing=module.resolve_cases([self.case(market.symbols),neighbor],market)
        self.assertEqual(len(cases),1);self.assertEqual(missing[0]['status'],'UNVERIFIED')
        self.assertIn('constructor',missing[0]['reason'])

    def test_undeclared_changes_and_bad_membership_fail_closed(self):
        module=self.module();market=sample_market(2,90)
        variants=[{'config':asdict(Config())|{'initial_cash':1.}},
                  {'symbols':[]},{'symbols':[market.symbols[0]]*2},
                  {'symbols':['sz999999']},{'cost':True},{'delay':0}]
        for change in variants:
            with self.subTest(change=change),self.assertRaises(ValueError):
                module.resolve_cases([self.case(market.symbols)|change],market)
        with self.assertRaises(ValueError):
            module.resolve_cases([self.case(market.symbols),self.case(market.symbols)],market)

    def test_executable_floor_uses_opening_nav_not_slipped_notional(self):
        module=self.module();market=constant_market(days=10)
        targets=pd.DataFrame(.01,index=market.calendar,columns=market.symbols)
        result=run(market,targets=targets,cost_multiplier=0.)
        check=module.floor_audit(market,result)
        self.assertEqual(check['subfloor_fills'],0)
        self.assertEqual(check['minimum_ordinary_fraction'],.01)
        historical=copy.deepcopy(result);fill=historical.orders[0]
        fill['units']/=2;fill['raw_quantity_equivalent']/=2;fill['notional']/=2
        start=pd.Timestamp(fill['date']);historical.equity.loc[start:,'cash']+=10_000.
        historical.equity.loc[start:,'holdings']-=10_000.
        historical.equity.loc[start:,'exposure']=historical.equity.loc[start:,'holdings']/historical.equity.loc[start:,'nav']
        check=module.floor_audit(market,historical)
        self.assertEqual(check['subfloor_fills'],1)
        self.assertEqual(check['violations'][0]['opening_notional'],10_000.)

    def test_protective_subfloor_sales_do_not_invalidate_equivalence(self):
        module=self.module();market=constant_market(volume=100_000.,days=28)
        frames={s:f.copy() for s,f in market.frames.items()};s=market.symbols[0]
        frames[s].iloc[0,frames[s].columns.get_loc('volume')]=1_000_000.
        from techquant.data import Market
        market=Market.from_frames(frames,market.calendar,quality='synthetic')
        targets=pd.DataFrame(.01,index=market.calendar,columns=market.symbols);targets.iloc[20:]=0.
        result=run(market,targets=targets,cost_multiplier=0.)
        check=module.floor_audit(market,result)
        self.assertEqual(check['subfloor_fills'],0)
        self.assertGreater(check['protective_fills'],0)
        damaged=copy.deepcopy(result);damaged.equity.iloc[-1,damaged.equity.columns.get_loc('cash')]+=1.
        with self.assertRaisesRegex(ValueError,'reconcile'):
            module.floor_audit(market,damaged)

    def test_small_real_validation_cache_rejects_tampering(self):
        module=self.module();market=sample_market(2,90)
        cases=[self.case(market.symbols),self.case(reversed(market.symbols),'alias')]
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'validation'
            report=module.validate_cases(market,cases,output)
            self.assertEqual(report['scenario_count'],2)
            self.assertEqual(report['unique_accounts'],4)
            self.assertEqual(report['new_accounts'],4)
            self.assertEqual(report['economic_acceptance'],'UNVERIFIED')
            rerun=module.validate_cases(market,cases,output)
            self.assertEqual(rerun['new_accounts'],0)
            result=next((output/'runs').glob('*/orders.csv'))
            result.write_text('changed evidence\n')
            with self.assertRaises(ValueError):module.validate_cases(market,cases,output)

    def test_frozen_plan_rejects_unpinned_archive_before_reading(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as temp:
            archive=Path(temp)/'untrusted.tar.gz';archive.write_bytes(b'not a pinned archive')
            with self.assertRaisesRegex(ValueError,'baseline archive mismatch'):
                module.frozen_plan(archive)

    def test_changed_plan_cannot_resume_same_output(self):
        module=self.module();market=sample_market(1,60)
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'validation';cases=[self.case(market.symbols)]
            module.validate_cases(market,cases,output)
            with self.assertRaisesRegex(ValueError,'identity'):
                module.validate_cases(market,[self.case(market.symbols,cost=2.)],output)


if __name__=='__main__':unittest.main()

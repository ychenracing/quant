"""First-passage labels describe ordered close paths, never assumed fills."""
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np
from test_core import sample_market
from techquant.engine import run
from research import nonlinear


class PathwiseTests(unittest.TestCase):
    def module(self):
        try:
            from research import pathwise
            return pathwise
        except ImportError as error:
            self.fail(f'declared first-passage learner is missing: {error}')

    def test_first_reached_barrier_not_terminal_return_decides_class(self):
        module = self.module()
        close = np.array([[100.,100.,100.],[89.,121.,101.],[130.,80.,99.],[140.,130.,102.]])
        opening = np.full_like(close,100.)
        labels = module.matured_outcomes(close,opening,horizon=3,cutoff=3,loss_barrier=.1)
        np.testing.assert_array_equal(labels[:, :, 0],[[-1.,1.,0.]])

    def test_early_hit_does_not_bypass_full_path_maturity(self):
        module = self.module()
        close = np.array([[100.],[120.],[125.],[126.],[10.]])
        opening = np.full_like(close,100.)
        self.assertEqual(module.matured_outcomes(close,opening,horizon=3,cutoff=2,loss_barrier=.08).shape,(0,1,1))
        expected = module.matured_outcomes(close,opening,horizon=3,cutoff=3,loss_barrier=.08)
        close[4] = 10000.
        np.testing.assert_array_equal(expected,module.matured_outcomes(close,opening,horizon=3,cutoff=3,loss_barrier=.08))

    def test_missing_after_hit_still_rejects_path(self):
        module = self.module()
        close = np.array([[100.],[121.],[np.nan],[125.]])
        labels = module.matured_outcomes(close,np.full_like(close,100.),horizon=3,cutoff=3,loss_barrier=.1)
        self.assertTrue(np.isnan(labels).all())

    def test_probability_expansion_smooths_missing_and_single_classes(self):
        module = self.module()
        p = module.expanded_probabilities(np.array([[.2,.8]]),np.array([-1,1]),40)
        np.testing.assert_allclose(p,[[9/43,1/43,33/43]])
        one = module.expanded_probabilities(np.array([[1.]]),np.array([0]),40)
        np.testing.assert_allclose(one,[[1/43,41/43,1/43]])
        self.assertAlmostEqual(float(p.sum()),1.)
        with self.assertRaises(ValueError):
            module.expanded_probabilities(np.array([[.5,.5]]),np.array([0,5]),40)

    def test_explicit_same_forecast_preserves_parent_actual_fills(self):
        market = sample_market(2,100);p = nonlinear.Parameters(horizon=20)
        f = nonlinear.forecast(market,p)
        direct = run(market,policy_factory=lambda m,c:nonlinear.Owner(m,p))
        supplied = run(market,policy_factory=lambda m,c:nonlinear.Owner(m,p,prediction=f))
        np.testing.assert_array_equal(direct.equity.nav,supplied.equity.nav)
        np.testing.assert_array_equal(direct.targets,supplied.targets)
        self.assertEqual(direct.orders,supplied.orders)

    def test_wrong_forecast_identity_rejected(self):
        market = sample_market(2,100);p = nonlinear.Parameters(horizon=20)
        f = nonlinear.forecast(market,p)
        with self.assertRaises(ValueError):
            nonlinear.Owner(market,p,prediction=replace(f,data_sha256='wrong market'))
        with self.assertRaises(ValueError):
            nonlinear.Owner(market,p,prediction=replace(f,symbols=tuple(reversed(f.symbols))))

    def test_predictions_and_mature_training_are_prefix_causal(self):
        module = self.module();market = sample_market(3,125);p = module.Parameters(horizon=20,loss_barrier=.08)
        full = module.forecast(market,p);short = module.forecast(market.prefix(market.calendar[99]),p)
        for name in ('expected','tail','ready','outcome_probability'):
            np.testing.assert_array_equal(getattr(full,name)[:100],getattr(short,name))
        self.assertTrue(full.fits)
        self.assertEqual([r for r in full.fits if r['session']<100],short.fits)
        self.assertTrue(all(r['last_label_session']<=r['session'] for r in full.fits))
        fitted=np.isfinite(full.outcome_probability).all(axis=-1)
        np.testing.assert_allclose(full.outcome_probability[fitted].sum(axis=-1),1.)
        self.assertTrue((full.outcome_probability[fitted]>0).all())

    def test_excluded_quotes_do_not_enter_features_or_training(self):
        from techquant.data import Market
        module=self.module();market=sample_market(3,100);names=market.symbols[:1]
        original=module.forecast(market.subset(names),module.Parameters(horizon=20))
        frames={s:f.copy() for s,f in market.frames.items()}
        for symbol in market.symbols[1:]:
            frames[symbol].loc[:,['open','high','low','close','raw_open','raw_close']]*=97.
        altered=Market.from_frames(frames,market.calendar,quality=market.quality)
        changed=module.forecast(altered.subset(names),module.Parameters(horizon=20))
        self.assertEqual(original.fingerprint(),changed.fingerprint())
        self.assertEqual(original.fits,changed.fits)

    def test_multiclass_learning_and_real_fill_accounting_remain_causal(self):
        import pandas as pd
        from techquant.data import Market
        from research.ledger_attribution import attribute
        module=self.module();base=sample_market(4,160);rng=np.random.default_rng(715)
        frames={}
        for symbol in base.symbols:
            changes=rng.normal(.002,.035,len(base.calendar))
            close=30.*np.exp(np.cumsum(changes))
            opening=np.r_[close[0],close[:-1]]*np.exp(rng.normal(0,.003,len(close)))
            frames[symbol]=pd.DataFrame({'open':opening,'close':close,
                'high':np.maximum(opening,close)*1.01,'low':np.minimum(opening,close)*.99,
                'volume':20_000_000.,'raw_open':opening,'raw_close':close},index=base.calendar)
        market=Market.from_frames(frames,base.calendar,quality='synthetic')
        p=module.Parameters();cut=market.calendar[119]
        owner=module.Owner(market,p)
        self.assertTrue(any('tree_sha256' in row for row in owner.f.fits))
        full=run(market,policy_factory=lambda m,c:owner)
        short=run(market.prefix(cut),policy_factory=lambda m,c:module.Owner(m,p))
        pd.testing.assert_frame_equal(full.equity.loc[:cut],short.equity)
        pd.testing.assert_frame_equal(full.targets.loc[:cut],short.targets)
        fills=[o for o in full.orders if o['status']=='FILLED']
        self.assertTrue(fills)
        self.assertTrue(all(o['signal_date']<o['date'] for o in fills))
        _,_,checks=attribute(market,full)
        self.assertLess(checks['max_reconciliation_error'],1e-6)


if __name__=='__main__':unittest.main()

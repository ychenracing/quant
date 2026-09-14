"""Optional research-model correctness; not a mandatory numerical CI matrix."""
import unittest
import numpy as np
import pandas as pd
from test_core import sample_market


class NonlinearTests(unittest.TestCase):
    def module(self):
        try:
            from research import nonlinear
            return nonlinear
        except ImportError as exc:self.fail(f'separate return and downside models missing: {exc}')

    def test_forecasts_and_fit_endpoints_are_prefix_causal(self):
        n=self.module();m=sample_market(3,120);p=n.Parameters(horizon=20)
        a=n.forecast(m,p);b=n.forecast(m.prefix(m.calendar[99]),p)
        np.testing.assert_allclose(a.expected[:100],b.expected,rtol=0,atol=1e-12)
        np.testing.assert_allclose(a.tail[:100],b.tail,rtol=0,atol=1e-12)
        self.assertEqual([f for f in a.fits if f['session']<100],b.fits)
        self.assertTrue(a.fits)
        self.assertTrue(all(f['last_label_session']<=f['session'] for f in a.fits))

    def test_excluded_name_changes_cannot_affect_forecast(self):
        from techquant.data import Market
        n=self.module();m=sample_market(3,100);p=n.Parameters(horizon=20)
        only=m.subset(m.symbols[:1]);a=n.forecast(only,p)
        frames={s:f.copy() for s,f in m.frames.items()}
        for symbol in m.symbols[1:]:
            for field in ('open','high','low','close','raw_open','raw_close'):
                frames[symbol][field]*=100.
        altered=Market.from_frames(frames,m.calendar,quality=m.quality)
        b=n.forecast(altered.subset(only.symbols),p)
        self.assertEqual(a.symbols,only.symbols)
        np.testing.assert_array_equal(a.expected,b.expected)
        np.testing.assert_array_equal(a.tail,b.tail)
        self.assertEqual(a.fits,b.fits)

    def test_single_tail_class_uses_smoothed_frequency(self):
        n=self.module();f=n.forecast(sample_market(1,100),n.Parameters())
        tail_fits=[r for r in f.fits if r['task']=='tail']
        self.assertTrue(tail_fits)
        self.assertTrue(all(0<r['smoothed_prevalence']<1 for r in tail_fits))
        self.assertTrue(((f.tail>0)&(f.tail<1)).all())

    def test_actual_cash_fills_and_prefix_remain_consistent(self):
        from techquant.data import Market
        from techquant.engine import run
        from research.ledger_attribution import attribute
        n=self.module();base=sample_market(3,180);rng=np.random.default_rng(631)
        frames={}
        for symbol in base.symbols:
            returns=rng.normal(.001,.028,len(base.calendar))
            returns[65:70]-=.045;returns[105:120]+=.015
            close=20*np.exp(np.cumsum(returns))
            opening=np.r_[close[0],close[:-1]]*np.exp(rng.normal(0,.005,len(close)))
            frames[symbol]=pd.DataFrame({'open':opening,'close':close,
                'high':np.maximum(opening,close)*1.01,'low':np.minimum(opening,close)*.99,
                'volume':20_000_000.,'raw_open':opening,'raw_close':close},index=base.calendar)
        market=Market.from_frames(frames,base.calendar,quality='synthetic')
        p=n.Parameters(horizon=20,positions=2);cut=market.calendar[139]
        full=run(market,policy_factory=lambda m,c:n.Owner(m,p))
        short=run(market.prefix(cut),policy_factory=lambda m,c:n.Owner(m,p))
        pd.testing.assert_frame_equal(full.equity.loc[:cut],short.equity)
        pd.testing.assert_frame_equal(full.targets.loc[:cut],short.targets)
        self.assertEqual([o for o in full.orders if o['date']<=str(cut.date())],short.orders)
        _,_,reconciliation=attribute(market,full)
        self.assertLess(reconciliation['max_reconciliation_error'],1e-6)
        fills=[o for o in full.orders if o['status']=='FILLED']
        self.assertTrue(fills)
        self.assertTrue(all(o['signal_date']<o['date'] for o in fills))

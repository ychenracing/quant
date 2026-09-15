"""Settlement, not a future mark or fixed return window, releases a label."""
from dataclasses import replace
import unittest
import numpy as np
from research.campaign_payoff import SettledCampaign, learn_campaign_payoff


def example():
    t = np.arange(48, dtype=float)[:, None]
    n = np.arange(3, dtype=float)[None, :]
    x = np.stack((np.sin(t/4+n), np.cos(t/7-n), (t%5+n)/5), axis=2)
    samples = [SettledCampaign(i, i+3, i%3, float(x[i,i%3] @ [0.3,-0.2,0.6]))
               for i in range(24)]
    return x, samples


class CampaignPayoffTests(unittest.TestCase):
    def test_no_label_before_actual_settlement(self):
        x, rows = example()
        issued, fits = learn_campaign_payoff(x, rows, refit=2)
        self.assertTrue(np.isnan(issued[:8]).all())
        self.assertTrue(np.isfinite(issued[8]).all())
        self.assertTrue(all(f['latest_settlement_session'] <= f['session'] for f in fits))
        self.assertEqual(fits[4]['samples'], 6)

    def test_future_payoff_and_future_context_cannot_change_issued_history(self):
        x, rows = example(); split = 14
        changed = [replace(r,payoff=r.payoff+999) if r.settled>split else r for r in rows]
        altered = x.copy(); altered[split+1:] *= -100
        a, fa = learn_campaign_payoff(x, rows, refit=2)
        b, fb = learn_campaign_payoff(altered, changed, refit=2)
        c, fc = learn_campaign_payoff(x[:split+1], [r for r in rows if r.settled<=split], refit=2)
        np.testing.assert_array_equal(a[:split+1], b[:split+1])
        np.testing.assert_array_equal(a[:split+1], c)
        self.assertEqual([f for f in fa if f['session']<=split],fc)
        self.assertEqual([f for f in fa if f['session']<=split],
                         [f for f in fb if f['session']<=split])

    def test_actual_payoff_magnitude_matters_not_only_win_sign(self):
        x, rows = example()
        small = [replace(r,payoff=0.1+0.01*i) for i,r in enumerate(rows)]
        large = [replace(r,payoff=r.payoff*10) for r in small]
        a, _ = learn_campaign_payoff(x, small, refit=2)
        b, _ = learn_campaign_payoff(x, large, refit=2)
        np.testing.assert_allclose(b[8:], a[8:]*10, rtol=1e-12, atol=1e-12)

    def test_ridge_is_on_mean_cross_product_with_training_only_scaling(self):
        x, rows = example(); day=12
        issued, _ = learn_campaign_payoff(x,rows,refit=2)
        available=[r for r in rows if r.settled<=day]
        xx=np.array([x[r.formation,r.symbol] for r in available]); yy=np.array([r.payoff for r in available])
        center=xx.mean(axis=0); scale=xx.std(axis=0)
        xx=(xx-center)/scale
        beta=np.linalg.solve(xx.T@xx/len(yy)+np.eye(3),xx.T@(yy-yy.mean())/len(yy))
        np.testing.assert_allclose(issued[day],((x[day]-center)/scale)@beta+yy.mean(),rtol=0,atol=1e-14)

    def test_sparse_single_symbol_constant_and_missing_feature_fallbacks(self):
        x, rows = example()
        for records in ([],rows[:5],[replace(r,symbol=0) for r in rows],
                        [replace(r,payoff=1.0) for r in rows]):
            p,_=learn_campaign_payoff(x,records,refit=2)
            self.assertTrue(np.isnan(p).all())
        changed=x.copy(); changed[30,1]=np.nan
        p,_=learn_campaign_payoff(changed,rows,refit=2)
        self.assertTrue(np.isnan(p[30,1]).all())
        self.assertTrue(np.isfinite(p[30,[0,2]]).all())

    def test_canonical_record_order_and_input_immutability(self):
        x,rows=example(); before=x.copy()
        a,fa=learn_campaign_payoff(x,rows,refit=2)
        b,fb=learn_campaign_payoff(x,list(reversed(rows)),refit=2)
        np.testing.assert_array_equal(a,b);self.assertEqual(fa,fb)
        np.testing.assert_array_equal(before,x)

    def test_invalid_records_and_duplicate_episodes_are_rejected(self):
        x, rows=example()
        for kwargs in ({'formation':True},{'settled':1},{'settled':-1},
                       {'symbol':3},{'symbol':-1},{'payoff':float('inf')},{'payoff':True}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                learn_campaign_payoff(x,[replace(rows[0],**kwargs)],refit=2)
        with self.assertRaises(ValueError):learn_campaign_payoff(x,rows+[rows[0]],refit=2)
        for refit in (0,True,2.0):
            with self.assertRaises(ValueError):learn_campaign_payoff(x,rows,refit=refit)
        bad=x.copy();bad[3,1,1]=np.inf
        with self.assertRaises(ValueError):learn_campaign_payoff(bad,rows,refit=2)

    def test_unavailable_origin_never_becomes_imputed_zero_sample(self):
        x,rows=example();x[:12]=np.nan
        p,fits=learn_campaign_payoff(x,rows,refit=2)
        self.assertTrue(np.isnan(p[:20]).all())
        self.assertLessEqual(fits[-1]['samples'],12)
        self.assertEqual(fits[-1]['first_formation_session'],12)


if __name__=='__main__':unittest.main()

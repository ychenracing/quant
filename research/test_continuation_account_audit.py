"""Read-only diagnostics must not backfill future turnover or count open labels."""
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
from research.continuation_account_audit import volume_context


class AuditContextTests(unittest.TestCase):
    def fixture(self):
        calendar=pd.date_range('2023-01-03',periods=8,freq='B')
        volume=pd.DataFrame({'a':[1,2,3,4,5,6,7,8],'b':[5]*8},index=calendar)
        panels={'volume':volume,'raw_close':volume*0+10}
        market=SimpleNamespace(panel=lambda k:panels[k],calendar=calendar)
        cfg=SimpleNamespace(fast=2,slow=4)
        return market,cfg,panels

    def test_only_settled_campaigns_use_their_known_signal_context(self):
        m,c,p=self.fixture();dates=m.calendar
        rows=pd.DataFrame([{'symbol':'a','entry_signal':str(dates[3].date()),'exit':str(dates[6].date()),'pnl':20.,'buy_notional':100.},
                           {'symbol':'b','entry_signal':str(dates[3].date()),'exit':'OPEN','pnl':999.,'buy_notional':100.}])
        out,summary=volume_context(m,rows,c)
        self.assertEqual(len(out),1);self.assertEqual(summary['censored_open'],1)
        self.assertAlmostEqual(out.iloc[0].turnover_ratio,1.4)
        self.assertAlmostEqual(out.iloc[0].payoff,.2)

    def test_future_turnover_does_not_change_entry_context(self):
        m,c,p=self.fixture();dates=m.calendar
        rows=pd.DataFrame([{'symbol':'a','entry_signal':str(dates[3].date()),'exit':str(dates[6].date()),'pnl':20.,'buy_notional':100.}])
        a,_=volume_context(m,rows,c)
        p['volume'].iloc[4:]=999999
        b,_=volume_context(m,rows,c)
        pd.testing.assert_frame_equal(a,b,check_exact=True)

    def test_missing_and_open_data_are_not_zero_payoffs(self):
        m,c,p=self.fixture();p['volume'].iloc[:4]=np.nan;d=m.calendar
        rows=pd.DataFrame([{'symbol':'a','entry_signal':str(d[3].date()),'exit':str(d[6].date()),'pnl':20.,'buy_notional':100.}])
        out,summary=volume_context(m,rows,c)
        self.assertEqual(len(out),0);self.assertEqual(summary['unavailable_context'],1)
        self.assertIsNone(summary['payoff_rank_correlation'])


if __name__=='__main__':unittest.main()

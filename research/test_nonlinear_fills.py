"""Actual opening liquidation starts the three subsequent close observations."""
from dataclasses import replace
import unittest
import numpy as np
from test_core import sample_market
from techquant.data import Market
from techquant.engine import run
from research import nonlinear as n


class NonlinearOwnershipTests(unittest.TestCase):
    def market(self,count=1,missing=(31,32)):
        original=sample_market(count,70)
        frames={s:f.copy() for s,f in original.frames.items()}
        for frame in frames.values():
            for column in ('open','high','low','close','raw_open','raw_close'):
                frame[column]=20.
        frames[original.symbols[0]]=frames[original.symbols[0]].drop(index=original.calendar[list(missing)])
        return Market.from_frames(frames,original.calendar,quality='synthetic')

    def prediction(self,market,p):
        shape=(len(market.calendar),len(market.symbols))
        f=n.forecast(market,p)
        return replace(f,expected=np.full(shape,.1),tail=np.full(shape,.01),ready=np.ones(shape,dtype=bool),
                       price=np.full(shape,20.),ema10=np.full(shape,19.),ema20=np.full(shape,19.),
                       ema60=np.full(shape,19.),momentum5=np.full(shape,.01),ret1=np.zeros(shape))

    def run_scripted(self,market,p,prediction):
        observed={}
        class Observed(n.Owner):
            def __init__(self,m,params):
                super().__init__(m,params);self.f=prediction
            def decide(self,o):
                request=super().decide(o)
                observed[o.session]=(o.units.copy(),request.weights.copy(),self.exit_pending.copy())
                return request
        result=run(market,policy_factory=lambda m,c:Observed(m,p))
        return result,observed

    def test_three_closes_after_actual_opening_liquidation(self):
        market=self.market();p=n.Parameters();f=self.prediction(market,p);f.ret1[30]=-.09
        result,seen=self.run_scripted(market,p,f)
        blocked=[o for o in result.orders if o['side']=='SELL' and o['reason']=='NO_OPEN']
        self.assertEqual(len(blocked),2)
        sold=[o for o in result.orders if o['side']=='SELL' and o['status']=='FILLED']
        self.assertEqual(sold[0]['date'],str(market.calendar[33].date()))
        for day in (30,31,32):self.assertTrue(seen[day][2][0])
        for day in (33,34):self.assertEqual(seen[day][1][0],0.)
        self.assertGreater(seen[35][1][0],0.)
        buys=[o for o in result.orders if o['side']=='BUY' and o['status']=='FILLED']
        self.assertEqual(buys[1]['date'],str(market.calendar[36].date()))
        self.assertTrue((result.equity.cash>=0).all())

    def test_full_replacement_exit_cannot_disappear_after_a_blocked_open(self):
        market=self.market(count=5,missing=(21,));p=n.Parameters(20,.25,2);f=self.prediction(market,p)
        f.expected[:20]=[.3,.2,.05,.04,.03]
        f.expected[20:]=[.001,.2,.3,.4,.5]
        result,seen=self.run_scripted(market,p,f)
        self.assertEqual(seen[20][1][0],0.)
        self.assertEqual(seen[21][1][0],0.)
        sold=[o for o in result.orders if o['symbol']==market.symbols[0] and o['side']=='SELL' and o['status']=='FILLED']
        self.assertTrue(sold)
        self.assertEqual(sold[0]['date'],str(market.calendar[22].date()))

    def test_forecast_cache_is_horizon_specific_and_evidence_is_checked(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from research.nonlinear_measurement import EvidenceStudy
        market=self.market();p=n.Parameters()
        first=n.prepared(market,p)
        other=n.prepared(market,n.Parameters(20,.5,4))
        self.assertIs(first,other)
        self.assertFalse(first.expected.flags.writeable)
        self.assertFalse(first.tail.flags.writeable)
        self.assertTrue(any('smoothed_prevalence' in f for f in first.fits))
        with TemporaryDirectory() as folder:
            study=EvidenceStudy(Path(folder));study.forecasts()
            root=Path(folder)/'forecasts'/(first.data_sha256+'_h20')
            receipt=json.loads((root/'receipt.json').read_text())
            self.assertEqual(receipt['identity']['forecast_sha256'],first.fingerprint())
            arrays=root/'arrays.npz';arrays.write_bytes(arrays.read_bytes()+b'tamper')
            with self.assertRaisesRegex(ValueError,'evidence identity'):
                EvidenceStudy(Path(folder)).forecasts()

"""Critical accounting/causality contracts. No internet or historical matrix in CI."""
from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


def sample_market(n=3, days=170):
    from techquant.data import Market
    dates = pd.bdate_range('2023-01-03', periods=days)
    frames = {}
    for k in range(n):
        close = 20 * np.exp(np.arange(days) * (.002 + k * .0003))
        frames[f'sz300{100+k:03}'] = pd.DataFrame({
            'open': close * .999, 'high': close * 1.015,
            'low': close * .985, 'close': close, 'volume': 20_000_000.,
            'raw_open': close * .999, 'raw_close': close}, index=dates)
    return Market.from_frames(frames, dates, quality='synthetic')


class CoreTests(unittest.TestCase):
    def setUp(self):
        try:
            self.data = importlib.import_module('techquant.data')
            self.engine = importlib.import_module('techquant.engine')
            self.config = importlib.import_module('techquant.config')
        except ImportError as exc:
            self.fail(f'Independent implementation missing: {exc}')

    def test_config_rejects_nonsense(self):
        for kwargs in ({'fast': 0}, {'single_cap': 1.1}, {'risk_drawdown': -1},
                       {'rebalance': True}, {'slow': 10}, {'slippage_bps': float('nan')}):
            with self.assertRaises(ValueError):
                self.config.Config(**kwargs)

    def test_calendar_is_not_intersection_or_ipo_backfill(self):
        m = sample_market()
        frames = dict(m.frames)
        frames['sh688999'] = next(iter(frames.values())).iloc[-20:].copy()
        extended = self.data.Market.from_frames(frames, m.calendar, quality='synthetic')
        self.assertEqual(len(m.calendar), len(extended.calendar))
        self.assertTrue(extended.panel('close')['sh688999'].iloc[:150].isna().all())
        r = self.engine.run(extended)
        self.assertFalse(any(o['symbol'] == 'sh688999' and o['side'] == 'BUY'
                             and o['status'] == 'FILLED' for o in r.orders))

    def test_duplicate_dates_and_nonpositive_prices_fail(self):
        m = sample_market(1)
        s = m.symbols[0]
        f = m.frames[s]
        for bad in (pd.concat([f, f.iloc[-1:]]), f.assign(close=-1)):
            with self.assertRaises(ValueError):
                self.data.Market.from_frames({s: bad}, m.calendar)

    def test_no_future_information(self):
        m = sample_market()
        full = self.engine.run(m)
        cut = m.calendar[110]
        short = self.engine.run(m.prefix(cut))
        pd.testing.assert_frame_equal(full.equity.loc[:cut], short.equity)
        pd.testing.assert_frame_equal(full.targets.loc[:cut], short.targets)
        self.assertEqual([o for o in full.orders if o['date'] <= str(cut.date())],
                         short.orders)

    def test_next_open_cash_conservation_and_terminal_not_liquidated(self):
        m = sample_market(1, 90)
        w = pd.DataFrame(.8, index=m.calendar, columns=m.symbols)
        r = self.engine.run(m, targets=w)
        fills = [o for o in r.orders if o['status'] == 'FILLED']
        self.assertGreater(len(fills), 0)
        self.assertEqual(fills[0]['date'], str(m.calendar[1].date()))
        self.assertTrue(all(o['signal_date'] < o['date'] for o in fills))
        self.assertTrue((r.equity.cash >= -1e-6).all())
        self.assertTrue((r.equity.exposure <= 1 + 1e-9).all())
        np.testing.assert_allclose(r.equity.nav, r.equity.cash + r.equity.holdings,
                                   rtol=1e-12)
        self.assertGreater(r.equity.iloc[-1].holdings, 0)

    def test_limit_open_blocks_exit_and_retries(self):
        m = sample_market(1, 90)
        s = m.symbols[0]
        f = m.frames[s].copy()
        f.iloc[11, f.columns.get_loc('open')] = f.iloc[10].close * .799
        f.iloc[11, f.columns.get_loc('low')] = f.iloc[11].open * .99
        f.iloc[11, f.columns.get_loc('raw_open')] = f.iloc[11].open
        m = self.data.Market.from_frames({s: f}, m.calendar, quality='synthetic')
        w = pd.DataFrame(.8, index=m.calendar, columns=m.symbols)
        w.iloc[10:] = 0
        r = self.engine.run(m, targets=w)
        self.assertTrue(any(o['reason'] == 'OPEN_LIMIT' and o['side'] == 'SELL'
                            for o in r.orders))
        self.assertEqual(float(r.equity.holdings.iloc[12]), 0)

    def test_today_volume_not_used_for_open_capacity(self):
        m = sample_market(1, 90)
        s = m.symbols[0]
        w = pd.DataFrame(0., index=m.calendar, columns=m.symbols)
        w.iloc[29:] = .8
        a = self.engine.run(m, targets=w)
        f = m.frames[s].copy()
        f.iloc[30, f.columns.get_loc('volume')] = 1.
        b = self.engine.run(self.data.Market.from_frames({s: f}, m.calendar,
                                                        quality='synthetic'), targets=w)
        da = [o for o in a.orders if o['date'] == str(m.calendar[30].date())]
        db = [o for o in b.orders if o['date'] == str(m.calendar[30].date())]
        self.assertEqual(da, db)

    def test_protective_exit_ignores_rebalance_and_trade_band(self):
        from techquant.strategy import target_weights
        m = sample_market()
        from techquant.features import build_features
        f = build_features(m, self.config.Config())
        w, reasons = target_weights(100, f, np.array([.005, 0., 0.]),
                                    self.config.Config(), cap=0., rebalance=False)
        np.testing.assert_equal(w, np.zeros(3))
        self.assertIn('RISK_REDUCTION', reasons)

    def test_subset_order_and_single_name_are_deterministic(self):
        m = sample_market()
        a = self.engine.run(m.subset(m.symbols))
        b = self.engine.run(m.subset(tuple(reversed(m.symbols))))
        pd.testing.assert_frame_equal(a.equity, b.equity)
        c = self.engine.run(m.subset(m.symbols[:1]))
        self.assertTrue(np.isfinite(c.equity.nav).all())
        self.assertTrue((c.targets.sum(axis=1) <= 1).all())

    def test_invalid_external_targets_not_silently_normalized(self):
        m = sample_market()
        for value in (-.1, 1.01, float('nan')):
            with self.assertRaises(ValueError):
                self.engine.run(m, targets=pd.DataFrame(value, index=m.calendar,
                                                        columns=m.symbols))

    def test_cost_stress_fixed_trades_and_trade_fee_dates(self):
        from techquant.execution import stamp_rate, round_quantity
        self.assertEqual(stamp_rate('2023-08-25'), .001)
        self.assertEqual(stamp_rate('2023-08-28'), .0005)
        self.assertEqual(round_quantity('sh688001', 199), 0)
        self.assertEqual(round_quantity('sh688001', 201), 201)
        self.assertEqual(round_quantity('sz300001', 201), 200)
        self.assertEqual(round_quantity('bj920045', 101), 101)
        m = sample_market(1, 90)
        w = pd.DataFrame(0., index=m.calendar, columns=m.symbols)
        w.iloc[1:45] = .8
        a = self.engine.run(m, targets=w)
        b = self.engine.run(m, targets=w, cost_multiplier=2.)
        self.assertLess(b.equity.nav.iloc[-1], a.equity.nav.iloc[-1])

    def test_evidence_is_non_overwriting_and_checksummed(self):
        from techquant.evidence import save_result, verify_evidence
        r = self.engine.run(sample_market())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'
            save_result(r, root)
            self.assertTrue(verify_evidence(root))
            with self.assertRaises(FileExistsError):
                save_result(r, root)
            (root / 'equity.csv').write_text('tampered')
            with self.assertRaises(ValueError):
                verify_evidence(root)


if __name__ == '__main__':
    unittest.main()

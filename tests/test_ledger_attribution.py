"""Attribution is cash-conserving accounting, not a hypothetical stop fill."""
import unittest
import numpy as np
import pandas as pd
from test_core import sample_market
from techquant.engine import run
from research.ledger_attribution import attribute


class LedgerAttributionTests(unittest.TestCase):
    def test_partial_sells_and_final_liquidation_reconcile(self):
        m = sample_market(2, 90)
        targets = pd.DataFrame(0., index=m.calendar, columns=m.symbols)
        targets.iloc[:30] = [.45, .45]
        targets.iloc[30:50] = [.20, .45]
        r = run(m, targets=targets)
        daily, episodes, summary = attribute(m, r)
        self.assertLess(summary['max_reconciliation_error'], 1e-6)
        self.assertAlmostEqual(episodes.pnl.sum(), r.equity.nav.iloc[-1] - 2_000_000, places=5)
        self.assertTrue((episodes.exit != 'OPEN').all())
        self.assertTrue((daily.units_after >= -1e-8).all())
        broken = run(m, targets=targets)
        broken.orders[0]['fee'] += 100.
        with self.assertRaisesRegex(ValueError, 'reconcile'):
            attribute(m, broken)

    def test_open_episode_summary_is_strict_json(self):
        import json
        market = sample_market(2, 90)
        targets = pd.DataFrame(.45, index=market.calendar, columns=market.symbols)
        result = run(market, targets=targets)
        _, episodes, summary = attribute(market, result)
        self.assertTrue((episodes.exit == 'OPEN').any())
        json.dumps(summary, allow_nan=False)

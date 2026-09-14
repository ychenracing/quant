"""Causal scoring and actual-inventory contracts for registered tail authority."""
from dataclasses import asdict, replace
import importlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from test_core import sample_market
from techquant.data import Market
from techquant.engine import run
from techquant.policy import CloseObservation
from research.pathwise import Prediction
from research.trend_book import price_signals
from research.recovery_memory import Owner as PriceControl, Parameters as PriceParameters


def prediction(market, tail=None):
    p = price_signals(market)
    shape = p.price.shape
    tail = np.zeros(shape) if tail is None else np.array(tail, dtype=float, copy=True)
    probabilities = np.stack([tail, 1. - tail, np.zeros(shape)], axis=-1)
    return Prediction(market.symbols, market.fingerprint(), 20, np.zeros(shape), tail,
        p.ready.copy(), p.price.copy(), p.ema10.copy(), p.ema20.copy(),
        market.panel('close').ffill().ewm(span=60, adjust=False).mean().to_numpy(),
        p.momentum5.copy(), p.ret1.copy(), [], probabilities)


class ReliabilityTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research.risk_reliability'),
                             'the registered reliability policy has not been implemented')
        return importlib.import_module('research.risk_reliability')

    def test_only_registered_boolean_choices_and_exact_forecast_pool(self):
        module = self.module()
        self.assertEqual([asdict(p) for p in module.grid()],
                         [{'audit_tail': False}, {'audit_tail': True}])
        for invalid in (0, 1, 'true', None):
            with self.assertRaises(ValueError):
                module.Parameters(invalid)
        market = sample_market(3, 160)
        with self.assertRaises(ValueError):
            module.Owner(market.subset(market.symbols[:-1]), module.Parameters(),
                         prediction=prediction(market))
        from research.finite_study import Study
        with self.assertRaisesRegex(ValueError, 'pinned issued evidence'):
            Study('risk_reliability')

    def test_reference_balances_dates_and_uses_only_matured_prefix(self):
        module = self.module()
        outcomes = np.zeros((80, 2))
        outcomes[:40, 0] = -1.
        outcomes[:20, 1] = np.nan
        valid = np.ones(outcomes.shape, dtype=bool)
        self.assertTrue(np.isnan(module.historical_probability(outcomes, valid, 38)).all())
        answer = module.historical_probability(outcomes, valid, 39)
        # Twenty singly observed loss dates followed by twenty half-loss dates.
        decay = 2. ** ((np.arange(40) - 39) / 120.)
        loss = 40 * (decay[:20].sum() + .5 * decay[20:].sum()) / decay.sum()
        self.assertAlmostEqual(answer[0], (loss + 1.) / 43., places=14)
        self.assertAlmostEqual(answer.sum(), 1.)
        outcomes[40:] = 1.
        np.testing.assert_array_equal(answer, module.historical_probability(outcomes, valid, 39))
        # Replicating all observations on each date cannot increase its vote.
        np.testing.assert_allclose(answer, module.historical_probability(
            np.repeat(outcomes, 2, axis=1), np.repeat(valid, 2, axis=1), 39), rtol=0., atol=1e-15)

    def test_no_authority_until_40_fully_matured_issued_dates(self):
        module = self.module()
        market = sample_market(2, 190)
        audit = module.audit_forecasts(market, prediction(market))
        issued = np.flatnonzero(np.isfinite(audit.reference_probability[:, 0]))[0]
        first_score = issued + 20
        first_authority = first_score + 39
        self.assertTrue(np.isnan(audit.model_loss[:first_score]).all())
        self.assertFalse(audit.active[:first_authority].any())
        self.assertEqual(audit.scored_dates[first_authority], 40)
        self.assertTrue(audit.active[first_authority])
        self.assertAlmostEqual(audit.skill[first_authority], 1.)
        self.assertFalse(audit.active.flags.writeable)
        # Before a fitted probability exists, a warm-up tail prior is not a
        # scored forecast even though it happens to have a numeric value.
        issued_prediction = prediction(market)
        issued_prediction.outcome_probability[:90] = np.nan
        limited = module.audit_forecasts(market, issued_prediction)
        self.assertFalse(limited.active[:149].any())
        self.assertTrue(limited.active[149])

    def test_future_quotes_and_forecasts_cannot_change_past_audits(self):
        module = self.module()
        market = sample_market(3, 205)
        cut = 159
        full = module.audit_forecasts(market, prediction(market))
        prefix = market.prefix(market.calendar[cut])
        short = module.audit_forecasts(prefix, prediction(prefix))
        frames = {s: f.copy() for s, f in market.frames.items()}
        for frame in frames.values():
            frame.loc[market.calendar[cut + 1]:,
                ['open','high','low','close','raw_open','raw_close']] *= .35
        changed = Market.from_frames(frames, market.calendar, quality='synthetic')
        future = prediction(changed)
        future.tail[cut + 1:] = .99
        future.outcome_probability[cut + 1:, :, :] = [.99, .01, 0.]
        other = module.audit_forecasts(changed, future)
        for field in ('reference_probability','model_loss','reference_loss',
                      'scored_dates','skill','active','matured_outcome'):
            np.testing.assert_array_equal(getattr(full, field)[:cut + 1], getattr(short, field))
            np.testing.assert_array_equal(getattr(full, field)[:cut + 1], getattr(other, field)[:cut + 1])
        # Changing an excluded security cannot enter the exact-pool scorer.
        names = market.symbols[:-1]
        outside = {s: f.copy() for s, f in market.frames.items()}
        outside[market.symbols[-1]].loc[:,
            ['open','high','low','close','raw_open','raw_close']] *= 50.
        outside = Market.from_frames(outside, market.calendar, quality='synthetic')
        a, b = market.subset(names), outside.subset(names)
        np.testing.assert_array_equal(module.audit_forecasts(a, prediction(a)).skill,
                                      module.audit_forecasts(b, prediction(b)).skill)

    def test_inactive_model_is_price_control_without_any_fit(self):
        module = self.module()
        market = sample_market(3, 155)
        def factory(m, cfg):
            supplied = prediction(m)
            supplied.outcome_probability[:] = np.nan  # no scored model evidence
            return module.Owner(m, module.Parameters(True), prediction=supplied)
        with patch('sklearn.ensemble.HistGradientBoostingClassifier.fit',
                   side_effect=AssertionError('no model refitting is authorized')):
            actual = run(market, policy_factory=factory, delay=2)
        expected = run(market, policy_factory=lambda m,c: PriceControl(m, PriceParameters(False)), delay=2)
        pd.testing.assert_frame_equal(actual.equity.drop(columns='reason'), expected.equity.drop(columns='reason'))
        pd.testing.assert_frame_equal(actual.targets, expected.targets)
        self.assertEqual(actual.orders, expected.orders)
        prefix = run(market.prefix(market.calendar[120]), policy_factory=factory, delay=2)
        pd.testing.assert_frame_equal(actual.equity.loc[:market.calendar[120]], prefix.equity)
        self.assertTrue((actual.equity.cash >= 0).all())
        self.assertTrue(all(o['signal_date'] < o['date'] for o in actual.orders if o['status'] == 'FILLED'))

    def test_losing_model_authority_does_not_cancel_unfilled_exit(self):
        module = self.module()
        market = sample_market(2, 190)
        frames = {s: f.copy() for s, f in market.frames.items()}
        frames[market.symbols[0]].loc[market.calendar[160],
            ['open','high','low','close','raw_open','raw_close']] *= .97
        market = Market.from_frames(frames, market.calendar, quality='synthetic')
        supplied = prediction(market)
        supplied.tail[160, 0] = .8
        supplied.outcome_probability[160, 0] = [.8, .2, 0.]
        owner = module.Owner(market, module.Parameters(True), prediction=supplied)
        active = np.zeros(len(market.calendar), dtype=bool); active[160] = True
        owner.audit = replace(owner.audit, active=active)
        owner.risk.cap = 1.
        owner.risk.update = lambda *args: (1., 'CONTROLLED_ACCOUNT_CAP')
        def observe(i, units, cash):
            units = np.asarray(units, dtype=float)
            marks = owner.price_signals.price[i]
            nav = float(cash + units @ marks)
            return CloseObservation.from_inventory(i, str(market.calendar[i].date()), nav,
                                                   cash, units, units * marks / nav)
        first = owner.decide(observe(160, [10000., 0.], 200000.))
        retry = owner.decide(observe(161, [10000., 0.], 200000.))
        self.assertEqual(first.unit_targets[0], 0.)
        self.assertEqual(retry.unit_targets[0], 0.)
        self.assertEqual(retry.unit_targets[1], 0.)
        self.assertTrue(owner.exit_pending[0])
        self.assertIn('TAIL_EXIT_WARNING:' + market.symbols[0], first.reason)
        self.assertIn('TAIL_AUTHORITY:OFF', retry.reason)
        after_fill = owner.decide(observe(162, [0., 0.], 450000.))
        self.assertEqual(after_fill.unit_targets[0], 0.)
        self.assertTrue(owner.readmit[0])

    def test_audit_archive_is_bound_and_corruption_is_rejected(self):
        module = self.module()
        market = sample_market(2, 150)
        owner = module.Owner(market, module.Parameters(True), prediction=prediction(market))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                owner.preserve_audit(root, require_existing=True)
            archive = owner.preserve_audit(root)
            self.assertTrue(archive.is_file())
            self.assertEqual(owner.preserve_audit(root, require_existing=True), archive)
            archive.write_bytes(archive.read_bytes() + b'corruption')
            with self.assertRaises(ValueError):
                owner.preserve_audit(root, require_existing=True)


if __name__ == '__main__':
    unittest.main()

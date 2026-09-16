"""Contracts for campaign-specific alpha-decay displacement."""
from dataclasses import asdict, replace
import importlib, importlib.util, unittest
import numpy as np
import pandas as pd
from test_core import sample_market
from techquant.engine import run
from techquant.policy import CloseObservation
from research.trend_book import Owner as ParentOwner, Parameters as ParentParameters


def obs(owner, session, units, cash):
    units = np.asarray(units, dtype=float)
    marks = owner.price_signals.price[session]
    values = units * marks
    nav = float(cash + values.sum())
    return CloseObservation.from_inventory(
        session, str(owner.market.calendar[session].date()), nav, cash, units, values / nav
    )


def force_session(owner, session, scores, *, entries=None, exits=None, market=True):
    scores = np.asarray(scores, dtype=float)
    features = owner.features.score.copy()
    features[session] = scores
    owner.features = replace(owner.features, score=features)

    p = owner.price_signals
    ready = p.ready.copy(); ready[session] = True
    price = p.price.copy(); ema20 = p.ema20.copy(); momentum5 = p.momentum5.copy(); ret1 = p.ret1.copy()
    ema20[session] = np.where(np.isfinite(price[session]), price[session] * 0.9, 0.0)
    momentum5[session] = 0.1
    ret1[session] = 0.0
    owner.price_signals = replace(p, ready=ready, ema20=ema20, momentum5=momentum5, ret1=ret1)

    trend = owner.trend
    entry = trend.entry.copy(); exit_ = trend.exit.copy(); market_state = trend.market.copy()
    if entries is None:
        entry[session] = True
    else:
        entry[session] = np.asarray(entries, dtype=bool)
    if exits is None:
        exit_[session] = False
    else:
        exit_[session] = np.asarray(exits, dtype=bool)
    market_state[session] = bool(market)
    owner.trend = replace(trend, entry=entry, exit=exit_, market=market_state)


class OffensiveAlphaDecayDisplacementTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(
            importlib.util.find_spec('research.offensive_alpha_decay_displacement'),
            'alpha-decay displacement implementation is absent',
        )
        return importlib.import_module('research.offensive_alpha_decay_displacement')

    def test_control_is_exact_parent(self):
        m = self.module()
        self.assertEqual([asdict(p) for p in m.grid()], [{'enabled': False}, {'enabled': True}])
        market = sample_market(5, 145)
        parent = run(market, policy_factory=lambda current, cfg: ParentOwner(current, ParentParameters(2)))
        control = run(market, policy_factory=lambda current, cfg: m.Owner(current, m.Parameters(False)))
        pd.testing.assert_frame_equal(parent.equity, control.equity)
        pd.testing.assert_frame_equal(parent.targets, control.targets)
        self.assertEqual(parent.orders, control.orders)

    def test_registered_with_existing_paired_overlay(self):
        self.module()
        from research.finite_study import Study
        study = Study('offensive_alpha_decay_displacement')
        self.assertEqual(study.family, 'offensive_alpha_decay_displacement')
        self.assertEqual(len(study.module.grid()), 2)
        self.assertIn('offensive_alpha_decay_displacement.py', study.identity()['dependencies'])

    def test_alpha_discovery_selector_uses_cross_scope_compounding_and_floor(self):
        from research.finite_study import alpha_discovery_screen
        strong = [
            {'scope': 'union', 'control': {'wealth': 4.0}, 'treatment': {'wealth': 3.8}},
            {'scope': 'chatgpt_5', 'control': {'wealth': 8.0}, 'treatment': {'wealth': 10.0}},
            {'scope': 'joint_optical_leader_removal', 'control': {'wealth': 1.2}, 'treatment': {'wealth': 1.5}},
        ]
        decision = alpha_discovery_screen(strong)
        self.assertTrue(decision['advance'])
        self.assertGreater(decision['treatment_mean_log_wealth'], decision['control_mean_log_wealth'])
        self.assertGreater(decision['treatment_min_wealth'], decision['control_min_wealth'])

        common_only = [
            {'scope': 'union', 'control': {'wealth': 4.0}, 'treatment': {'wealth': 4.0}},
            {'scope': 'chatgpt_5', 'control': {'wealth': 8.0}, 'treatment': {'wealth': 16.0}},
            {'scope': 'joint_optical_leader_removal', 'control': {'wealth': 1.2}, 'treatment': {'wealth': 0.8}},
        ]
        rejected = alpha_discovery_screen(common_only)
        self.assertFalse(rejected['advance'])
        self.assertFalse(rejected['minimum_wealth_improved'])

    def test_initial_fill_records_causal_alpha_reference_when_inventory_arrives(self):
        m = self.module()
        owner = m.Owner(sample_market(5, 145), m.Parameters(True))
        first = owner.decide(obs(owner, 100, np.zeros(5), 2_000_000.0))
        chosen = np.flatnonzero(first.unit_targets > 1e-10)
        self.assertEqual(len(chosen), 2)
        self.assertAlmostEqual(float(first.weights.sum()), 1.0, 12)
        admission_scores = owner.features.score[100, chosen].copy()
        owner.decide(obs(owner, 101, first.unit_targets, 0.0))
        np.testing.assert_allclose(owner.owned_alpha_reference[chosen], admission_scores, rtol=0, atol=0)

    def test_full_book_displaces_decayed_incumbent_without_same_close_replacement(self):
        m = self.module()
        owner = m.Owner(sample_market(5, 145), m.Parameters(True))
        first = owner.decide(obs(owner, 100, np.zeros(5), 2_000_000.0))
        units = first.unit_targets.copy()
        owner.decide(obs(owner, 101, units, 0.0))
        chosen = list(np.flatnonzero(units > 1e-10))
        incumbent, survivor = chosen
        challenger = next(j for j in range(len(units)) if j not in chosen)
        scores = np.full(len(units), -1.0)
        scores[incumbent] = owner.owned_alpha_reference[incumbent] - 0.1
        scores[survivor] = owner.owned_alpha_reference[survivor] + 0.1
        scores[challenger] = owner.owned_alpha_reference[incumbent] + 1.0
        entries = np.zeros(len(units), dtype=bool); entries[challenger] = True
        force_session(owner, 102, scores, entries=entries)

        decision = owner.decide(obs(owner, 102, units, 0.0))

        self.assertEqual(float(decision.unit_targets[incumbent]), 0.0)
        self.assertEqual(float(decision.unit_targets[challenger]), 0.0)
        self.assertAlmostEqual(float(decision.unit_targets[survivor]), float(units[survivor]), 12)
        self.assertIn('ALPHA_DECAY_DISPLACEMENT', decision.reason)

    def test_blocked_displacement_sale_retries_without_creating_new_handoff(self):
        m = self.module()
        owner = m.Owner(sample_market(5, 145), m.Parameters(True))
        first = owner.decide(obs(owner, 100, np.zeros(5), 2_000_000.0))
        units = first.unit_targets.copy()
        owner.decide(obs(owner, 101, units, 0.0))
        chosen = list(np.flatnonzero(units > 1e-10))
        incumbent, survivor = chosen
        challenger = next(j for j in range(len(units)) if j not in chosen)
        scores = np.full(len(units), -1.0)
        scores[incumbent] = owner.owned_alpha_reference[incumbent] - 0.1
        scores[survivor] = owner.owned_alpha_reference[survivor] + 0.1
        scores[challenger] = owner.owned_alpha_reference[incumbent] + 1.0
        entries = np.zeros(len(units), dtype=bool); entries[challenger] = True
        force_session(owner, 102, scores, entries=entries)
        owner.decide(obs(owner, 102, units, 0.0))
        first_count = sum(r.get('action') == 'ALPHA_DECAY_DISPLACEMENT' for r in owner.trace)

        # Model a blocked next-open sale: the same inventory is still observed.
        force_session(owner, 103, scores, entries=entries)
        retry = owner.decide(obs(owner, 103, units, 0.0))
        second_count = sum(r.get('action') == 'ALPHA_DECAY_DISPLACEMENT' for r in owner.trace)
        self.assertEqual(first_count, 1)
        self.assertEqual(second_count, 1)
        self.assertEqual(float(retry.unit_targets[incumbent]), 0.0)
        self.assertIn('RETIRED_EXIT_RETRY', retry.reason)

    def test_displaced_symbol_reenters_only_after_later_campaign_break(self):
        m = self.module()
        owner = m.Owner(sample_market(5, 145), m.Parameters(True))
        first = owner.decide(obs(owner, 100, np.zeros(5), 2_000_000.0))
        units = first.unit_targets.copy()
        owner.decide(obs(owner, 101, units, 0.0))
        chosen = list(np.flatnonzero(units > 1e-10))
        incumbent, survivor = chosen
        challenger = next(j for j in range(len(units)) if j not in chosen)

        scores = np.full(len(units), -1.0)
        scores[incumbent] = owner.owned_alpha_reference[incumbent] - 0.1
        scores[survivor] = owner.owned_alpha_reference[survivor] + 0.1
        scores[challenger] = owner.owned_alpha_reference[incumbent] + 1.0
        entries = np.zeros(len(units), dtype=bool); entries[challenger] = True
        force_session(owner, 102, scores, entries=entries)
        displaced = owner.decide(obs(owner, 102, units, 0.0))
        after_sale = displaced.unit_targets.copy()

        # Even if the displaced name becomes the strongest eligible name on the
        # next close, it remains retired while its old campaign is intact.
        scores_103 = np.full(len(units), -1.0); scores_103[incumbent] = 50.0
        entries_103 = np.zeros(len(units), dtype=bool); entries_103[incumbent] = True
        force_session(owner, 103, scores_103, entries=entries_103)
        still_retired = owner.decide(obs(owner, 103, after_sale, 600_000.0))
        self.assertEqual(float(still_retired.unit_targets[incumbent]), 0.0)

        # A later security-specific break ends that retired campaign.  The break
        # close itself cannot buy; a later fresh entry may be funded normally.
        exits_104 = np.zeros(len(units), dtype=bool); exits_104[incumbent] = True
        force_session(owner, 104, scores_103, entries=entries_103, exits=exits_104)
        owner.decide(obs(owner, 104, after_sale, 600_000.0))
        force_session(owner, 105, scores_103, entries=entries_103)
        fresh = owner.decide(obs(owner, 105, after_sale, 600_000.0))
        self.assertGreater(float(fresh.unit_targets[incumbent]), 0.0)


if __name__ == '__main__':
    unittest.main()

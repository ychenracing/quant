"""Inventory tests use synthetic units; private integration needs an explicit path."""
import ast
import copy
import hashlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from research.native_inventory import (
    Inventory, RetryQuantity, REFERENCE_BLOB, EXPECTED_ANCHORS,
    _Correction, corrected_module,
)


def position(qty):
    return SimpleNamespace(shares=qty, cost=10.0)


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.book = Inventory()
        self.book.begin(0, 'day0', {})

    def buy(self, qty=100):
        self.book.record(('day0', 'A', 'BUY', 10., qty, 'entry'))

    def test_new_inventory_is_locked(self):
        self.buy()
        self.assertEqual(self.book.quote('A', 100, 100, 'hard_stop'), 0)
        self.assertEqual(self.book.owed['A'][0][0], 100)

    def test_prior_units_remain_sellable_after_topup(self):
        book = Inventory()
        book.begin(0, 'day0', {'A': position(100)})
        book.record(('day0', 'A', 'BUY', 10., 200, 'topup'))
        self.assertEqual(book.quote('A', 300, 300, 'hard_stop'), 100)
        book.record(('day0', 'A', 'SELL', 10., 100, 'hard_stop'))
        self.assertEqual(book.owed['A'][0][0], 200)
        book.close(1000., {'A': position(200)}, 3000.)

    def test_repeated_locked_requests_do_not_duplicate_obligation(self):
        self.buy(300)
        self.book.quote('A', 100, 300, 'hard_stop')
        self.book.quote('A', 100, 300, 'hard_stop')
        self.book.quote('A', 200, 300, 'hard_stop')
        self.assertEqual(self.book.owed['A'][0][0], 200)

    def test_actual_fill_not_quote_consumes_availability(self):
        book = Inventory()
        book.begin(0, 'day0', {'A': position(200)})
        self.assertEqual(book.quote('A', 100, 200, 'trim'), 100)
        self.assertEqual(book.quote('A', 100, 200, 'trim'), 100)
        book.record(('day0', 'A', 'SELL', 10., 100, 'trim'))
        self.assertEqual(book.available['A'], 100)

    def test_illegal_fill_raises(self):
        self.buy()
        with self.assertRaisesRegex(ValueError, 'opening sellable'):
            self.book.record(('day0', 'A', 'SELL', 10., 100, 'hard_stop'))

    def test_next_session_release_and_partial_retry(self):
        self.buy(300)
        self.book.quote('A', 200, 300, 'hard_stop')
        self.book.begin(1, 'day1', {'A': position(300)})
        self.assertEqual(self.book.eligible('A'), 200)
        self.book.record(('day1', 'A', 'SELL', 10., 100, 'other_protection'))
        self.assertEqual(self.book.quote('A', 200, 200, 'retry', RetryQuantity(200)), 100)
        self.book.record(('day1', 'A', 'SELL', 10., 100, 'retry'))
        self.assertFalse(self.book.owed)
        self.assertEqual(self.book.expected['A'], 100)

    def test_retry_keeps_units_without_fraction_roundoff(self):
        self.buy(73100)
        self.book.quote('A', 23300, 73100, 'hard_stop')
        self.book.begin(1, 'day1', {'A': position(73100)})
        orders = self.book.orders([], {'A': position(73100)})
        self.assertEqual(orders[0][2].quantity, 23300)
        self.assertEqual(self.book.quote('A', 73100, 73100, orders[0][3], orders[0][2]), 23300)

    def test_missing_session_does_not_lose_retry(self):
        self.buy()
        self.book.quote('A', 100, 100, 'hard_stop')
        for i in (1, 2):
            self.book.begin(i, f'day{i}', {'A': position(100)})
            self.assertEqual(len(self.book.orders([], {'A': position(100)})), 1)
            self.assertEqual(self.book.eligible('A'), 100)

    def test_native_sale_can_discharge_retry_without_double_selling(self):
        self.buy()
        self.book.quote('A', 100, 100, 'hard_stop')
        self.book.begin(1, 'day1', {'A': position(100)})
        self.book.record(('day1', 'A', 'SELL', 10., 100, 'signal_exit'))
        self.assertEqual(self.book.quote('A', 100, 0, 'retry', RetryQuantity(100)), 0)

    def test_pending_buys_blocked_until_protection_is_executed(self):
        self.buy()
        self.book.quote('A', 100, 100, 'hard_stop')
        self.book.begin(1, 'day1', {'A': position(100)})
        orders = self.book.orders([('A', 'BUY', .1, 'topup'), ('B', 'BUY', .1, 'entry')],
                                 {'A': position(100)})
        self.assertNotIn(('A', 'BUY', .1, 'topup'), orders)
        self.assertIn(('B', 'BUY', .1, 'entry'), orders)

    def test_flat_reset_waits_for_actual_sale_and_fires_once(self):
        self.buy()
        self.book.quote('A', 100, 100, 'dd_hard_limit')
        self.assertFalse(self.book.completed_flat_hard_exit({'A': position(100)}))
        self.book.begin(1, 'day1', {'A': position(100)})
        self.book.record(('day1', 'A', 'SELL', 10., 100, 'retry'))
        self.assertTrue(self.book.completed_flat_hard_exit({}))
        self.assertFalse(self.book.completed_flat_hard_exit({}))

    def test_unrelated_stop_does_not_create_hard_reset(self):
        self.buy()
        self.book.quote('A', 100, 100, 'hard_stop')
        self.book.begin(1, 'day1', {'A': position(100)})
        self.book.record(('day1', 'A', 'SELL', 10., 100, 'retry'))
        self.assertFalse(self.book.completed_flat_hard_exit({}))

    def test_terminal_new_inventory_is_not_forced_out(self):
        self.buy()
        self.assertEqual(self.book.quote('A', 100, 100, 'final_liquidate'), 0)
        self.book.close(1000., {'A': position(100)}, 2000., terminal=True)
        self.assertEqual(self.book.daily[-1]['holdings'], {'A': 100})

    def test_unrecorded_position_change_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'position changes'):
            self.book.begin(1, 'day1', {'A': position(100)})

    def test_invalid_units_and_duplicate_session_rejected(self):
        for value in (-1, .1, float('inf'), float('nan')):
            with self.assertRaises((ValueError, OverflowError)):
                self.book.units(value)
        with self.assertRaisesRegex(ValueError, 'advance'):
            self.book.begin(0, 'day0', {})

    def test_negative_cash_and_unreconciled_close_rejected(self):
        with self.assertRaisesRegex(ValueError, 'cash'):
            self.book.close(-1., {}, -1.)
        with self.assertRaisesRegex(ValueError, 'retained positions'):
            self.book.close(0., {'A': position(100)}, 1000.)

    def test_source_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'source.py'
            file.write_text('x = 1\n')
            with self.assertRaisesRegex(ValueError, 'immutable Git blob'):
                corrected_module('not_a_reference', file)


@unittest.skipUnless(os.environ.get('QUANT_NATIVE_REFERENCE'), 'private reference not configured')
class FixedReferenceIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = Path(os.environ['QUANT_NATIVE_REFERENCE']).read_bytes()
        assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == REFERENCE_BLOB
        tree = ast.parse(data)
        correction = _Correction()
        cls.tree = correction.visit(tree)
        assert sorted(correction.hits) == EXPECTED_ANCHORS
        ast.fix_missing_locations(cls.tree)

    def helper(self, old, new):
        book = Inventory()
        positions = {'A': position(old)} if old else {}
        book.begin(0, 'day0', positions)
        if new:
            book.record(('day0', 'A', 'BUY', 10., new, 'entry'))
        positions['A'] = position(old + new)
        scope = dict(np=np, positions=positions, px_at=lambda *args: 10.,
                     p={'slip': .001, 'fee': .00025, 'stamp_tax': .0005,
                        'cooldown_days': 10, 'vote_min': 2},
                     trades=[], cooldown={}, stop_count={}, _record_close=lambda *args: None,
                     _inventory=book)
        function = next(n for n in ast.walk(self.tree)
                        if isinstance(n, ast.FunctionDef) and n.name == 'exec_close_sell')
        exec(compile(ast.Module(body=[copy.deepcopy(function)], type_ignores=[]), '<native-helper>', 'exec'), scope)
        return scope, book

    def test_real_close_helper_retains_locked_inventory_and_no_fee(self):
        scope, book = self.helper(0, 52100)
        cash = scope['exec_close_sell']('A', 1., 'dd_hard_limit', 'day0', 0)
        self.assertEqual(cash, 0.)
        self.assertEqual(scope['trades'], [])
        self.assertEqual(scope['positions']['A'].shares, 52100)
        self.assertEqual(book.owed['A'][0][0], 52100)

    def test_real_close_helper_charges_only_actual_old_inventory(self):
        scope, book = self.helper(100, 200)
        cash = scope['exec_close_sell']('A', 1., 'dd_hard_limit', 'day0', 0)
        self.assertAlmostEqual(cash, 100 * 9.99 * (1 - .00075))
        self.assertEqual(scope['trades'][0][4], 100)
        self.assertEqual(scope['positions']['A'].shares, 200)
        book.close(cash, scope['positions'], cash + 2000.)

    def test_real_close_helper_unchanged_when_all_inventory_sellable(self):
        scope, book = self.helper(200, 0)
        cash = scope['exec_close_sell']('A', 1., 'dd_hard_limit', 'day0', 0)
        self.assertAlmostEqual(cash, 200 * 9.99 * (1 - .00075))
        self.assertEqual(scope['positions'], {})
        self.assertFalse(book.owed)
        book.close(cash, {}, cash)

    def test_full_private_module_compiles_under_strict_hash_and_anchor_guard(self):
        mod = corrected_module('inventory_test_reference', Path(os.environ['QUANT_NATIVE_REFERENCE']))
        self.assertEqual(mod._correction_identity['reference_git_blob'], REFERENCE_BLOB)
        self.assertEqual(sorted(mod._correction_identity['anchors']), EXPECTED_ANCHORS)


if __name__ == '__main__':
    unittest.main()

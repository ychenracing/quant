"""Tests for the fixed WorkBuddy execution-timing comparator.

Private-reference checks run only when QUANT_NATIVE_REFERENCE points at the
hash-pinned WorkBuddy entrypoint. No private source is copied into quant.
"""
from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import unittest

from research import native_inventory
from research.native_timing import (
    REFERENCE_BLOB,
    _TimingCorrection,
    corrected_module,
)


def _transform(file: Path):
    data = file.read_bytes()
    tree = ast.parse(data, filename=str(file))
    patch = _TimingCorrection()
    tree = patch.visit(tree)
    ast.fix_missing_locations(tree)
    return data, tree, patch


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    return next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


@unittest.skipUnless(os.environ.get('QUANT_NATIVE_REFERENCE'), 'private reference not configured')
class FixedReferenceTimingIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.file = Path(os.environ['QUANT_NATIVE_REFERENCE'])
        cls.data, cls.tree, cls.patch = _transform(cls.file)
        cls.blob = hashlib.sha1(
            b'blob ' + str(len(cls.data)).encode() + b'\0' + cls.data
        ).hexdigest()

    def test_reference_identity_and_all_correction_anchors_are_exact(self):
        self.assertEqual(self.blob, REFERENCE_BLOB)
        self.assertEqual(sorted(self.patch.hits), native_inventory.EXPECTED_ANCHORS)
        self.assertEqual(
            sorted(self.patch.timing_hits),
            sorted(_TimingCorrection.TIMING_ANCHORS),
        )

    def test_intraday_stop_inputs_use_only_prior_completed_session(self):
        run = _function(self.tree, 'run_backtest')
        text = ast.unparse(run)
        self.assertIn("px_at(sym, dates[i - 1], 'atr')", text)
        self.assertIn("px_at(sym, dates[i - 1], 'close')", text)
        self.assertNotIn("atr = row['atr']", text)
        self.assertNotIn("vol_pct = atr / row['close']", text)

    def test_close_derived_hard_breaker_is_deferred_to_pending_next_open(self):
        run = _function(self.tree, 'run_backtest')
        text = ast.unparse(run)
        self.assertIn("pending.append((s2, 'SELL', 0.0, 'dd_hard_limit'))", text)
        self.assertNotIn(
            "cash += exec_close_sell(s2, 1.0, 'dd_hard_limit', d, i, cd_ov)",
            text,
        )
        self.assertIn('_inventory.hard_wait = True', text)

    def test_full_private_module_compiles_under_strict_hash_and_anchor_guard(self):
        module = corrected_module('timing_test_reference', self.file)
        identity = module._correction_identity
        self.assertEqual(identity['reference_git_blob'], REFERENCE_BLOB)
        self.assertEqual(sorted(identity['inventory_anchors']), native_inventory.EXPECTED_ANCHORS)
        self.assertEqual(
            sorted(identity['timing_anchors']),
            sorted(_TimingCorrection.TIMING_ANCHORS),
        )

    def test_inventory_only_transform_retains_the_timing_defects_red_witness(self):
        # RED witness: the inventory-only correction deliberately leaves these
        # timing defects intact, so a separate correction is required.
        tree = ast.parse(self.data, filename=str(self.file))
        patch = native_inventory._Correction()
        tree = patch.visit(tree)
        ast.fix_missing_locations(tree)
        text = ast.unparse(_function(tree, 'run_backtest'))
        self.assertIn("atr = row['atr']", text)
        self.assertIn(
            "cash += exec_close_sell(s2, 1.0, 'dd_hard_limit', d, i, cd_ov)",
            text,
        )


if __name__ == '__main__':
    unittest.main()

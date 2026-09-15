"""Causal timing correction layered on the pinned WorkBuddy inventory comparator.

Research-only comparator. It preserves the native policy while correcting two
execution-timing inconsistencies: intraday stop levels use only prior-session
ATR/close inputs, and close-derived hard-breaker liquidations execute no earlier
than the next available open. T+1 inventory correction remains enforced.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
import types

from research import native
from research import native_inventory

REFERENCE_BLOB = native_inventory.REFERENCE_BLOB
CLASSIFICATION = 'TIMING_AND_INVENTORY_CORRECTED_NATIVE_NOT_UNIFORM_EXECUTION'


def _statements(code: str):
    return ast.parse(code).body


class _TimingCorrection(native_inventory._Correction):
    """Add causal stop inputs and defer close-derived breaker fills."""
    TIMING_ANCHORS = [1257, 1260, 1649, 1655]

    def __init__(self):
        super().__init__()
        self.timing_hits: list[int] = []

    def visit_Assign(self, node):
        line = node.lineno
        if line == 1257:
            # Stop thresholds used against today's intraday low must be known
            # before that low can occur. Use the last fully completed session.
            self.timing_hits.append(line)
            return _statements(
                'atr = px_at(sym, dates[i - 1], "atr") if i > 0 else np.nan'
            )
        if line == 1260:
            self.timing_hits.append(line)
            return _statements(
                '_stop_ref_close = px_at(sym, dates[i - 1], "close") if i > 0 else np.nan\n'
                'vol_pct = atr / _stop_ref_close if (not np.isnan(_stop_ref_close) and _stop_ref_close > 0) else 0'
            )
        return super().visit_Assign(node)

    def visit_AugAssign(self, node):
        line = node.lineno
        if line == 1655:
            self.timing_hits.append(line)
            # The completed close establishes this drawdown. Preserve the
            # decision but execute through the native pending-order next-open
            # path instead of granting the same close as a fill.
            return _statements(
                'pending.append((s2, "SELL", 0.0, "dd_hard_limit"))'
            )
        return self.generic_visit(node)

    def visit_If(self, node):
        line = node.lineno
        result = super().visit_If(node)
        if line == 1649:
            self.timing_hits.append(line)
            result.body = _statements('_inventory.hard_wait = True') + result.body
        return result


def corrected_module(name: str, file: Path):
    data = file.read_bytes()
    blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    if blob != REFERENCE_BLOB:
        raise ValueError('reference source is not the audited immutable Git blob')
    tree = ast.parse(data, filename=str(file))
    patch = _TimingCorrection()
    tree = patch.visit(tree)
    if sorted(patch.hits) != native_inventory.EXPECTED_ANCHORS:
        raise ValueError(f'inventory correction anchors differ: {sorted(patch.hits)}')
    if sorted(patch.timing_hits) != sorted(_TimingCorrection.TIMING_ANCHORS):
        raise ValueError(f'timing correction anchors differ: {sorted(patch.timing_hits)}')
    ast.fix_missing_locations(tree)
    value = types.ModuleType(name)
    value.__file__ = str(file)
    value.__dict__.update(
        _Inventory=native_inventory.Inventory,
        _RetryQuantity=native_inventory.RetryQuantity,
    )
    sys.modules[name] = value
    exec(compile(tree, str(file), 'exec'), value.__dict__)
    value._correction_identity = {
        'reference_git_blob': blob,
        'inventory_adapter_sha256': native.file_hash(Path(native_inventory.__file__)),
        'timing_adapter_sha256': native.file_hash(Path(__file__)),
        'transformed_ast_sha256': hashlib.sha256(ast.dump(tree).encode()).hexdigest(),
        'inventory_anchors': patch.hits,
        'timing_anchors': patch.timing_hits,
        'classification': CLASSIFICATION,
    }
    return value


def main() -> int:
    args = sys.argv[1:]
    if '--reference' not in args or args[args.index('--reference') + 1] != 'workbuddy':
        raise ValueError('timing comparison supports only the pinned workbuddy reference')
    output = Path(args[args.index('--output') + 1])
    original_loader = native.module
    captured = {}

    def loader(name, file):
        mod = corrected_module(name, file)
        function = mod.run_backtest

        def execute(*a, **kw):
            result = function(*a, **kw)
            captured['audit'] = result.pop('_inventory_audit')
            captured['identity'] = mod._correction_identity
            return result

        mod.run_backtest = execute
        return mod

    native.module = loader
    try:
        code = native.main()
    finally:
        native.module = original_loader
    if code:
        return code
    identity = json.loads((output / 'identity.json').read_text())
    identity['classification'] = CLASSIFICATION
    identity['execution_timing_correction'] = captured['identity']
    identity['limitations'] = [
        'WorkBuddy common-five only; not uniform four-reference execution or quant acceptance',
        'intraday stop thresholds use prior-session ATR/close but still preserve native stop logic',
        'close-derived hard breaker is deferred to native next-open pending execution',
        'T+1 sellable-inventory correction remains enforced',
        'native fees, liquidity, corporate-action units and other reference engines remain unnormalized',
    ]
    native.write_json(output / 'identity.json', identity)
    native.write_json(output / 'inventory_audit.json', captured['audit'])
    summary = json.loads((output / 'summary.json').read_text())
    summary['status'] = 'TIMING_AND_INVENTORY_CORRECTED_NATIVE_MEASURED'
    summary['acceptance'] = 'NOT_EQUIVALENT_EXECUTION'
    summary['outstanding_protection'] = captured['audit']['outstanding']
    native.write_json(output / 'summary.json', summary)
    native.write_json(output / 'manifest.json', {p.name: native.file_hash(p)
                      for p in sorted(output.iterdir()) if p.name != 'manifest.json'})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

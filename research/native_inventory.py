"""Inventory-only correction of one hash-pinned, private native comparator.

This module is research infrastructure, never a production policy. It changes
only sale quantities, execution-bound retries, and the accounting consequences
of retained inventory. The reference is transformed in memory; its source is
neither copied here nor changed on disk. All other native limitations remain.
"""
from __future__ import annotations

import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import types

from research import native

REFERENCE_BLOB = 'e19e611cd332e7a01c0f8677e0ae60600b295c6d'
CLASSIFICATION = 'INVENTORY_ONLY_CORRECTED_NATIVE_NOT_UNIFORM_EXECUTION'


class RetryQuantity(float):
    """Keep the existing native order tuple, but bind a retry to actual units."""
    def __new__(cls, quantity: int):
        value = super().__new__(cls, 1.0)
        value.quantity = quantity
        return value


class Inventory:
    """Opening inventory less actual sales; additions settle next session.

    Residuals are quantity-bound and grouped by creation session. Repeated
    requests for the same locked inventory overlap (maximum, not sum). Sales
    cannot discharge a residual created from today's newly bought inventory.
    """
    def __init__(self):
        self.session = -1
        self.date = ''
        self.available: dict[str, int] = {}
        self.expected: dict[str, int] = {}
        self.owed: dict[str, dict[int, tuple[int, str]]] = defaultdict(dict)
        self.events: list[dict] = []
        self.daily: list[dict] = []
        self.hard_wait = False

    @staticmethod
    def units(value) -> int:
        result = int(value)
        if result != value or result < 0:
            raise ValueError('inventory must be nonnegative integral units')
        return result

    def begin(self, session: int, date, positions) -> None:
        if session <= self.session:
            raise ValueError('inventory sessions must advance exactly once')
        actual = {s: self.units(p.shares) for s, p in positions.items() if p.shares}
        if self.session >= 0 and actual != {s: q for s, q in self.expected.items() if q}:
            raise ValueError('position changes were not reconciled to actual fills')
        self.session, self.date = session, str(date)
        self.available = dict(actual)
        self.expected = dict(actual)

    def eligible(self, symbol: str) -> int:
        return sum(q for day, (q, _) in self.owed.get(symbol, {}).items()
                   if day < self.session)

    def orders(self, pending, positions):
        locked_names = {s for s in self.owed if self.eligible(s)}
        # Do not execute a previously queued buy against a protective retry.
        orders = []
        for order in pending:
            if order[1] == 'BUY' and order[0] in locked_names:
                self.events.append(dict(date=self.date, kind='BUY_BLOCKED_BY_RETRY',
                                        symbol=order[0], reason=order[3]))
            else:
                orders.append(order)
        for symbol in sorted(locked_names):
            qty = self.eligible(symbol)
            if symbol not in positions or qty > positions[symbol].shares:
                raise ValueError('protective residual exceeds retained inventory')
            reason = next(r for day, (_, r) in sorted(self.owed[symbol].items())
                          if day < self.session)
            orders.append((symbol, 'SELL', RetryQuantity(qty), 'inventory_retry:' + reason))
        return orders

    def quote(self, symbol: str, requested, held, reason: str, token=None) -> int:
        held, requested = self.units(held), self.units(requested)
        if isinstance(token, RetryQuantity):
            requested = min(token.quantity, self.eligible(symbol))
        requested = min(requested, held)
        allowed = min(requested, self.available.get(symbol, 0))
        blocked = requested - allowed
        if blocked:
            prior, prior_reason = self.owed[symbol].get(self.session, (0, reason))
            self.owed[symbol][self.session] = (max(prior, blocked), prior_reason)
            self.hard_wait |= reason in ('dd_hard_limit', 'dd_hard_limit_open')
            self.events.append(dict(date=self.date, kind='INVENTORY_BLOCK', symbol=symbol,
                                    requested=requested, available=allowed, blocked=blocked,
                                    reason=reason))
        return allowed

    def record(self, trade) -> None:
        date, symbol, side, price, quantity, reason = trade
        qty = self.units(quantity)
        if not qty or str(date) != self.date:
            raise ValueError('invalid fill quantity or session')
        if side == 'BUY':
            self.expected[symbol] = self.expected.get(symbol, 0) + qty
            return
        if side != 'SELL' or qty > self.available.get(symbol, 0):
            raise ValueError('actual sale exceeded opening sellable inventory')
        self.available[symbol] -= qty
        self.expected[symbol] = self.expected.get(symbol, 0) - qty
        left = qty
        for day in sorted(list(self.owed.get(symbol, {}))):
            if day >= self.session or not left:
                continue
            outstanding, original_reason = self.owed[symbol][day]
            discharge = min(left, outstanding)
            left -= discharge
            remaining = outstanding - discharge
            if remaining:
                self.owed[symbol][day] = (remaining, original_reason)
            else:
                del self.owed[symbol][day]
            self.events.append(dict(date=self.date, kind='RETRY_DISCHARGED', symbol=symbol,
                                    quantity=discharge, origin_session=day,
                                    original_reason=original_reason, fill_reason=reason))
        if symbol in self.owed and not self.owed[symbol]:
            del self.owed[symbol]

    def completed_flat_hard_exit(self, positions) -> bool:
        # Preserve native reset semantics, but only once clearance truly fills.
        if self.hard_wait and not positions and not self.owed:
            self.hard_wait = False
            self.events.append(dict(date=self.date, kind='DEFERRED_NATIVE_FLAT_RESET'))
            return True
        return False

    def close(self, cash, positions, nav, terminal=False) -> None:
        actual = {s: self.units(p.shares) for s, p in positions.items() if p.shares}
        if actual != {s: q for s, q in self.expected.items() if q}:
            raise ValueError('retained positions do not equal reconciled fills')
        if cash < -1e-7:
            raise ValueError('native cash became negative')
        self.daily.append(dict(date=self.date, cash=float(cash), holdings=actual,
                               nav=float(nav), terminal=terminal))


def _statements(code: str):
    return ast.parse(code).body


class _Correction(ast.NodeTransformer):
    """The numeric anchors are safe only together with the full-source hash."""
    FILLS = {1068, 1194, 1219, 1293, 1314, 1983}
    FULL = {1291: ('sym', 'reason'),
            1312: ('s2', "'dd_hard_limit_open'"),
            1981: ('sym', "'final_liquidate'")}

    def __init__(self):
        self.hits: list[int] = []

    def generic_visit(self, node):
        # Transform every original block before visiting newly inserted nodes.
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                new = []
                for item in value:
                    if isinstance(item, ast.AST):
                        result = self.visit(item)
                        new.extend(result if isinstance(result, list) else [result])
                    else:
                        new.append(item)
                if field in ('body', 'orelse', 'finalbody'):
                    for first, last in ((1294, 1303), (1315, 1318), (1984, 1984)):
                        indices = [j for j, x in enumerate(new)
                                   if first <= getattr(x, 'lineno', -1) <= last]
                        if indices and any(getattr(x, 'lineno', -1) == first for x in new):
                            lo, hi = min(indices), max(indices)
                            original = new[lo:hi + 1]
                            conditional = ast.If(
                                test=ast.parse('_sale == pos.shares', mode='eval').body,
                                body=original,
                                orelse=_statements('pos.shares -= _sale'))
                            new[lo:hi + 1] = [conditional]
                            self.hits.append(first)
                setattr(node, field, new)
            elif isinstance(value, ast.AST):
                setattr(node, field, self.visit(value))
        return node

    def visit_Assign(self, node):
        line = node.lineno
        node = self.generic_visit(node)
        if line == 940:
            self.hits.append(line)
            return [node] + _statements('_inventory = _Inventory()')
        if line in (1062, 1201):
            self.hits.append(line)
            token = ', tgt_w' if line == 1201 else ''
            extra = _statements(f'sell_shares = _inventory.quote(sym, sell_shares, pos.shares, reason{token})')
            if line == 1201:
                extra += _statements('if sell_shares <= 0:\n    continue')
            return [node] + extra
        if line in self.FULL:
            self.hits.append(line)
            symbol, reason = self.FULL[line]
            node.value.left = ast.Name(id='_sale', ctx=ast.Load())
            return _statements(f'_sale = _inventory.quote({symbol}, pos.shares, pos.shares, {reason})\nif _sale <= 0:\n    continue') + [node]
        if line == 1238:
            self.hits.append(line)
            return [node] + _statements('''if _inventory.completed_flat_hard_exit(positions) and p['dd_hard_reset']:
    peak_equity = cash
    chandelier_level = 0''')
        if line == 1985:
            self.hits.append(line)
            node.value.elts[1] = ast.Call(func=ast.Name(id='_recompute_equity', ctx=ast.Load()), args=[], keywords=[])
            return [node] + _statements('_inventory.close(cash, positions, equity_curve[-1][1], terminal=True)')
        return node

    def visit_Expr(self, node):
        line = node.lineno
        node = self.generic_visit(node)
        if line in self.FILLS:
            self.hits.append(line)
            if line in (1293, 1314, 1983):
                node.value.args[0].elts[4] = ast.Name(id='_sale', ctx=ast.Load())
            return [node] + _statements('_inventory.record(trades[-1])')
        return node

    def visit_For(self, node):
        line = node.lineno
        node = self.generic_visit(node)
        if line == 1093:
            self.hits.append(line)
            node.body = _statements('_inventory.begin(i, d, positions)\npending = _inventory.orders(pending, positions)') + node.body
            node.body += _statements('_inventory.close(cash, positions, equity_curve[-1][1])')
        return node

    def visit_If(self, node):
        line = node.lineno
        node = self.generic_visit(node)
        if line in (1213, 1321):
            self.hits.append(line)
            extra = 'not isinstance(tgt_w, _RetryQuantity)' if line == 1213 else 'not positions'
            node.test = ast.BoolOp(op=ast.And(), values=[ast.parse(extra, mode='eval').body, node.test])
        return node

    def visit_Return(self, node):
        if node.lineno == 2036:
            self.hits.append(node.lineno)
            return _statements("result['_inventory_audit'] = dict(events=_inventory.events, daily=_inventory.daily, outstanding=dict(_inventory.owed))") + [node]
        return node


EXPECTED_ANCHORS = sorted([940, 1062, 1201, 1291, 1312, 1981, 1238, 1985,
                           1068, 1194, 1219, 1293, 1314, 1983, 1093, 1213,
                           1321, 2036, 1294, 1315, 1984])


def corrected_module(name: str, file: Path):
    data = file.read_bytes()
    blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    if blob != REFERENCE_BLOB:
        raise ValueError('reference source is not the audited immutable Git blob')
    tree = ast.parse(data, filename=str(file))
    patch = _Correction()
    tree = patch.visit(tree)
    if sorted(patch.hits) != EXPECTED_ANCHORS:
        raise ValueError(f'native correction anchors differ: {sorted(patch.hits)}')
    ast.fix_missing_locations(tree)
    value = types.ModuleType(name)
    value.__file__ = str(file)
    value.__dict__.update(_Inventory=Inventory, _RetryQuantity=RetryQuantity)
    sys.modules[name] = value
    exec(compile(tree, str(file), 'exec'), value.__dict__)
    value._correction_identity = {
        'reference_git_blob': blob,
        'adapter_sha256': native.file_hash(Path(__file__)),
        'transformed_ast_sha256': hashlib.sha256(ast.dump(tree).encode()).hexdigest(),
        'anchors': patch.hits,
        'classification': CLASSIFICATION,
    }
    return value


def main() -> int:
    args = sys.argv[1:]
    if '--reference' not in args or args[args.index('--reference') + 1] != 'workbuddy':
        raise ValueError('inventory comparison supports only the pinned workbuddy reference')
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
    identity['inventory_correction'] = captured['identity']
    identity['limitations'] = [
        'T+1 inventory only; not uniform execution or original quant acceptance',
        'native same-session close/ATR information timing remains unchanged',
        'native risk resets are retained, delayed only until actual flat clearance',
        'native fees, sell liquidity, price availability and corporate-action units remain unnormalized',
        'terminal newly acquired inventory remains marked to market with explicit obligations',
    ]
    native.write_json(output / 'identity.json', identity)
    native.write_json(output / 'inventory_audit.json', captured['audit'])
    summary = json.loads((output / 'summary.json').read_text())
    summary['status'] = 'INVENTORY_CORRECTED_NATIVE_MEASURED'
    summary['acceptance'] = 'NOT_EQUIVALENT_EXECUTION'
    summary['outstanding_protection'] = captured['audit']['outstanding']
    native.write_json(output / 'summary.json', summary)
    native.write_json(output / 'manifest.json', {p.name: native.file_hash(p)
                      for p in sorted(output.iterdir()) if p.name != 'manifest.json'})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Offline command surface checks; no market service or broker is contacted."""
import contextlib
import importlib
import io
from pathlib import Path
import tempfile
import unittest
from test_core import sample_market
from techquant.engine import run
from techquant.evidence import save_result


class CommandTests(unittest.TestCase):
    def setUp(self):
        try:
            self.cli = importlib.import_module('techquant.cli')
        except ImportError as exc:
            self.fail(str(exc))

    def test_help_is_offline(self):
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as e:
            self.cli.main(['--help'])
        self.assertEqual(e.exception.code, 0)

    def test_verify_command_checks_actual_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'result'
            save_result(run(sample_market()), root)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(self.cli.main(['verify', str(root)]), 0)
            (root/'equity.csv').write_text('damaged')
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(self.cli.main(['verify', str(root)]), 2)

    def test_missing_snapshot_is_an_error_not_an_empty_backtest(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(self.cli.main(['audit','--data','/nonexistent/snapshot',
                                           '--catalog','/nonexistent/catalog.json']), 2)

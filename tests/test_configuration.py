"""One regression: the public API and documented research preset must agree."""
from dataclasses import asdict
import json
from pathlib import Path
import unittest
from techquant.config import Config

class ConfigurationTests(unittest.TestCase):
    def test_api_defaults_equal_explicit_research_configuration(self):
        root = Path(__file__).resolve().parents[1]
        selected = json.loads((root / "config/research.json").read_text())
        self.assertEqual(asdict(Config()), selected)

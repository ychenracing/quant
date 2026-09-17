"""One-shot deterministic migration for independent market risk confirmation.

This helper is removed by the workflow after it applies and verifies the final
production/test changes.  It exists only to avoid transmitting large whole-file
replacements through a string-only connector.
"""
from pathlib import Path


def replace_once(path: str, old: str, new: str, label: str) -> None:
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit(f"{label} anchor mismatch")
    file.write_text(text.replace(old, new), encoding="utf-8")


replace_once(
    "src/techquant/risk_ownership.py",
    '''        channels = {
            "TREND": bool(self.trend_damage[i]),
            "BREADTH": bool(self.breadth_damage[i]),
            "VOLATILITY": bool(self.volatility_damage[i]),
            "ACCOUNT_ACCELERATION": account,
        }
        active = [name for name, enabled in channels.items() if enabled]
        defensive = len(active) >= 2
        crisis = bool(self.market_shock[i] and len(active) >= 1)
        return active, defensive, crisis, drawdown
''',
    '''        market_channels = {
            "TREND": bool(self.trend_damage[i]),
            "BREADTH": bool(self.breadth_damage[i]),
            "VOLATILITY": bool(self.volatility_damage[i]),
        }
        active_market = [
            name for name, enabled in market_channels.items() if enabled
        ]
        active = active_market + (["ACCOUNT_ACCELERATION"] if account else [])
        # Market structure must confirm the event independently. Account loss is
        # path-dependent severity evidence, not a second vote for the same price
        # shock and never establishes protection by itself.
        defensive = len(active_market) >= 2
        crisis = bool(defensive and (self.market_shock[i] or account))
        return active, defensive, crisis, drawdown
''',
    "risk snapshot",
)

replace_once(
    "tests/test_risk_ownership.py",
    "        policy.trend_damage[10] = True\n        policy.strength[10] = np.array([3.0, 2.0, -3.0])\n",
    "        policy.trend_damage[10] = True\n        policy.breadth_damage[10] = True\n        policy.strength[10] = np.array([3.0, 2.0, -3.0])\n",
    "selective protection test",
)
replace_once(
    "tests/test_risk_ownership.py",
    "        policy.market_shock[0] = True\n        policy.trend_damage[0] = True\n        units = np.array([150.0])\n",
    "        policy.market_shock[0] = True\n        policy.trend_damage[0] = True\n        policy.breadth_damage[0] = True\n        units = np.array([150.0])\n",
    "shock recovery test",
)
replace_once(
    "tests/test_risk_ownership.py",
    '''    def test_persistent_risk_reuses_one_absolute_protection_goal(self):
''',
    '''    def test_account_loss_cannot_confirm_a_market_shock(self):
        market = sample_market(2, 20)
        policy = RiskAwareOwnershipPolicy(market, Config())
        policy.trend_damage[:] = False
        policy.breadth_damage[:] = False
        policy.volatility_damage[:] = False
        policy.market_shock[:] = False
        units = np.zeros(2)
        policy.decide(direct_observation(policy, 0, units, units, nav=100.0))
        policy.market_shock[1] = True

        decision = policy.decide(
            direct_observation(policy, 1, units, units, nav=80.0)
        )

        self.assertEqual(policy.state, "CAUTION")
        self.assertIn("SHOCK,ACCOUNT_ACCELERATION", decision.reason)
        self.assertNotIn("SYSTEMIC_PROTECTION", decision.reason)

    def test_persistent_risk_reuses_one_absolute_protection_goal(self):
''',
    "account severity regression",
)
replace_once(
    "research/risk_ownership_screen.py",
    '                        reasons.str.contains("FUNDED_RECOVERY_COMPLETE:", regex=False).sum()\n',
    '                        reasons.str.contains("FUNDED_RECOVERY_COMPLETE", regex=False).sum()\n',
    "recovery metric",
)

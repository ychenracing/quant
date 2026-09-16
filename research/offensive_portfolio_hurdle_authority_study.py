"""Source-bound portfolio-hurdle study against the current alpha leader."""
from __future__ import annotations

from dataclasses import asdict
import importlib
import json
import math
from pathlib import Path

import pandas as pd

from techquant.config import Config
from techquant.data import file_hash
from techquant.engine import run
from techquant.evidence import load_result, metrics, save_result, source_identity
from research.expectation_study import scopes, write_json
from research.offensive_fresh_challenger_authority import Owner as LeaderOwner, Parameters as LeaderParameters


class Study:
    FAMILY = "offensive_portfolio_hurdle_authority"
    REGISTRATION_COMMIT = "c4cdea7bfb13507ba9db44f47c6a4fb927472aec"

    def __init__(self, family: str, *, issued_evidence=None):
        if family != self.FAMILY:
            raise ValueError("portfolio hurdle study only accepts its preregistered family")
        if issued_evidence is not None:
            raise ValueError("portfolio hurdle study does not accept issued forecasts")
        self.family = family
        self.module = importlib.import_module("research." + family)

    def identity(self):
        root = Path(__file__).parent
        names = [
            "offensive_portfolio_hurdle_authority_study.py",
            "offensive_portfolio_hurdle_authority.py",
            "offensive_portfolio_hurdle_authority_contract.json",
            "offensive_fresh_challenger_authority.py",
            "offensive_fresh_challenger_authority_contract.json",
            "offensive_alpha_decay_displacement.py",
            "offensive_alpha_decay_displacement_contract.json",
            "finite_study/__init__.py",
            "trend_book.py", "trend_book_contract.json", "coherent.py", "coherent_contract.json",
            "observed_trend.py", "observed_trend_contract.json", "nonlinear.py", "nonlinear_contract.json",
            "pathwise.py", "pathwise_contract.json", "expectation_study.py", "leadership.py", "expectation.py",
            "quantity_obligation.py", "quantity_obligation_contract.json", "support_budget.py",
            "support_budget_contract.json", "funded_risk.py", "funded_risk_contract.json",
            "observed_readiness.py", "observed_readiness_contract.json",
            "admission_budget_completion.py", "admission_budget_completion_contract.json",
            "observed_admission_completion.py", "observed_admission_completion_contract.json",
            "decision_review.py", "ledger_attribution.py",
        ]
        return {"source": source_identity(), "family": self.family,
                "dependencies": {name: file_hash(root / name) for name in names}}

    def _saved(self, market, path, owner, *, costs=1.0, delay=1):
        cfg = Config()
        expected = {
            "config": asdict(cfg), "universe": list(market.symbols), "quality": market.quality,
            "data_sha256": market.fingerprint(), "source": source_identity(), "provenance": market.provenance,
            "delay": delay, "cost_multiplier": costs, "benchmark": None,
            "start": str(market.calendar[0].date()), "end": str(market.calendar[-1].date()),
            "economic_acceptance": "UNVERIFIED", "accounting": "adjusted economic units, not actual shares",
            "study": self.identity(), "policy": owner.identity(),
        }
        if path.exists():
            return load_result(path, expected=expected)
        result = run(market, cfg, policy_factory=lambda _m, _c: owner,
                     cost_multiplier=costs, delay=delay)
        result.metadata["study"] = self.identity()
        if result.metadata != expected:
            raise AssertionError("unexpected portfolio hurdle study identity")
        save_result(result, path)
        return result

    @staticmethod
    def _objective(rows, label):
        wealth = [float(row[label]["wealth"]) for row in rows]
        if any(value <= 0 for value in wealth):
            raise ValueError("terminal wealth must be positive")
        return {"mean_log": math.fsum(math.log(value) for value in wealth) / len(wealth),
                "minimum": min(wealth),
                "wealth": {row["scope"]: float(row[label]["wealth"]) for row in rows}}

    def select(self, market, catalog, out):
        root = Path(__file__).parent
        contract = json.loads((root / "offensive_portfolio_hurdle_authority_contract.json").read_text())
        full_sha = market.fingerprint()
        if full_sha != contract["frozen_inputs"]["full_sha256"]:
            raise ValueError("portfolio hurdle requires frozen full market")
        market = market.prefix(contract["measurement"]["window"]["end"])
        scoped = scopes(market, catalog)
        out = Path(out)
        plan = {"identity": self.identity(), "full_data_sha256": full_sha,
                "data_sha256": market.fingerprint(),
                "selection_end": contract["measurement"]["window"]["end"],
                "scopes": scoped, "registration_commit": self.REGISTRATION_COMMIT,
                "policies": ["baseline", "current_leader", "treatment"]}
        write_json(out / "plan.json", plan)
        rows = []
        for name, names in scoped.items():
            subset = market.subset(names)
            owners = {
                "baseline": self.module.Owner(subset, self.module.Parameters(False)),
                "current_leader": LeaderOwner(subset, LeaderParameters(True)),
                "treatment": self.module.Owner(subset, self.module.Parameters(True)),
            }
            row = {"scope": name}
            for label, owner in owners.items():
                result = self._saved(subset, out / "runs" / f"{name}_{label}", owner)
                row[label] = dict(metrics(result), average_exposure=float(result.equity.exposure.mean()))
            rows.append(row)
            write_json(out / "paired-progress.json", rows)
            print(json.dumps(row), flush=True)
        baseline = self._objective(rows, "baseline")
        leader = self._objective(rows, "current_leader")
        treatment = self._objective(rows, "treatment")
        advance = treatment["mean_log"] > leader["mean_log"] + 1e-12 and treatment["minimum"] >= leader["minimum"] - 1e-12
        decision = {
            "status": "BEATS_CURRENT_ALPHA_LEADER" if advance else "REJECTED_VS_CURRENT_ALPHA_LEADER",
            "advance": advance, "rows": rows,
            "baseline_objective": baseline, "current_leader_objective": leader,
            "treatment_objective": treatment, "identity": self.identity(),
            "data_sha256": market.fingerprint(), "full_data_sha256": full_sha,
            "plan_sha256": file_hash(out / "plan.json"),
            "registration_commit": self.REGISTRATION_COMMIT,
            "economic_acceptance": "NOT_ESTABLISHED",
            "historical_exposure": "RETROSPECTIVE_NOT_UNSEEN_OUT_OF_SAMPLE",
        }
        write_json(out / "selection.json", decision)

    def evaluate(self, market, catalog, selection, out):
        selection = Path(selection)
        chosen = json.loads(selection.read_text())
        out = Path(out)
        if chosen["identity"] != self.identity() or chosen["full_data_sha256"] != market.fingerprint():
            raise ValueError("portfolio hurdle selection identity mismatch")
        if not chosen["advance"]:
            write_json(out.parent / "evaluation-decision.json", {
                "status": "NOT_RUN_REJECTED_VS_CURRENT_LEADER",
                "selection_sha256": file_hash(selection), "full_history_runs": 0})
            return
        windows = {
            "full": (None, None), "bull": ("2023-01-03", "2026-06-30"),
            "late_june_through_august": ("2026-06-22", "2026-08-31"),
            "july_august": ("2026-07-01", "2026-08-31"), "retrospective_2026": ("2026-01-01", None),
        }
        rows = []
        for name, names in scopes(market, catalog).items():
            subset = market.subset(names)
            owners = {
                "current_leader": LeaderOwner(subset, LeaderParameters(True)),
                "treatment": self.module.Owner(subset, self.module.Parameters(True)),
            }
            for label, owner in owners.items():
                result = self._saved(subset, out / "runs" / f"{name}_{label}", owner)
                for window, (start, end) in windows.items():
                    rows.append({"scope": name, "policy": label, "window": window,
                                 **metrics(result, start, end)})
            pd.DataFrame(rows).to_csv(out / "matrix.csv", index=False, float_format="%.17g")
        write_json(out / "status.json", {"identity": self.identity(), "runs": 6, "rows": len(rows),
            "status": "MEASURED_NOT_FINAL_ACCEPTANCE", "formal_matrix": "NOT_RUN"})

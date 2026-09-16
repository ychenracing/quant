"""Source-bound paired study for the preregistered settled-reference-rearm family."""
from __future__ import annotations

from dataclasses import asdict
import importlib
import json
from pathlib import Path

import pandas as pd

from techquant.config import Config
from techquant.data import file_hash
from techquant.engine import run
from techquant.evidence import load_result, metrics, save_result, source_identity
from research.expectation_study import scopes, write_json
from research.finite_study import alpha_discovery_screen


class Study:
    FAMILY = "offensive_settled_reference_rearm"
    REGISTRATION_COMMIT = "bce12d4edb27025537ec08b098901fd6f91073c7"

    def __init__(self, family: str, *, issued_evidence=None):
        if family != self.FAMILY:
            raise ValueError("settled reference rearm study only accepts its preregistered family")
        if issued_evidence is not None:
            raise ValueError("settled reference rearm study does not accept issued forecasts")
        self.family = family
        self.module = importlib.import_module("research." + family)

    def identity(self):
        root = Path(__file__).parent
        names = [
            "offensive_settled_reference_rearm_study.py",
            "offensive_settled_reference_rearm.py",
            "offensive_settled_reference_rearm_contract.json",
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
        return {
            "source": source_identity(),
            "family": self.family,
            "dependencies": {name: file_hash(root / name) for name in names},
        }

    def saved(self, market, path, parameters, benchmark=None, costs=1.0, delay=1):
        cfg = Config()
        owner = self.module.Owner(market, parameters)
        expected = {
            "config": asdict(cfg), "universe": list(market.symbols), "quality": market.quality,
            "data_sha256": market.fingerprint(), "source": source_identity(), "provenance": market.provenance,
            "delay": delay, "cost_multiplier": costs, "benchmark": benchmark,
            "start": str(market.calendar[0].date()), "end": str(market.calendar[-1].date()),
            "economic_acceptance": "UNVERIFIED", "accounting": "adjusted economic units, not actual shares",
            "study": self.identity(), "policy": owner.identity(),
        }
        intent_path = path.parent.parent / "intents" / (path.name + ".json")
        if path.exists():
            self.module.verify_trace(intent_path, expected)
            return load_result(path, expected=expected)
        result = run(
            market, cfg, benchmark=benchmark, policy_factory=lambda _m, _c: owner,
            cost_multiplier=costs, delay=delay,
        )
        result.metadata["study"] = self.identity()
        if result.metadata != expected:
            raise AssertionError("unexpected settled reference rearm study identity")
        save_result(result, path)
        self.module.preserve_trace(intent_path, expected, owner.trace)
        return result

    @staticmethod
    def _trace_summary(trace):
        base = [r for r in trace if r.get("kind") == "ALPHA_DECAY_DISPLACEMENT_EVENT"]
        rearm = [r for r in trace if r.get("kind") == "SETTLED_REFERENCE_REARM_EVENT"]
        return {
            "base_records": len(base),
            "security_exits": sum(r.get("action") == "SECURITY_EXIT" for r in base),
            "vacancy_fills": sum(r.get("action") == "VACANCY_FILL" for r in base),
            "displacements": sum(r.get("action") == "ALPHA_DECAY_DISPLACEMENT" for r in base),
            "invalidations": sum(r.get("action") == "ACUTE_REFERENCE_INVALIDATION" for r in rearm),
            "reference_rearms": sum(r.get("action") == "REFERENCE_ALPHA_REARM" for r in rearm),
            "settlement_blocks": sum(r.get("action") == "SETTLEMENT_REARM_BLOCKED" for r in rearm),
            "trend_edge_rearms": sum(r.get("action") == "TREND_EDGE_REARM" for r in rearm),
            "epoch_armed": sum(r.get("action") == "FRESH_EPOCH_ARMED" for r in rearm),
            "exit_retries": sum(r.get("action") == "INVALIDATED_EXIT_RETRY" for r in rearm),
        }

    def select(self, market, catalog, out):
        from research.ledger_attribution import attribute

        root = Path(__file__).parent
        contract = json.loads((root / "offensive_settled_reference_rearm_contract.json").read_text())
        full_sha = market.fingerprint()
        if full_sha != contract["frozen_inputs"]["full_sha256"]:
            raise ValueError("settled reference rearm requires the frozen full market")
        selection_end = contract["measurement"]["selection_window"]["end"]
        market = market.prefix(selection_end)
        scoped = scopes(market, catalog)
        grid = self.module.grid()
        out = Path(out)
        plan = {
            "identity": self.identity(), "full_data_sha256": full_sha, "data_sha256": market.fingerprint(),
            "selection_end": selection_end, "grid": [asdict(p) for p in grid], "scopes": scoped,
            "registration_commit": self.REGISTRATION_COMMIT,
        }
        if (out / "plan.json").exists() and json.loads((out / "plan.json").read_text()) != plan:
            raise ValueError("existing settled reference rearm plan is not equivalent")
        write_json(out / "plan.json", plan)
        rows = []
        for name, names in scoped.items():
            m = market.subset(names)
            row = {"scope": name}
            for enabled, label in ((False, "control"), (True, "treatment")):
                run_name = f"{name}_{label}"
                result = self.saved(m, out / "runs" / run_name, self.module.Parameters(enabled))
                days, episodes, audit = attribute(m, result)
                target = out / "attribution" / run_name
                target.mkdir(parents=True, exist_ok=True)
                days.to_csv(target / "pnl.csv", index=False, float_format="%.17g")
                episodes.to_csv(target / "episodes.csv", index=False, float_format="%.17g")
                row[label] = dict(metrics(result), average_exposure=float(result.equity.exposure.mean()))
                row[label + "_ledger"] = audit
                if enabled:
                    trace = json.loads((out / "intents" / f"{run_name}.json").read_text())["trace"]
                    row[self.family] = self._trace_summary(trace)
            rows.append(row)
            write_json(out / "paired-progress.json", rows)
            print(json.dumps(row), flush=True)
        decision = alpha_discovery_screen(rows)
        decision.update(
            status="PAIRED_SCREEN_ADVANCE" if decision["advance"] else "REJECTED_PAIRED_SCREEN",
            rows=rows, identity=self.identity(), data_sha256=market.fingerprint(), full_data_sha256=full_sha,
            new_accounts=6, candidate=int(decision["advance"]), plan_sha256=file_hash(out / "plan.json"),
            economic_acceptance="NOT_ESTABLISHED", historical_exposure="RETROSPECTIVE_NOT_UNSEEN_OUT_OF_SAMPLE",
        )
        write_json(out / "selection.json", decision)

    def evaluate(self, market, catalog, selection, out):
        selection = Path(selection)
        chosen = json.loads(selection.read_text())
        out = Path(out)
        if (
            chosen["identity"] != self.identity()
            or chosen["data_sha256"] != market.prefix("2025-12-31").fingerprint()
            or chosen["full_data_sha256"] != market.fingerprint()
            or chosen["plan_sha256"] != file_hash(selection.with_name("plan.json"))
        ):
            raise ValueError("settled reference rearm selection identity mismatch")
        if not chosen["advance"]:
            write_json(out.parent / "evaluation-decision.json", {
                "status": "NOT_RUN_REJECTED_PAIR", "selection_sha256": file_hash(selection),
                "treatment_2026_runs": 0,
            })
            return
        windows = {
            "full": (None, None),
            "bull": ("2023-01-03", "2026-06-30"),
            "late_june_through_august": ("2026-06-22", "2026-08-31"),
            "july_august": ("2026-07-01", "2026-08-31"),
            "retrospective_2026": ("2026-01-01", None),
        }
        write_json(out / "plan.json", {
            "identity": self.identity(), "data_sha256": market.fingerprint(),
            "selection_sha256": file_hash(selection), "parameters": asdict(self.module.Parameters(True)),
            "windows": windows,
        })
        rows = []
        for name, names in scopes(market, catalog).items():
            m = market.subset(names)
            for enabled, label in ((False, "control"), (True, "treatment")):
                result = self.saved(m, out / "runs" / f"{name}_{label}", self.module.Parameters(enabled))
                for window, (start, end) in windows.items():
                    rows.append({"scope": name, "policy": label, "window": window, **metrics(result, start, end)})
            pd.DataFrame(rows).to_csv(out / "matrix.csv", index=False, float_format="%.17g")
        write_json(out / "status.json", {
            "identity": self.identity(), "runs": 6, "rows": len(rows),
            "status": "MEASURED_NOT_ORIGINAL_ACCEPTANCE", "final_matrix": "NOT_RUN",
        })

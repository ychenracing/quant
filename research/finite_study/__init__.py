"""Compatibility overlay for the latest fixed paired studies.

All existing families delegate byte-for-byte to the historical finite_study.py.
Only explicitly preregistered paired families use the overlay below.
"""
from __future__ import annotations
from dataclasses import asdict
import importlib.util
import importlib
import json
from pathlib import Path
import sys
import pandas as pd

_here = Path(__file__).resolve()
_base_path = _here.parent.parent / "finite_study.py"
_spec = importlib.util.spec_from_file_location("research._finite_study_base", _base_path)
if _spec is None or _spec.loader is None:
    raise ImportError("cannot load finite_study base")
_base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _base
_spec.loader.exec_module(_base)

from techquant.config import Config
from techquant.data import file_hash
from techquant.engine import run
from techquant.evidence import load_result, metrics, save_result, source_identity
from research.expectation_study import write_json, scopes


_PAIRED = {"confirmed_shock", "profit_trend_shield", "shock_reclaim"}


class Study(_base.Study):
    def __init__(self, family: str, *, issued_evidence: Path | None = None):
        if family not in _PAIRED:
            super().__init__(family, issued_evidence=issued_evidence)
            return
        if issued_evidence is not None:
            raise ValueError(f"{family} does not accept issued forecasts")
        self.family = family
        self.module = importlib.import_module("research." + family)
        self.issued = None

    def identity(self):
        if self.family not in _PAIRED:
            return super().identity()
        root = _here.parent.parent
        names = [
            "finite_study.py", "finite_study/__init__.py", "expectation_study.py",
            self.family + ".py", self.family + "_contract.json", "leadership.py",
            "expectation.py", "quantity_obligation.py", "quantity_obligation_contract.json",
            "support_budget.py", "support_budget_contract.json", "funded_risk.py",
            "funded_risk_contract.json", "observed_readiness.py",
            "observed_readiness_contract.json", "admission_budget_completion.py",
            "admission_budget_completion_contract.json", "observed_admission_completion.py",
            "observed_admission_completion_contract.json", "decision_review.py",
            "ledger_attribution.py",
        ]
        return {
            "source": source_identity(), "family": self.family,
            "dependencies": {name: file_hash(root / name) for name in names},
        }

    def saved(self, market, path, parameters=None, benchmark=None, costs=1., delay=1, *, configuration=None):
        if self.family not in _PAIRED:
            return super().saved(market, path, parameters, benchmark, costs, delay, configuration=configuration)
        if configuration is not None:
            raise ValueError(f"{self.family} uses the unchanged production Config")
        cfg = Config()
        expected = {
            "config": asdict(cfg), "universe": list(market.symbols), "quality": market.quality,
            "data_sha256": market.fingerprint(), "source": source_identity(),
            "provenance": market.provenance, "delay": delay, "cost_multiplier": costs,
            "benchmark": benchmark, "start": str(market.calendar[0].date()),
            "end": str(market.calendar[-1].date()), "economic_acceptance": "UNVERIFIED",
            "accounting": "adjusted economic units, not actual shares", "study": self.identity(),
        }
        owner = self.module.Owner(market, parameters) if parameters is not None else None
        if owner is not None:
            expected["policy"] = owner.identity()
        intent_path = path.parent.parent / "intents" / (path.name + ".json")
        if path.exists():
            if owner is not None:
                self.module.verify_trace(intent_path, expected)
            return load_result(path, expected=expected)
        factory = (lambda m, c: owner) if owner is not None else None
        result = run(market, cfg, benchmark=benchmark, policy_factory=factory,
                     cost_multiplier=costs, delay=delay)
        result.metadata["study"] = self.identity()
        if result.metadata != expected:
            raise AssertionError(f"unexpected {self.family} study identity")
        save_result(result, path)
        if owner is not None:
            self.module.preserve_trace(intent_path, expected, owner.trace)
        return result

    def select(self, market, catalog, out):
        if self.family not in _PAIRED:
            return super().select(market, catalog, out)
        from research.decision_review import paired_screen
        from research.ledger_attribution import attribute
        contract = json.loads((_here.parent.parent / (self.family + "_contract.json")).read_text())
        if self.family == "confirmed_shock":
            full_data_sha = contract["data"]["full_sha256"]
            selection_end = contract["data"]["selection_end"]
            registration_commit = "1ce819815f7c6a5cd6e9f00e64cf972d183e1888"
        else:
            full_data_sha = contract["frozen_inputs"]["sha256"]
            selection_end = contract["measurement"]["window"]["end"]
            registration_commit = ("39b7e7aa3ec63f9ba29d2cfd034facd9c35659bd" if self.family == "shock_reclaim" else "66354029843c1704808fa8717b804c745389e302")
        if market.fingerprint() != full_data_sha:
            raise ValueError(f"{self.family} requires the frozen full market")
        full_sha = market.fingerprint()
        market = market.prefix(selection_end)
        scoped = scopes(market, catalog)
        grid = self.module.grid()
        out = Path(out)
        plan = {
            "identity": self.identity(), "full_data_sha256": full_sha,
            "data_sha256": market.fingerprint(),
            "selection_end": selection_end,
            "grid": [asdict(p) for p in grid], "scopes": scoped,
            "registration_commit": registration_commit,
        }
        if (out / "plan.json").exists() and json.loads((out / "plan.json").read_text()) != plan:
            raise ValueError(f"existing {self.family} plan is not equivalent")
        write_json(out / "plan.json", plan)
        rows = []
        for name, names in scoped.items():
            m = market.subset(names)
            row = {"scope": name}
            for enabled, label in ((False, "control"), (True, "treatment")):
                run_name = name + "_" + label
                result = self.saved(m, out / "runs" / run_name, self.module.Parameters(enabled))
                days, episodes, audit = attribute(m, result)
                target = out / "attribution" / run_name
                target.mkdir(parents=True, exist_ok=True)
                days.to_csv(target / "pnl.csv", index=False, float_format="%.17g")
                episodes.to_csv(target / "episodes.csv", index=False, float_format="%.17g")
                row[label] = dict(metrics(result), average_exposure=float(result.equity.exposure.mean()))
                row[label + "_ledger"] = audit
                if enabled:
                    trace = json.loads((out / "intents" / (run_name + ".json")).read_text())["trace"]
                    if self.family == "confirmed_shock":
                        state = [r for r in trace if r.get("kind") == "CONFIRMED_MARKET_SHOCK_STATE"]
                        row["confirmed_shock"] = {
                            "state_records": len(state),
                            "first_half_risk": sum(r.get("after_cap") == .5 and r.get("after_pending") for r in state),
                            "zero_risk": sum(r.get("after_cap") == 0 for r in state),
                            "account_drawdown_zero": sum(
                                "PORTFOLIO_DRAWDOWN_SHOCK" in r.get("decision_reason", "")
                                and r.get("after_cap") == 0 for r in state),
                        }
                    elif self.family == "profit_trend_shield":
                        state = [r for r in trace if r.get("kind") == "PROFIT_TREND_SHIELD"]
                        row["profit_trend_shield"] = {"events": len(state), "retained_symbols": sum(len(r["symbols"]) for r in state)}
                    else:
                        state = [r for r in trace if r.get("kind") == "SHOCK_RECLAIM_PERMISSION"]
                        row["shock_reclaim"] = {"events": len(state), "released": sum(len(r["released"]) for r in state), "requested": sum(len(r["requested"]) for r in state)}
            rows.append(row)
            write_json(out / "paired-progress.json", rows)
            print(json.dumps(row), flush=True)
        if self.family == "confirmed_shock":
            decision = paired_screen(rows)
        else:
            nonregression = all(
                row["treatment"]["wealth"] >= row["control"]["wealth"] - 1e-12
                for row in rows
            )
            strict = any(
                row["treatment"]["wealth"] > row["control"]["wealth"] + 1e-12
                for row in rows
            )
            decision = {
                "advance": nonregression and strict,
                "wealth_nonregression_all_scopes": nonregression,
                "strict_wealth_improvement": strict,
            }
        decision.update(
            status="PAIRED_SCREEN_ADVANCE" if decision["advance"] else "REJECTED_PAIRED_SCREEN",
            rows=rows, identity=self.identity(), data_sha256=market.fingerprint(),
            full_data_sha256=full_sha, new_accounts=6, candidate=int(decision["advance"]),
            plan_sha256=file_hash(out / "plan.json"), economic_acceptance="NOT_ESTABLISHED",
            historical_exposure="RETROSPECTIVE_NOT_UNSEEN_OUT_OF_SAMPLE",
        )
        write_json(out / "selection.json", decision)

    def evaluate(self, market, catalog, selection, out):
        if self.family not in _PAIRED:
            return super().evaluate(market, catalog, selection, out)
        chosen = json.loads(selection.read_text())
        out = Path(out)
        if (chosen["identity"] != self.identity()
                or chosen["data_sha256"] != market.prefix("2025-12-31").fingerprint()
                or chosen["full_data_sha256"] != market.fingerprint()
                or chosen["plan_sha256"] != file_hash(selection.with_name("plan.json"))):
            raise ValueError(f"{self.family} selection identity mismatch")
        if not chosen["advance"]:
            write_json(out.parent / "evaluation-decision.json", {
                "status": "NOT_RUN_REJECTED_PAIR",
                "selection_sha256": file_hash(selection), "treatment_2026_runs": 0,
            })
            return
        windows = {
            "full": (None, None), "bull": ("2023-01-03", "2026-06-30"),
            "late_june_through_august": ("2026-06-22", "2026-08-31"),
            "july_august": ("2026-07-01", "2026-08-31"),
            "retrospective_2026": ("2026-01-01", None),
        }
        write_json(out / "plan.json", {
            "identity": self.identity(), "data_sha256": market.fingerprint(),
            "selection_sha256": file_hash(selection),
            "parameters": asdict(self.module.Parameters(True)), "windows": windows,
        })
        rows = []
        for name, names in scopes(market, catalog).items():
            m = market.subset(names)
            for enabled, label in ((False, "control"), (True, "treatment")):
                result = self.saved(m, out / "runs" / (name + "_" + label), self.module.Parameters(enabled))
                for window, (start, end) in windows.items():
                    rows.append({"scope": name, "policy": label, "window": window,
                                 **metrics(result, start, end)})
            pd.DataFrame(rows).to_csv(out / "matrix.csv", index=False, float_format="%.17g")
        write_json(out / "status.json", {
            "identity": self.identity(), "runs": 6, "rows": len(rows),
            "status": "MEASURED_NOT_ORIGINAL_ACCEPTANCE", "final_matrix": "NOT_RUN",
        })


__all__ = ["Study"]

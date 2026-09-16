"""Compatibility overlay for the latest fixed paired studies.

All existing families delegate byte-for-byte to the historical finite_study.py.
Only explicitly preregistered paired families use the overlay below.
"""
from __future__ import annotations
from dataclasses import asdict
import importlib.util
import importlib
import json
import math
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


_PAIRED = {
    "confirmed_shock", "profit_trend_shield", "shock_reclaim",
    "theme_campaign", "leader_anchor_slots", "offensive_core", "committed_offensive_core",
    "offensive_alpha_decay_displacement", "offensive_native_ownership", "offensive_trend_quality",
}


def alpha_discovery_screen(rows):
    if {row["scope"] for row in rows} != {
        "union", "chatgpt_5", "joint_optical_leader_removal"
    }:
        raise ValueError("alpha discovery requires the three fixed scopes")
    control = [float(row["control"]["wealth"]) for row in rows]
    treatment = [float(row["treatment"]["wealth"]) for row in rows]
    if any(value <= 0 for value in control + treatment):
        raise ValueError("terminal wealth must stay positive for log objective")
    control_mean_log = math.fsum(math.log(value) for value in control) / len(control)
    treatment_mean_log = math.fsum(math.log(value) for value in treatment) / len(treatment)
    control_min = min(control)
    treatment_min = min(treatment)
    removal = next(row for row in rows if row["scope"] == "joint_optical_leader_removal")
    broad_improvement = any(
        row["treatment"]["wealth"] > row["control"]["wealth"] + 1e-12
        for row in rows if row["scope"] in {"union", "joint_optical_leader_removal"}
    )
    mean_improvement = treatment_mean_log > control_mean_log + 1e-12
    minimum_improvement = treatment_min > control_min + 1e-12
    removal_positive = float(removal["treatment"]["wealth"]) > 1.0
    return {
        "advance": mean_improvement and minimum_improvement and broad_improvement and removal_positive,
        "control_mean_log_wealth": control_mean_log,
        "treatment_mean_log_wealth": treatment_mean_log,
        "mean_log_wealth_improved": mean_improvement,
        "control_min_wealth": control_min,
        "treatment_min_wealth": treatment_min,
        "minimum_wealth_improved": minimum_improvement,
        "broad_scope_improvement": broad_improvement,
        "leader_removal_positive_wealth": removal_positive,
    }


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
        if self.family in {"theme_campaign", "leader_anchor_slots", "offensive_core", "committed_offensive_core", "offensive_alpha_decay_displacement", "offensive_native_ownership", "offensive_trend_quality"}:
            names += [
                "trend_book.py", "trend_book_contract.json", "coherent.py",
                "coherent_contract.json", "observed_trend.py",
                "observed_trend_contract.json", "nonlinear.py",
                "nonlinear_contract.json", "pathwise.py", "pathwise_contract.json",
            ]
        if self.family == "offensive_trend_quality":
            names += [
                "offensive_alpha_decay_displacement.py",
                "offensive_alpha_decay_displacement_contract.json",
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
        elif self.family == "offensive_core":
            full_data_sha = contract["frozen_inputs"]["sha256"]
            selection_end = contract["measurement"]["window"]["end"]
            registration_commit = "d8f34acb66313023a1166fa6e0f283bf4580e7d0"
        elif self.family == "committed_offensive_core":
            full_data_sha = contract["frozen_inputs"]["sha256"]
            selection_end = contract["measurement"]["window"]["end"]
            registration_commit = "df9b79a18d99cd5ab2ee31560eb29b3c4474137d"
        elif self.family == "offensive_alpha_decay_displacement":
            full_data_sha = contract["frozen_inputs"]["full_sha256"]
            selection_end = contract["measurement"]["window"]["end"]
            registration_commit = "d749786da34b4b26f5e1ecaec3b84e6d80718d6c"
        elif self.family == "offensive_native_ownership":
            full_data_sha = contract["frozen_inputs"]["full_sha256"]
            selection_end = contract["measurement"]["window"]["end"]
            registration_commit = "89b7a4f963d5dd6bf9b0cd37bc839949ec45bb56"
        elif self.family == "offensive_trend_quality":
            full_data_sha = contract["frozen_inputs"]["full_sha256"]
            selection_end = contract["measurement"]["window"]["end"]
            registration_commit = "937825546ad587f19bc3446bad4687fd7b89fde5"
        elif self.family in {"theme_campaign", "leader_anchor_slots"}:
            full_data_sha = contract["data_sha256"]
            selection_end = "2025-12-31"
            registration_commit = (
                "38136909efb4124ec3ca02822e26c012debced59"
                if self.family == "theme_campaign"
                else "fbab56360b555bc69c9a771b4b27c5db1d603c61"
            )
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
                    elif self.family == "theme_campaign":
                        state = [r for r in trace if r.get("kind") == "THEME_CAMPAIGN_STATE"]
                        row["theme_campaign"] = {
                            "records": len(state),
                            "transitions": sum(r.get("previous") != r.get("active") for r in state),
                            "suppressed": sum(len(r.get("suppressed", [])) for r in state),
                        }
                    elif self.family == "leader_anchor_slots":
                        state = [r for r in trace if r.get("kind") == "LEADER_ANCHOR_SLOTS"]
                        row["leader_anchor_slots"] = {
                            "records": len(state),
                            "sector_companion": sum(
                                r.get("mode") == "SECTOR_COMPANION" for r in state
                            ),
                            "parent_fallback": sum(
                                r.get("mode") == "PARENT_FALLBACK" for r in state
                            ),
                            "suppressed": sum(len(r.get("suppressed", [])) for r in state),
                        }
                    elif self.family == "offensive_core":
                        state = [r for r in trace if r.get("kind") == "OFFENSIVE_CORE_REVIEW"]
                        row["offensive_core"] = {
                            "records": len(state),
                            "full_exposure_selections": sum(
                                r.get("action") == "FULL_EXPOSURE_SELECTION" for r in state
                            ),
                            "cash_no_eligible": sum(
                                r.get("action") == "CASH_NO_ELIGIBLE_NAME" for r in state
                            ),
                        }
                    elif self.family == "committed_offensive_core":
                        state = [r for r in trace if r.get("kind") == "COMMITTED_OFFENSIVE_EVENT"]
                        row["committed_offensive_core"] = {
                            "records": len(state),
                            "security_exits": sum(r.get("action") == "SECURITY_EXIT" for r in state),
                            "vacancy_fills": sum(r.get("action") == "VACANCY_FILL" for r in state),
                        }
                    elif self.family == "offensive_alpha_decay_displacement":
                        state = [r for r in trace if r.get("kind") == "ALPHA_DECAY_DISPLACEMENT_EVENT"]
                        row["offensive_alpha_decay_displacement"] = {
                            "records": len(state),
                            "security_exits": sum(r.get("action") == "SECURITY_EXIT" for r in state),
                            "vacancy_fills": sum(r.get("action") == "VACANCY_FILL" for r in state),
                            "displacements": sum(r.get("action") == "ALPHA_DECAY_DISPLACEMENT" for r in state),
                            "retirement_releases": sum(r.get("action") == "RETIREMENT_RELEASE" for r in state),
                        }
                    elif self.family == "offensive_native_ownership":
                        state = [r for r in trace if r.get("kind") == "OFFENSIVE_NATIVE_OWNERSHIP_EVENT"]
                        row["offensive_native_ownership"] = {
                            "records": len(state),
                            "native_replacements": sum(r.get("action") == "NATIVE_REPLACEMENT" for r in state),
                            "security_or_inventory_reductions": sum(
                                r.get("action") == "SECURITY_OR_INVENTORY_REDUCTION" for r in state
                            ),
                            "full_cap_records": sum(r.get("account_cap") == 1.0 for r in state),
                        }
                    elif self.family == "offensive_trend_quality":
                        state = [r for r in trace if r.get("kind") == "ALPHA_DECAY_DISPLACEMENT_EVENT"]
                        row["offensive_trend_quality"] = {
                            "records": len(state),
                            "security_exits": sum(r.get("action") == "SECURITY_EXIT" for r in state),
                            "vacancy_fills": sum(r.get("action") == "VACANCY_FILL" for r in state),
                            "displacements": sum(r.get("action") == "ALPHA_DECAY_DISPLACEMENT" for r in state),
                            "retirement_releases": sum(r.get("action") == "RETIREMENT_RELEASE" for r in state),
                        }
                    else:
                        state = [r for r in trace if r.get("kind") == "SHOCK_RECLAIM_PERMISSION"]
                        row["shock_reclaim"] = {"events": len(state), "released": sum(len(r["released"]) for r in state), "requested": sum(len(r["requested"]) for r in state)}
            rows.append(row)
            write_json(out / "paired-progress.json", rows)
            print(json.dumps(row), flush=True)
        if self.family == "confirmed_shock":
            decision = paired_screen(rows)
        elif self.family in {"offensive_alpha_decay_displacement", "offensive_native_ownership", "offensive_trend_quality"}:
            decision = alpha_discovery_screen(rows)
        else:
            nonregression = all(
                row["treatment"]["wealth"] >= row["control"]["wealth"] - 1e-12
                for row in rows
            )
            strict = any(
                row["treatment"]["wealth"] > row["control"]["wealth"] + 1e-12
                for row in rows
            )
            union_strict = any(
                row["scope"] == "union"
                and row["treatment"]["wealth"] > row["control"]["wealth"] + 1e-12
                for row in rows
            )
            decision = {
                "advance": nonregression and strict
                           and (union_strict if self.family in {
                               "theme_campaign", "leader_anchor_slots"
                           } else True),
                "wealth_nonregression_all_scopes": nonregression,
                "strict_wealth_improvement": strict,
            }
            if self.family in {"theme_campaign", "leader_anchor_slots"}:
                decision["strict_union_improvement"] = union_strict
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


__all__ = ["Study", "alpha_discovery_screen"]

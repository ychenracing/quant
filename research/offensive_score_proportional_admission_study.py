"""Source-bound wealth study for the preregistered score-proportional admission rule."""
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
from research.offensive_campaign_peak_authority import Owner as ChampionOwner, Parameters as ChampionParameters


class Study:
    FAMILY = "offensive_score_proportional_admission"
    REGISTRATION_COMMIT = "51b43854ede28467cd07e7854aa4893a95025217"

    def __init__(self, family: str, *, issued_evidence=None):
        if family != self.FAMILY:
            raise ValueError("score proportional admission study accepts only its preregistered family")
        if issued_evidence is not None:
            raise ValueError("score proportional admission study does not accept issued forecasts")
        self.family = family
        self.module = importlib.import_module("research." + family)

    def identity(self):
        root=Path(__file__).parent
        names=["offensive_score_proportional_admission_study.py","offensive_score_proportional_admission.py","offensive_score_proportional_admission_contract.json","offensive_campaign_peak_authority.py","offensive_campaign_peak_authority_contract.json","offensive_fresh_challenger_authority.py","offensive_fresh_challenger_authority_contract.json","offensive_alpha_decay_displacement.py","offensive_alpha_decay_displacement_contract.json","finite_study/__init__.py","trend_book.py","trend_book_contract.json","coherent.py","coherent_contract.json","observed_trend.py","observed_trend_contract.json","nonlinear.py","nonlinear_contract.json","pathwise.py","pathwise_contract.json","expectation_study.py","leadership.py","expectation.py","quantity_obligation.py","quantity_obligation_contract.json","support_budget.py","support_budget_contract.json","funded_risk.py","funded_risk_contract.json","observed_readiness.py","observed_readiness_contract.json","admission_budget_completion.py","admission_budget_completion_contract.json","observed_admission_completion.py","observed_admission_completion_contract.json","decision_review.py","ledger_attribution.py"]
        return {"source":source_identity(),"family":self.family,"dependencies":{n:file_hash(root/n) for n in names}}

    def _saved(self,market,path,owner):
        cfg=Config(); expected={"config":asdict(cfg),"universe":list(market.symbols),"quality":market.quality,"data_sha256":market.fingerprint(),"source":source_identity(),"provenance":market.provenance,"delay":1,"cost_multiplier":1.0,"benchmark":None,"start":str(market.calendar[0].date()),"end":str(market.calendar[-1].date()),"economic_acceptance":"UNVERIFIED","accounting":"adjusted economic units, not actual shares","study":self.identity(),"policy":owner.identity()}
        if path.exists(): return load_result(path,expected=expected)
        result=run(market,cfg,policy_factory=lambda _m,_c:owner); result.metadata["study"]=self.identity()
        if result.metadata!=expected: raise AssertionError("unexpected score proportional admission study identity")
        save_result(result,path); return result

    @staticmethod
    def _objective(rows,label):
        wealth=[float(r[label]["wealth"]) for r in rows]; return {"mean_log":math.fsum(math.log(v) for v in wealth)/len(wealth),"minimum":min(wealth),"wealth":{r["scope"]:float(r[label]["wealth"]) for r in rows}}

    def select(self,market,catalog,out):
        root=Path(__file__).parent; full_sha=market.fingerprint()
        if full_sha!="d9ded1d933f292933aa09145bc36a5696421c43dbf266fce018a34ef0b9f3f9b": raise ValueError("score proportional admission requires frozen full market")
        market=market.prefix("2025-12-31"); scoped=scopes(market,catalog); out=Path(out); write_json(out/"plan.json",{"identity":self.identity(),"full_data_sha256":full_sha,"data_sha256":market.fingerprint(),"selection_end":"2025-12-31","scopes":scoped,"registration_commit":self.REGISTRATION_COMMIT,"contract_sha256":file_hash(root/"offensive_score_proportional_admission_contract.json")})
        rows=[]
        for name,names in scoped.items():
            subset=market.subset(names); champion=ChampionOwner(subset,ChampionParameters(True)); treatment=self.module.Owner(subset,self.module.Parameters(True)); row={"scope":name}
            for label,owner in (("current_champion",champion),("treatment",treatment)):
                result=self._saved(subset,out/"runs"/f"{name}_{label}",owner); row[label]=dict(metrics(result),average_exposure=float(result.equity.exposure.mean()))
            events=[r for r in treatment.trace if r.get("kind")=="SCORE_PROPORTIONAL_ADMISSION_EVENT" and r.get("action")=="SCORE_PROPORTIONAL_ADMISSION"]
            row["allocation"]={"multi_fill_events":len(events),"events":events}; rows.append(row); write_json(out/"paired-progress.json",rows); print(json.dumps(row),flush=True)
        champion_obj=self._objective(rows,"current_champion"); treatment_obj=self._objective(rows,"treatment"); advance=treatment_obj["mean_log"]>champion_obj["mean_log"]+1e-12 and treatment_obj["minimum"]>=champion_obj["minimum"]-1e-12
        write_json(out/"selection.json",{"status":"BEATS_CURRENT_ALPHA_CHAMPION" if advance else "REJECTED_VS_CURRENT_ALPHA_CHAMPION","advance":advance,"rows":rows,"current_champion_objective":champion_obj,"treatment_objective":treatment_obj,"identity":self.identity(),"data_sha256":market.fingerprint(),"full_data_sha256":full_sha,"plan_sha256":file_hash(out/"plan.json"),"registration_commit":self.REGISTRATION_COMMIT,"economic_acceptance":"NOT_ESTABLISHED","historical_exposure":"RETROSPECTIVE_NOT_UNSEEN_OUT_OF_SAMPLE"})

    def evaluate(self,market,catalog,selection,out):
        selection=Path(selection); chosen=json.loads(selection.read_text()); out=Path(out)
        if chosen["identity"]!=self.identity() or chosen["full_data_sha256"]!=market.fingerprint(): raise ValueError("score proportional selection identity mismatch")
        if not chosen["advance"]: write_json(out.parent/"evaluation-decision.json",{"status":"NOT_RUN_REJECTED_VS_CURRENT_CHAMPION","selection_sha256":file_hash(selection),"full_history_runs":0}); return
        windows={"full":(None,None),"bull":("2023-01-03","2026-06-30"),"late_june_through_august":("2026-06-22","2026-08-31"),"july_august":("2026-07-01","2026-08-31"),"retrospective_2026":("2026-01-01",None)}; rows=[]
        for name,names in scopes(market,catalog).items():
            subset=market.subset(names)
            for label,owner in (("current_champion",ChampionOwner(subset,ChampionParameters(True))),("treatment",self.module.Owner(subset,self.module.Parameters(True)))):
                result=self._saved(subset,out/"runs"/f"{name}_{label}",owner)
                for window,(start,end) in windows.items(): rows.append({"scope":name,"policy":label,"window":window,**metrics(result,start,end)})
            pd.DataFrame(rows).to_csv(out/"matrix.csv",index=False,float_format="%.17g")
        write_json(out/"status.json",{"identity":self.identity(),"runs":6,"rows":len(rows),"status":"MEASURED_NOT_FINAL_ACCEPTANCE","formal_matrix":"NOT_RUN"})

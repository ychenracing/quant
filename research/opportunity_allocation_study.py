"""One immutable paired experiment, using the existing engine and evidence store."""
from dataclasses import asdict
import json
from pathlib import Path
from techquant.data import load_market, file_hash
from techquant.evidence import metrics
from research.expectation_study import scopes, write_json
from research.finite_study import Study
from research.decision_review import paired_screen
from research.ledger_attribution import attribute


def paired(output, data, supplement):
    output=Path(output);root=Path(__file__).parent
    catalog=json.loads((root/'catalog.json').read_text())
    contract=json.loads((root/'opportunity_allocation_contract.json').read_text())
    market=load_market(data,supplement=supplement,sectors=catalog['sectors'])
    if market.fingerprint()!=contract['data']['full_sha256']:
        raise ValueError('only the original frozen market is admissible')
    market=market.prefix(contract['data']['selection_end'])
    study=Study('opportunity_allocation');scoped=scopes(market,catalog)
    path=output/'selection';path.mkdir(parents=True,exist_ok=True)
    plan=dict(identity=study.identity(),data_sha256=market.fingerprint(),scopes=scoped,
              parameters=[asdict(p) for p in study.module.grid()],
              contract_commit='80c0252aa3f1bb33fb4b158209521f14aed43dcf',
              contract_sha256=file_hash(root/'opportunity_allocation_contract.json'))
    if (path/'plan.json').exists() and json.loads((path/'plan.json').read_text())!=plan:
        raise ValueError('existing paired plan is not identity-equivalent')
    write_json(path/'plan.json',plan)
    rows=[]
    for scope,names in scoped.items():
        row=dict(scope=scope)
        for enabled,label in ((False,'control'),(True,'treatment')):
            result=study.saved(market.subset(names),path/'runs'/f'{scope}_{label}',study.module.Parameters(enabled))
            days,episodes,reconciliation=attribute(market.subset(names),result)
            destination=path/'attribution'/f'{scope}_{label}';destination.mkdir(parents=True,exist_ok=True)
            days.to_csv(destination/'pnl.csv',index=False,float_format='%.17g')
            episodes.to_csv(destination/'episodes.csv',index=False,float_format='%.17g')
            row[label]=dict(metrics(result),average_exposure=float(result.equity.exposure.mean()))
            row[label+'_ledger']=reconciliation
        rows.append(row);write_json(path/'paired-progress.json',rows)
        print(json.dumps(row),flush=True)
    decision=paired_screen(rows)
    decision.update(rows=rows,identity=study.identity(),data_sha256=market.fingerprint(),
                    candidate=1 if decision['advance'] else 0,new_accounts=6,
                    full_evaluation='NOT_RUN: original final validation requires a justified complete candidate',
                    historical_exposure='RETROSPECTIVE_NOT_UNSEEN_OUT_OF_SAMPLE',
                    research_budget='renewed continuation: hypothesis opportunity_allocation; no neighboring retries')
    write_json(path/'selection.json',decision)
    return decision


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('output','data','supplement'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();paired(args.output,args.data,args.supplement)

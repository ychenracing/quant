"""Reconcile recorded accounts before choosing a new economic hypothesis.

No policy parameters, targets, orders or fills are changed here. The log-gap
partition is an accounting identity between TWO actually simulated accounts,
not an investable counterfactual or a feasibility bound.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
from techquant.data import file_hash, load_market
from techquant.evidence import load_result, metrics, source_identity
from techquant.policy import CloseObservation
from research.ledger_attribution import attribute


def decompose(candidate: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Reference minus candidate: participation, held-name mix, execution.

    The reference's *realized invested-book return* defines the allocation
    attribution convention. At zero reference opening exposure the carry
    difference is allocated to held-name mix. Divided differences of log(1+r)
    make the daily partition exactly additive to log(reference wealth/candidate
    wealth), including offsetting components when total daily returns coincide.
    This convention does NOT say that buying the reference holdings was feasible.
    """
    columns = ['opening_exposure', 'carry_rate', 'execution_rate']
    if not candidate.index.equals(reference.index):
        raise ValueError('recorded account dates must match exactly')
    arrays = [x[columns].to_numpy(dtype=float) for x in (candidate, reference)]
    for x in arrays:
        if not np.isfinite(x).all() or (x[:, 0] < -1e-12).any():
            raise ValueError('nonfinite attribution or negative opening exposure')
    c, b = arrays
    cr, br = c[:, 1:].sum(axis=1), b[:, 1:].sum(axis=1)
    if (cr <= -1).any() or (br <= -1).any():
        raise ValueError('log attribution requires positive NAV')
    invested = np.divide(b[:, 1], b[:, 0], out=np.zeros(len(b)), where=b[:, 0] > 1e-12)
    participation = np.where(b[:, 0] > 1e-12, (b[:, 0]-c[:, 0])*invested, 0.)
    selection = b[:, 1]-c[:, 1]-participation
    execution = b[:, 2]-c[:, 2]
    delta = br-cr
    total_log = np.log1p(br)-np.log1p(cr)
    multiplier = np.divide(total_log, delta, out=1/(1+cr), where=np.abs(delta) > 1e-12)
    values = np.column_stack([participation, selection, execution])*multiplier[:, None]
    if not np.allclose(values.sum(axis=1), total_log, rtol=1e-9, atol=1e-12):
        raise ValueError('attribution does not reconcile to daily relative wealth')
    return pd.DataFrame(np.column_stack([values,total_log]), index=candidate.index,
                        columns=['participation_log','selection_log','execution_log','total_log'])


def recorded_account(market, path: Path):
    identity = json.loads((path/'identity.json').read_text())
    result = load_result(path, expected=identity)
    m = market.subset(identity['universe']).prefix(identity['end'])
    if str(m.calendar[0].date()) != identity['start']:
        raise ValueError('review does not silently skip account initialization')
    days, episodes, reconciliation = attribute(m, result)
    days['date'] = pd.to_datetime(days['date'])
    previous_prices = m.panel('close').ffill().shift().fillna(0.)
    lookup = previous_prices.stack().to_dict()
    days['opening_value'] = [q*lookup[(d,s)] for d,s,q in zip(days.date, days.symbol, days.units_before, strict=True)]
    totals = days.groupby('date')[['carry','execution','opening_value']].sum().reindex(result.equity.index,fill_value=0.)
    previous_nav = result.equity.nav.shift().fillna(identity['config']['initial_cash'])
    daily = pd.DataFrame({'opening_exposure':totals.opening_value/previous_nav,
                          'carry_rate':totals['carry']/previous_nav,
                          'execution_rate':totals.execution/previous_nav})
    error = (daily.carry_rate+daily.execution_rate-(result.equity.nav/previous_nav-1)).abs().max()
    if error > 1e-10:
        raise ValueError('attributed rates do not equal recorded account returns')
    reconciliation['maximum_daily_return_error'] = float(error)
    return result, m, days, episodes, daily, reconciliation


def decision_matches(weight_error, observed_cap, recorded_cap, observed_reason, recorded_reason):
    """Same original tolerances; a text/cap mismatch is never a PASS."""
    return weight_error <= 1e-13 and observed_cap == recorded_cap and observed_reason == recorded_reason


def observe_same_decisions(market, result):
    """Replay the SAME policy against recorded fills, never rematch any order."""
    from research.admission_structure import Owner, Parameters
    owner = Owner(market, Parameters(False))
    by_day = defaultdict(list)
    for o in result.orders:
        if o['status'] == 'FILLED':
            by_day[o['date']].append(o)
    units = np.zeros(len(market.symbols))
    index = {s:j for j,s in enumerate(market.symbols)}
    rows, error, mismatches = [], 0., []
    for i,date in enumerate(market.calendar):
        day = str(date.date())
        for fill in by_day[day]:
            j = index[fill['symbol']]
            units[j] += float(fill['units'])*(1 if fill['side']=='BUY' else -1)
            if abs(units[j]) < 1e-7:
                units[j] = 0.
        equity = result.equity.loc[date]
        p = owner.inner
        price = np.nan_to_num(p.features.close[i],nan=0.)
        o = CloseObservation.from_inventory(i,day,float(equity.nav),float(equity.cash),units,units*price/equity.nav)
        decision = owner.decide(o)
        gap = float(np.max(np.abs(decision.weights-result.targets.loc[date].to_numpy())))
        error = max(error,gap)
        if not decision_matches(gap, decision.cap, equity.target_cap, decision.reason, equity.reason):
            mismatches.append(dict(date=day,maximum_weight_error=gap,
                observed_cap=float(decision.cap),recorded_cap=float(equity.target_cap),
                observed_reason=decision.reason,recorded_reason=equity.reason,
                actual_units=units.tolist(),observed_targets=decision.weights.tolist(),
                recorded_targets=result.targets.loc[date].to_list()))
        held = units > 1e-10
        allowed = (p.ready[i] & p.features.entry[i] & ~p.features.exit[i]
                   & np.isfinite(p.features.score[i]) & (p.features.score[i] > 0))
        price_stop = held & (price <= p.stop)
        feature_exit = held & p.features.exit[i]
        stale = held & ~p.ready[i]
        cuts = decision.unit_targets < units-1e-10
        local_exit = price_stop | feature_exit | stale
        weights = units*price/equity.nav
        distance = np.maximum(price-p.stop,.02*price)
        risk_book = float(units@distance)
        risk_limit = float(p._risk_limit(o,p.risk.cap))
        rows.append(dict(date=day,nav=float(equity.nav),cash=float(equity.cash),
            actual_exposure=float(weights.sum()),risk_cap=p.risk.cap,
            risk_budget=risk_limit,actual_stop_risk=risk_book,
            held_names=int(held.sum()),qualified_names=int(allowed.sum()),
            readmission_blocked_names=int((allowed & p.readmit & ~held).sum()),
            free_qualified_names=int((allowed & ~held & ~p.readmit).sum()),
            stop_exit_names=int(price_stop.sum()),feature_exit_names=int(feature_exit.sum()),
            stale_exit_names=int(stale.sum()),both_stop_feature_names=int((price_stop & feature_exit).sum()),
            cut_names=int(cuts.sum()),cut_value=float(np.maximum(units-decision.unit_targets,0)@price),
            nonlocal_cut_names=int((cuts & ~local_exit).sum()),
            held_qualified_names=int((held&allowed).sum()),
            held_but_no_add_breakout=int((held&allowed&~p.breakout[i]).sum()),
            reason=decision.reason))
    return pd.DataFrame(rows), {'observed_closes':len(rows),'maximum_target_error':error,
        'exact_reason_and_cap':not bool(mismatches),'counterfactual':False,
        'status':'STRICT_MATCH' if not mismatches else 'STRICT_IDENTITY_NOT_EQUIVALENT',
        'mismatches':mismatches,
        'policy_implementation_sha256':file_hash(Path(__file__).with_name('admission_structure.py'))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--supplement', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('preserve prior review rather than overwrite it')
    args.output.mkdir(parents=True)
    catalog = json.loads(Path(__file__).with_name('catalog.json').read_text())
    market = load_market(args.data,supplement=args.supplement,sectors=catalog['sectors'])
    if market.fingerprint() != 'd9ded1d933f292933aa09145bc36a5696421c43dbf266fce018a34ef0b9f3f9b':
        raise ValueError('review requires the original frozen market')
    reports = []
    for scope in ('union','chatgpt_5','joint_optical_leader_removal'):
        accounts = {}
        output = args.output/scope
        output.mkdir()
        for policy in ('admission_structure','incumbent','buy_hold'):
            path = args.evidence/'evaluation/runs'/f'{scope}_{policy}'
            result,m,days,episodes,daily,reconciliation = recorded_account(market,path)
            if result.metadata['source']['commit'] != '8ebcb5311f08f77d8e3077277126bd6bbd43504c':
                raise ValueError('unexpected source in recovered accounts')
            days.to_csv(output/f'{policy}_pnl.csv',index=False,float_format='%.17g')
            episodes.to_csv(output/f'{policy}_episodes.csv',index=False,float_format='%.17g')
            accounts[policy] = dict(result=result,daily=daily,metrics=metrics(result),ledger=reconciliation,
                                    identity_sha256=file_hash(path/'identity.json'),manifest_sha256=file_hash(path/'manifest.json'))
        c = accounts['admission_structure']
        result = c['result']
        states, equivalence = observe_same_decisions(m,result)
        (output/'decision_identity.json').write_text(json.dumps(equivalence,indent=2)+'\n')
        states.to_csv(output/'observed_decisions.csv',index=False,float_format='%.17g')
        comparisons = {}
        for policy in ('incumbent','buy_hold'):
            b = accounts[policy]
            for key in ('config','universe','data_sha256','delay','cost_multiplier','start','end'):
                if c['result'].metadata[key] != b['result'].metadata[key]:
                    raise ValueError('incomparable accounts: '+key)
            gap = decompose(c['daily'],b['daily'])
            gap.to_csv(output/f'{policy}_minus_candidate_log.csv',float_format='%.17g')
            expected = np.log(b['metrics']['wealth']/c['metrics']['wealth'])
            if abs(gap.total_log.sum()-expected) > 1e-10:
                raise ValueError('whole-account relative wealth does not reconcile')
            pre = gap.loc[:'2025-12-31']
            comparisons[policy] = {'full':gap.sum().to_dict(),'pre_2026':pre.sum().to_dict()}
        fills = [o for o in result.orders if o['status']=='FILLED']
        blocks = [o for o in result.orders if o['status']=='BLOCKED']
        pd.DataFrame(blocks).to_csv(output/'blocked_orders.csv',index=False,float_format='%.17g')
        signal_reasons = result.equity.reason.to_dict()
        protective = [o for o in fills if o['side']=='SELL' and 'PROTECTIVE' in signal_reasons[pd.Timestamp(o['signal_date'])]]
        by_reason = states.groupby('reason').agg(sessions=('date','count'),mean_exposure=('actual_exposure','mean'),
            cut_value=('cut_value','sum')).reset_index()
        by_reason.to_csv(output/'reason_counts.csv',index=False,float_format='%.17g')
        report = dict(scope=scope,accounts={k:{a:v for a,v in value.items() if a not in ('result','daily')} for k,value in accounts.items()},
            comparison_convention='reference minus candidate; exact accounting, NOT achievable counterfactual',
            log_gap=comparisons,decision_equivalence=equivalence,
            observed=dict(flat_sessions=int(states.actual_exposure.lt(.01).sum()),
                zero_cap_sessions=int(states.risk_cap.eq(0).sum()),
                half_cap_sessions=int(states.risk_cap.eq(.5).sum()),
                protective_signal_sessions=int(states.cut_names.gt(0).sum()),
                price_stop_sessions=int(states.stop_exit_names.gt(0).sum()),
                feature_exit_sessions=int(states.feature_exit_names.gt(0).sum()),
                overlapping_stop_feature_sessions=int(states.both_stop_feature_names.gt(0).sum()),
                low_exposure_full_cap_sessions=int((states.actual_exposure.lt(.5)&states.risk_cap.eq(1)).sum()),
                low_exposure_full_cap_with_free_qualified=int((states.actual_exposure.lt(.5)&states.risk_cap.eq(1)&states.free_qualified_names.gt(0)).sum()),
                positive_readmit_block_sessions=int(states.readmission_blocked_names.gt(0).sum()),
                protective_sell_fills=len(protective),actual_protective_sell_fees=float(sum(o['fee'] for o in protective)),
                blocked_attempts=len(blocks)))
        reports.append(report)
        print(json.dumps({'scope':scope,'log_gap':comparisons,'observed':report['observed']},indent=2),flush=True)
    summary=dict(status='RECORDED_ACCOUNT_ATTRIBUTION_NOT_ECONOMIC_ACCEPTANCE',
        measured_source='8ebcb5311f08f77d8e3077277126bd6bbd43504c',measured_run='34915730072',
        market_sha256=market.fingerprint(),generator_sha256=file_hash(Path(__file__)),
        runtime=source_identity(),new_economic_hypotheses=0,reports=reports,
        limitation='Historical realized-account decomposition is convention-dependent, not an intervention, missed-profit sum, cost-free simulation or global feasibility bound.')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    files={str(p.relative_to(args.output)):file_hash(p) for p in sorted(args.output.rglob('*')) if p.is_file()}
    (args.output/'MANIFEST.json').write_text(json.dumps(files,indent=2)+'\n')


def paired_screen(rows):
    expected={'union','chatgpt_5','joint_optical_leader_removal'}
    if len(rows)!=3 or {r['scope'] for r in rows}!=expected:
        raise ValueError('the registered three paired scopes must each be present once')
    failures=[];strict=False
    for row in rows:
        c,t=row['control'],row['treatment']
        if any(not np.isfinite(v[k]) for v in (c,t) for k in ('wealth','max_drawdown')) or min(c['wealth'],t['wealth'])<=0 or min(c['max_drawdown'],t['max_drawdown'])<0:
            raise ValueError('paired metrics must be finite, positive wealth and nonnegative drawdown')
        if t['wealth'] < c['wealth']-1e-12:
            failures.append(dict(scope=row['scope'],metric='wealth',control=c['wealth'],treatment=t['wealth']))
        if t['max_drawdown'] > c['max_drawdown']+1e-12:
            failures.append(dict(scope=row['scope'],metric='max_drawdown',control=c['max_drawdown'],treatment=t['max_drawdown']))
        strict |= t['wealth']>c['wealth']+1e-12 or t['max_drawdown']<c['max_drawdown']-1e-12
    advance=not failures and strict
    return dict(advance=advance,status='PAIRED_SCREEN_ONLY' if advance else 'REJECTED',
                failures=failures,strict_improvement=bool(strict),economic_acceptance='UNVERIFIED')


def hosted(output, data, supplement):
    """Use the existing explicit hosted runner; never redispatch an old grid."""
    import math
    import shutil
    import tarfile
    import urllib.request
    from research.finite_study import Study
    from research.expectation_study import scopes
    output=Path(output)
    recovery=output/'recorded_origin';recovery.mkdir()
    archive=recovery/'nonlinear-evidence.tar.gz'
    origin='nonlinear/8ebcb5311f08f77d8e3077277126bd6bbd43504c/34915730072/'
    url='https://raw.githubusercontent.com/ychenracing/quant/621b147189278b96ce1cbc6e1f22a4df66280824/'+origin+archive.name
    with urllib.request.urlopen(url,timeout=45) as src,archive.open('wb') as dst:
        shutil.copyfileobj(src,dst)
    expected='4cc550c9ca4875cc10cd59b4f7e0770e0aaa421001f384484ce36b53aee994a2'
    if file_hash(archive)!=expected:raise ValueError('recorded parent archive mismatch')
    with tarfile.open(archive) as bundle:bundle.extractall(recovery,filter='data')
    parent=recovery/'nonlinear'
    for name,digest in json.loads((parent/'MANIFEST.json').read_text()).items():
        if not (parent/name).resolve().is_relative_to(parent.resolve()) or file_hash(parent/name)!=digest:
            raise ValueError('recorded parent manifest mismatch: '+name)
    main(['--data',str(data),'--supplement',str(supplement),
          '--evidence',str(parent),'--output',str(output/'decision_review')])
    catalog=json.loads(Path(__file__).with_name('catalog.json').read_text())
    market=load_market(data,supplement=supplement,sectors=catalog['sectors']).prefix('2025-12-31')
    study=Study('healthy_hold_funding');scoped=scopes(market,catalog)
    selection=output/'selection';selection.mkdir()
    plan=dict(identity=study.identity(),data_sha256=market.fingerprint(),scopes=scoped,
              parameters=[{'continuous_held_funding':False},{'continuous_held_funding':True}],
              preregistration='080767e443b0dcfc97e531c204b6283e365e5c78',
              selection_rule='paired_screen from healthy_hold_funding_contract.json; old selectors unchanged')
    (selection/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    rows=[]
    for scope,names in scoped.items():
        row=dict(scope=scope)
        for enabled,label in ((False,'control'),(True,'treatment')):
            result=study.saved(market.subset(names),selection/'runs'/f'{scope}_{label}',study.module.Parameters(enabled))
            # Every actual fill is independently reconciled before reporting the paired account.
            _,_,reconciliation=attribute(market.subset(names),result)
            row[label]=metrics(result);row[label+'_ledger']=reconciliation
            # This original-selector statistic is diagnostic only, not our new paired rule.
            bpath=parent/'selection/runs'/('incumbent_'+scope)
            b=json.loads((bpath/'metrics.json').read_text())
            v=row[label]
            row[label+'_original_incumbent_deficit']=max(0.,math.log(b['wealth']/v['wealth']),
                v['max_drawdown']/b['max_drawdown']-1.,v['orders']/max(1,b['orders'])-1.)
        rows.append(row)
        (selection/'paired-progress.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps(row),flush=True)
    decision=paired_screen(rows)
    decision.update(rows=rows,identity=study.identity(),data_sha256=market.fingerprint(),
                    candidate=1 if decision['advance'] else 0,
                    parameters={'continuous_held_funding':decision['advance']},
                    new_accounts=6,research_budget='one preregistered structural hypothesis; no adjacent retries',
                    full_evaluation='NOT_RUN: preserve independent final coverage until joint screen warrants it')
    (selection/'selection.json').write_text(json.dumps(decision,indent=2)+'\n')
    (output/'decision_review/pair_outcome.json').write_text(json.dumps(decision,indent=2)+'\n')
    # Original archives already have immutable remote identities; avoid publishing a duplicate copy.
    (output/'decision_review/origin.json').write_text(json.dumps(dict(evidence_path=origin,
        archive_sha256=expected,manifest_members_verified=len(json.loads((parent/'MANIFEST.json').read_text()))),indent=2)+'\n')
    shutil.rmtree(recovery)
    review=output/'decision_review'
    manifest={str(p.relative_to(review)):file_hash(p) for p in sorted(review.rglob('*')) if p.is_file() and p.name!='MANIFEST.json'}
    (review/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return decision


if __name__ == '__main__':
    main()

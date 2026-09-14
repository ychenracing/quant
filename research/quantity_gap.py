"""Attribute matched pre-2026 log-wealth gaps; never execute a counterfactual."""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from techquant.data import load_market,file_hash
from techquant.evidence import load_result,source_identity
from research.ledger_attribution import attribute
from research.quantity_obligation import Owner, Parameters
from techquant.policy import CloseObservation



def attribute_gap(root: Path, market, out: Path):
    market = market.prefix('2025-12-31')
    out.mkdir(parents=True, exist_ok=True)
    reports=[]
    for scope in ('union','chatgpt_5','joint_optical_leader_removal'):
        by_policy={}; daily={}; ids={}; state=None
        for policy,label in [('candidate','candidate0_'),('incumbent','incumbent_')]:
            p=root/'selection/runs'/(label+scope)
            identity=json.loads((p/'identity.json').read_text());ids[policy]=identity
            assert identity['end']=='2025-12-31'
            assert identity['source']['commit']=='180640bff6d3a125cdf1b8928a6e0fc32a1d5f1a'
            assert identity['source']['package_sha256']==source_identity()['package_sha256']
            for name in ('quantity_obligation.py','support_budget.py','funded_risk.py'):
                assert file_hash(Path(__file__).with_name(name))==identity['study']['dependencies'][name]
            result=load_result(p,expected=identity)
            m=market.subset(identity['universe']); days,episodes,ledger=attribute(m,result)
            nav=result.equity.nav;previous=nav.shift().fillna(identity['config']['initial_cash']);ret=nav/previous-1
            multiplier=pd.Series(1.,index=ret.index);nonzero=ret.ne(0);multiplier[nonzero]=np.log1p(ret[nonzero])/ret[nonzero]
            days['date']=pd.to_datetime(days.date)
            days['log_contribution']=days.pnl/days.date.map(previous)*days.date.map(multiplier)
            days['carry_log_contribution']=days['carry']/days.date.map(previous)*days.date.map(multiplier)
            days['execution_log_contribution']=days.execution/days.date.map(previous)*days.date.map(multiplier)
            daily[policy]=days.pivot(index='date',columns='symbol',values='log_contribution').reindex(index=m.calendar,columns=m.symbols).fillna(0)
            assert abs(daily[policy].to_numpy().sum()-math.log(nav.iloc[-1]/identity['config']['initial_cash']))<1e-11
            days.to_csv(out/f'{scope}_{policy}_symbol_days.csv',index=False,float_format='%.17g')
            episodes.to_csv(out/f'{scope}_{policy}_episodes.csv',index=False,float_format='%.17g')
            by_policy[policy]={'wealth':float(nav.iloc[-1]/identity['config']['initial_cash']),'ledger':ledger}
            if policy=='candidate':
                owner=Owner(m,Parameters('support'));units=np.zeros(len(m.symbols));index={s:j for j,s in enumerate(m.symbols)}
                fill={str(d.date()):[] for d in m.calendar}
                for order in result.orders:
                    if order['status']=='FILLED':fill[order['date']].append(order)
                states=[];max_error=0.
                for i,date in enumerate(m.calendar):
                    day=str(date.date())
                    for order in fill[day]:
                        j=index[order['symbol']];q=float(order['units']);units[j]=units[j]+q if order['side']=='BUY' else max(0.,units[j]-q)
                    equity=result.equity.loc[date];price=np.nan_to_num(owner.inner.features.close[i],nan=0.)
                    obs=CloseObservation.from_inventory(i,day,float(equity.nav),float(equity.cash),units,units*price/equity.nav)
                    decision=owner.decide(obs);error=float(np.abs(decision.weights-result.targets.loc[date].to_numpy()).max());max_error=max(max_error,error)
                    assert error<1e-13,(scope,day,error)
                    f=owner.inner.features
                    for j in range(len(units)):
                        states.append(dict(date=day,symbol=m.symbols[j],actual_units=units[j],weight=units[j]*price[j]/equity.nav,
                            economic_target=float(owner.inner.reduction_ceiling[j]) if np.isfinite(owner.inner.reduction_ceiling[j]) else None,
                            declared_units=decision.unit_targets[j],stop=owner.inner.stop[j],close=price[j],
                            stop_broken=bool(units[j]>1e-10 and price[j]<=owner.inner.stop[j]),price_exit=bool(f.exit[i,j]),
                            ready=bool(owner.inner.ready[i,j]),entry=bool(f.entry[i,j]),score=float(f.score[i,j]) if np.isfinite(f.score[i,j]) else None,
                            cap=owner.inner.risk.cap,cash_fraction=equity.cash/equity.nav,reason=decision.reason))
                state=pd.DataFrame(states);state.to_csv(out/f'{scope}_candidate_decisions.csv',index=False,float_format='%.17g')
                by_policy[policy]['maximum_target_error']=max_error
        for key in ('config','universe','data_sha256','delay','cost_multiplier','start','end'):
            assert ids['candidate'][key]==ids['incumbent'][key],key
        assert ids['candidate']['source']==ids['incumbent']['source']
        diff=daily['candidate']-daily['incumbent'];daily_total=diff.sum(axis=1)
        assert abs(float(diff.to_numpy().sum())-math.log(by_policy['candidate']['wealth']/by_policy['incumbent']['wealth']))<1e-11
        diff.to_csv(out/f'{scope}_relative_log_contributions.csv',float_format='%.17g')
        monthly=diff.groupby(diff.index.to_period('M')).sum();monthly.to_csv(out/f'{scope}_monthly_relative_log.csv',float_format='%.17g')
        details=[]
        for month,val in monthly.sum(axis=1).sort_values().head(5).items():
            details.append({'month':str(month),'relative_log_wealth':float(val),'symbols':{s:float(x) for s,x in monthly.loc[month].sort_values().items()}})
        report=dict(scope=scope,policies=by_policy,relative_log_wealth=float(diff.to_numpy().sum()),symbol_relative_log_wealth={s:float(x) for s,x in diff.sum().sort_values().items()},worst_months=details)
        reports.append(report);print(json.dumps(report,indent=2),flush=True)
    result=dict(status='MATCHED_PRE_2026_DIAGNOSTIC_NOT_ECONOMIC_ACCEPTANCE',measured_source='180640bff6d3a125cdf1b8928a6e0fc32a1d5f1a',original_archive_sha256='ffc82ff43d6d5ac01d2acbc9f0c265885e916d09d1c717a492df989687f45ec7',
        method='Use original filled-inventory PnL reconciliation. Convert per-stock daily arithmetic PnL/NAV to exact additive log-wealth contributions using log1p(account_return)/account_return (continuous limit one). Compare two actual funded paths with identical source/data/config/universe/execution. Reconstruct every selected close decision. Contributions are descriptive, not replacement strategy returns or causal treatment effects.',
        research_cutoff='2025-12-31',no_new_portfolio_runs=True,no_new_model_fits=True,scopes=reports)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'MANIFEST.json').write_text(json.dumps({p.name:file_hash(p) for p in sorted(out.glob('*')) if p.name!='MANIFEST.json'},indent=2)+'\n')
    return result


def hosted(output: Path, data: Path, supplement: Path):
    """Retrieve only the already-measured parent; no original study is rerun."""
    import tarfile
    import urllib.request
    origin = '180640bff6d3a125cdf1b8928a6e0fc32a1d5f1a/34836286943'
    expected = 'ffc82ff43d6d5ac01d2acbc9f0c265885e916d09d1c717a492df989687f45ec7'
    folder = output.parent/'quantity-attribution-origin'
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder/'evidence.tar.gz'
    url = 'https://raw.githubusercontent.com/ychenracing/quant/research/evidence/nonlinear/'+origin+'/nonlinear-evidence.tar.gz'
    with urllib.request.urlopen(url, timeout=60) as response:
        archive.write_bytes(response.read())
    if file_hash(archive) != expected:
        raise ValueError('pinned quantity parent archive mismatch')
    with tarfile.open(archive) as bundle:
        bundle.extractall(folder, filter='data')
    root = folder/'nonlinear'
    for name, digest in json.loads((root/'MANIFEST.json').read_text()).items():
        if file_hash(root/name) != digest:
            raise ValueError('quantity parent manifest mismatch: '+name)
    if (root/'source-commit.txt').read_text().strip() != origin.split('/')[0]:
        raise ValueError('quantity parent source mismatch')
    catalog = json.loads(Path(__file__).with_name('catalog.json').read_text())
    market = load_market(data, supplement=supplement, sectors=catalog['sectors'])
    return attribute_gap(root, market, output/'quantity_gap')

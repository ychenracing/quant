"""Read-only recovery of two completed pairs; no new portfolio replay or fitting."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd

PAIRS=(
    ('campaign_payoff','eddbc617787d85215124d974ea10ec8c4e89da68','34964181664',
     'd5266c62a8d75c8cbd7fed3d2c8cbf8e14639738','5881c2d6eb7efff5252baf7dce5ca0ad201021e83ba61fd19f133c6bbbec0349'),
    ('joint_funding','0578a23c52007bde657ed97e9ce033fa58675ce6','34966383125',
     '565660bea72e1dc2783aea2fe27f8d5e747ffdaf','abd1b3687813a9e18dc7030669991db8a272d9d4cb3ffad2f96bf0bffa1cba96'))


def volume_context(market,episodes,cfg):
    """Describe original closed campaigns, never label open positions as settled.

    This historical association is a hypothesis diagnostic, not an investment
    strategy, a causal effect, or untouched out-of-sample prediction evidence.
    """
    amount=market.panel('volume')*market.panel('raw_close')
    ratio=amount.rolling(cfg.fast,min_periods=cfg.fast).mean()/amount.rolling(cfg.slow,min_periods=cfg.fast).mean()
    rows=[];opened=missing=0
    for row in episodes.to_dict('records'):
        if row['exit']=='OPEN':opened+=1;continue
        value=float(ratio.loc[pd.Timestamp(row['entry_signal']),row['symbol']])
        if not np.isfinite(value) or row['buy_notional']<=0:missing+=1;continue
        rows.append({'symbol':row['symbol'],'signal':row['entry_signal'],'settled':row['exit'],
                     'turnover_ratio':value,'payoff':float(row['pnl']/row['buy_notional'])})
    frame=pd.DataFrame(rows,columns=['symbol','signal','settled','turnover_ratio','payoff'])
    winners=frame.loc[frame.payoff>0,'turnover_ratio'];losers=frame.loc[frame.payoff<0,'turnover_ratio']
    correlation=(frame.turnover_ratio.corr(frame.payoff,method='spearman') if len(frame)>=3 else np.nan)
    summary={'settled_contexts':len(frame),'censored_open':opened,'unavailable_context':missing,
        'winning_context_median':float(winners.median()) if len(winners) else None,
        'losing_context_median':float(losers.median()) if len(losers) else None,
        'payoff_rank_correlation':float(correlation) if np.isfinite(correlation) else None,
        'interpretation':'selected already-studied control campaigns; no threshold/model fitted, no causal or predictive guarantee'}
    return frame,summary


def check_dependencies(node, research, runtime):
    """File hashes and runtime version maps are different provenance namespaces."""
    if not isinstance(node,dict):
        return
    root=Path(research).resolve()
    for key,value in node.items():
        if key == 'source' and isinstance(value,dict):
            if (value.get('python')!=runtime.get('python')
                    or value.get('dependencies')!=runtime.get('dependencies')):
                raise ValueError('recorded numerical runtime differs')
        elif key.endswith('dependencies') and isinstance(value,dict):
            for name,expected in value.items():
                path=Path(name)
                if (path.is_absolute() or '..' in path.parts
                        or not isinstance(expected,str) or len(expected)!=64
                        or any(c not in '0123456789abcdef' for c in expected)):
                    raise ValueError('invalid source dependency path or digest')
                target=(root/path).resolve()
                if not target.is_relative_to(root) or not target.is_file():
                    raise ValueError('missing or unsafe measurement dependency: '+name)
                if hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
                    raise ValueError('changed measurement dependency: '+name)
        elif isinstance(value,dict):
            check_dependencies(value,root,runtime)


def review(output,data,supplement):
    from techquant.config import Config
    from techquant.data import file_hash,load_market
    from techquant.evidence import source_identity,metrics
    from research.relative_leadership_audit import unpack_verified
    from research.expectation_study import write_json,scopes
    from research.decision_review import recorded_account,decompose
    out=Path(output)/'account_review';out.mkdir(parents=True,exist_ok=True)
    research=Path(__file__).parent;current=source_identity();cfg=Config()
    catalog=json.loads((research/'catalog.json').read_text())
    market=load_market(data,supplement=supplement,sectors=catalog['sectors'])
    if market.fingerprint()!='d9ded1d933f292933aa09145bc36a5696421c43dbf266fce018a34ef0b9f3f9b':
        raise ValueError('review requires original frozen market')
    market=market.prefix('2025-12-31');records=[];controls={}
    for family,source,run,evidence,digest in PAIRS:
        url=f'https://raw.githubusercontent.com/ychenracing/quant/{evidence}/nonlinear/{source}/{run}/nonlinear-evidence.tar.gz'
        with urllib.request.urlopen(url,timeout=60) as response:payload=response.read(90_000_001)
        if len(payload)>90_000_000:raise ValueError('unexpected archive size')
        root,members=unpack_verified(payload,digest,Path(output).parent/('recorded-'+family))
        selection=json.loads((root/'selection/selection.json').read_text());receipt=json.loads((root/'receipt.json').read_text())
        identity=selection['identity']
        if (identity['source']['commit']!=source or receipt['source_commit']!=source
                or str(receipt['run_id'])!=run or selection['advance']
                or (root/'source-commit.txt').read_text().strip()!=source
                or identity['source']['files']!=current['files']):
            raise ValueError('recorded source, rejected pair or accounting identity mismatch')
        check_dependencies(identity,research,current);scoped=[]
        for scope,names in scopes(market,catalog).items():
            accounts={label:recorded_account(market,root/'selection/runs'/(scope+'_'+label))
                      for label in ('control','treatment')}
            c,t=accounts['control'],accounts['treatment']
            for label,account in accounts.items():
                if account[0].metadata['source']['commit']!=source:raise ValueError('wrong account source')
                reported=next(r[label] for r in selection['rows'] if r['scope']==scope)
                computed=metrics(account[0])
                if any(abs(computed[k]-reported[k])>1e-10 for k in ('wealth','max_drawdown','orders')):
                    raise ValueError('recorded economic headline does not recompute')
                account[3].sort_values('pnl').to_csv(out/f'{family}-{scope}-{label}-episodes.csv',index=False,float_format='%.17g')
            gap=decompose(c[4],t[4]);gap.to_csv(out/f'{family}-{scope}-daily-gap.csv',float_format='%.17g')
            pnl=pd.concat([c[2].groupby('symbol').pnl.sum().rename('control'),t[2].groupby('symbol').pnl.sum().rename('treatment')],axis=1).fillna(0.)
            pnl['difference']=pnl.treatment-pnl.control;pnl=pnl.sort_values('difference')
            pnl.to_csv(out/f'{family}-{scope}-symbol-pnl.csv',float_format='%.17g')
            delta=float(t[0].equity.nav.iloc[-1]-c[0].equity.nav.iloc[-1])
            error=float(pnl.difference.sum()-delta)
            if abs(error)>1e-5:raise ValueError('terminal symbol attribution mismatch')
            if scope not in controls:
                contexts,controls[scope]=volume_context(c[1],c[3],cfg)
                contexts.to_csv(out/f'{scope}-control-turnover-context.csv',index=False,float_format='%.17g')
            if family=='campaign_payoff':
                learned=json.loads((root/'selection/payoffs'/f'{scope}_treatment.json').read_text())
                if any(f['latest_settlement_session']>f['session'] for f in learned['fits']):
                    raise ValueError('immature campaign sample used in a fit')
            def extremes(frame):
                return [{'symbol':str(s),**{k:float(v) for k,v in r.items()}} for s,r in frame.iterrows()]
            scoped.append({'scope':scope,'terminal_nav_difference':delta,'symbol_error':error,
                'log_gap':{k:float(v) for k,v in gap.sum().items()},
                'largest_negative':extremes(pnl.loc[pnl.difference<0].head(5)),
                'largest_positive':extremes(pnl.loc[pnl.difference>0].tail(5).iloc[::-1]),
                'control_ledger':c[5],'treatment_ledger':t[5]})
        records.append({'family':family,'measurement_source':source,'measurement_run':run,
            'evidence_commit':evidence,'archive_sha256':digest,'archive_bytes':len(payload),
            'manifest_files_verified':members,'source_tree':(root/'source-tree.txt').read_text().strip(),
            'accounts_verified':6,'rows':scoped})
        write_json(out/'progress.json',records)
    write_json(out/'review.json',{'status':'TWO_REJECTED_PAIRS_INDEPENDENTLY_RECONCILED',
        'producer':current,'pairs':records,'controls_nonprice_context':controls,
        'new_strategy_accounts':0,'parameter_trials':0,'model_fits':0,'economic_acceptance':'NOT_MET',
        'limitations':['Arithmetic account attribution is not a feasible profit bound or causal intervention.',
                       'Turnover diagnostics are retrospective selected-control associations, not a trained strategy or new authorization.',
                       'Do not combine per-pool winners or relabel old source economics as current HEAD acceptance.']})
    print(json.dumps({'verified_accounts':12,'new_accounts':0,'manifest_files':sum(r['manifest_files_verified'] for r in records)}),flush=True)

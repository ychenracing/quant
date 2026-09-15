"""Read-only attribution of the exact completed relative-leadership pair.

No strategy replay, refit, parameter search or new economic treatment is allowed.
Forecast accuracy and recorded-account attribution are not investable profits.
"""
from __future__ import annotations
import hashlib,io,json,tarfile
from pathlib import Path,PurePosixPath
import urllib.request
import numpy as np

SOURCE='0cca61f6f38019fc47b0901eb58b891662e71d0f'
RUN='34960173025'
EVIDENCE='1291edc243eec1aab9f8e71f9eaa133cff3ecd4e'
DIGEST='ce5a4bbfefe741f0e721dc85ce43231e94bd1f406b5f0d00eddd34291a533aad'


def unpack_verified(payload:bytes, expected:str, destination:Path):
    if hashlib.sha256(payload).hexdigest()!=expected:
        raise ValueError('recorded archive hash mismatch')
    destination=Path(destination);seen=set();files={}
    with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:
        for member in archive.getmembers():
            name=PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or name.parts[0]!='nonlinear':
                raise ValueError('unsafe archive member')
            if member.isdir():continue
            if not member.isfile() or str(name) in seen:
                raise ValueError('nonregular or duplicate archive member')
            seen.add(str(name));body=archive.extractfile(member).read()
            target=destination.joinpath(*name.parts);target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(body);files[str(name.relative_to('nonlinear'))]=hashlib.sha256(body).hexdigest()
    root=destination/'nonlinear';manifest=json.loads((root/'MANIFEST.json').read_text())
    actual={k:v for k,v in files.items() if k!='MANIFEST.json'}
    if actual!=manifest:raise ValueError('recorded manifest members or hashes mismatch')
    return root,len(manifest)


def pair_accuracy(prediction, outcome):
    p,y=np.asarray(prediction,dtype=float),np.asarray(outcome,dtype=float)
    if p.ndim!=1 or y.shape!=p.shape or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('paired accuracy needs aligned finite cross-sectional samples')
    left,right=np.triu_indices(len(y),1);truth=np.sign(y[left]-y[right]);pred=np.sign(p[left]-p[right])
    mask=truth!=0
    if not mask.any():return None
    return float(np.mean(np.where(pred[mask]==0,.5,(pred[mask]==truth[mask]).astype(float))))


def review(output,data,supplement):
    import pandas as pd
    from techquant.config import Config
    from techquant.data import file_hash,load_market
    from techquant.evidence import source_identity
    from techquant.features import build_features
    from research.expectation_study import write_json,scopes
    from research.decision_review import recorded_account,decompose
    from research.ledger_attribution import attribute
    out=Path(output)/'account_review';out.mkdir(parents=True,exist_ok=True)
    url=f'https://raw.githubusercontent.com/ychenracing/quant/{EVIDENCE}/nonlinear/{SOURCE}/{RUN}/nonlinear-evidence.tar.gz'
    with urllib.request.urlopen(url,timeout=60) as response:payload=response.read(30_000_001)
    if len(payload)>30_000_000:raise ValueError('unexpected recorded archive size')
    root,members=unpack_verified(payload,DIGEST,Path(output).parent/'relative-recorded')
    selection=json.loads((root/'selection/selection.json').read_text())
    receipt=json.loads((root/'receipt.json').read_text())
    if (selection['identity']['source']['commit']!=SOURCE or receipt['source_commit']!=SOURCE
            or str(receipt['run_id'])!=RUN or selection['advance']
            or selection['status']!='REJECTED_PAIRED_SCREEN'):
        raise ValueError('only the exact already-rejected pair is being reviewed')
    if (root/'source-commit.txt').read_text().strip()!=SOURCE:
        raise ValueError('recorded source receipt mismatch')
    current=source_identity()
    if current['files']!=selection['identity']['source']['files']:
        raise ValueError('accounting implementation differs from recorded measurement')
    research=Path(__file__).parent
    for name,digest in selection['identity']['relative_dependencies'].items():
        if file_hash(research/name)!=digest:
            raise ValueError('measurement dependencies cannot change during review: '+name)
    catalog=json.loads((research/'catalog.json').read_text());cfg=Config()
    market=load_market(data,supplement=supplement,sectors=catalog['sectors'])
    if market.fingerprint()!='d9ded1d933f292933aa09145bc36a5696421c43dbf266fce018a34ef0b9f3f9b':
        raise ValueError('only the original frozen market is admissible')
    market=market.prefix('2025-12-31');reviews=[]
    for scope,names in scopes(market,catalog).items():
        accounts={}
        for label in ('control','treatment'):
            path=root/'selection/runs'/(scope+'_'+label)
            record=recorded_account(market,path)
            if record[0].metadata['source']['commit']!=SOURCE:
                raise ValueError('account source mismatch')
            accounts[label]=record
        c,t=accounts['control'],accounts['treatment']
        allocation=decompose(c[4],t[4])
        allocation.to_csv(out/(scope+'-daily-log-gap.csv'),index=True,float_format='%.17g')
        pnl=pd.concat([c[2].groupby('symbol').pnl.sum().rename('control'),
                       t[2].groupby('symbol').pnl.sum().rename('treatment')],axis=1).fillna(0)
        pnl['difference']=pnl.treatment-pnl.control
        pnl=pnl.sort_values('difference')
        pnl.to_csv(out/(scope+'-symbol-pnl.csv'),float_format='%.17g')
        wealth_delta=float(t[0].equity.nav.iloc[-1]-c[0].equity.nav.iloc[-1])
        if abs(pnl.difference.sum()-wealth_delta)>1e-5:
            raise ValueError('symbol differences do not reconcile with recorded terminal NAV')
        for label,account in accounts.items():
            account[3].sort_values('pnl').to_csv(out/(scope+'-'+label+'-episodes.csv'),index=False,float_format='%.17g')
        differs=(t[0].equity.nav-c[0].equity.nav).abs()>1e-6
        ranks=root/'selection/rankings'/(scope+'_treatment.npz')
        rank_info=json.loads(ranks.with_suffix('.json').read_text())
        if any(f['latest_label_session']>f['session'] for f in rank_info['fits']):
            raise ValueError('issued fit provenance contains immature labels')
        with np.load(ranks,allow_pickle=False) as saved:pred=saved['relative']
        m=t[1];opening=m.panel('open').to_numpy();active=(m.panel('volume').gt(0)&m.panel('close').notna()).to_numpy()
        f=build_features(m,cfg);cohorts=[]
        for i in range(len(m.calendar)-cfg.rebalance-1):
            end=i+cfg.rebalance+1
            usable=(np.isfinite(pred[i]) & np.isfinite(f.score[i]) & active[i+1]&active[end]
                & np.isfinite(opening[i+1])&np.isfinite(opening[end])&(opening[i+1]>0)&(opening[end]>0))
            if np.count_nonzero(usable)<2:continue
            y=np.log(opening[end,usable]/opening[i+1,usable])
            a,b=pair_accuracy(pred[i,usable],y),pair_accuracy(f.score[i,usable],y)
            if a is not None:
                cohorts.append({'formation':str(m.calendar[i].date()),'maturity':str(m.calendar[end].date()),
                    'names':int(usable.sum()),'learned':a,'original':b})
        frame=pd.DataFrame(cohorts);frame.to_csv(out/(scope+'-issued-rank-audit.csv'),index=False,float_format='%.17g')
        annual=[]
        for year in (2023,2024,2025):
            row={'year':year}
            for label,account in accounts.items():
                nav=account[0].equity.nav;before=nav.loc[nav.index<pd.Timestamp(f'{year}-01-01')]
                opening_nav=float(before.iloc[-1]) if len(before) else cfg.initial_cash
                ending_nav=float(nav.loc[nav.index<=pd.Timestamp(f'{year}-12-31')].iloc[-1])
                row[label]=ending_nav/opening_nav-1
            annual.append(row)
        def tails(part):
            return [{'symbol':str(s),**{k:float(v) for k,v in row.items()}} for s,row in part.iterrows()]
        reviews.append({'scope':scope,'terminal_nav_difference':wealth_delta,
            'first_nav_divergence':str(t[0].equity.index[differs][0].date()) if differs.any() else None,
            'log_gap':{k:float(v) for k,v in allocation.sum().items()},
            'symbol_pnl_sum_error':float(pnl.difference.sum()-wealth_delta),
            'largest_negative_symbol_contributions':tails(pnl.head(5)),
            'largest_positive_symbol_contributions':tails(pnl.tail(5).iloc[::-1]),
            'annual_actual_account_returns':annual,
            'issued_rank_diagnostic':{'cohorts':len(frame),'learned_pair_accuracy':float(frame.learned.mean()) if len(frame) else None,
                'original_pair_accuracy':float(frame.original.mean()) if len(frame) else None,
                'population':'same available matured cross-sectional pairs, equal cohort weight; not actual traded opportunities or an independent unseen sample'},
            'control_ledger':c[5],'treatment_ledger':t[5]})
        write_json(out/'progress.json',reviews)
    write_json(out/'review.json',{'status':'READ_ONLY_RECORDED_ACCOUNT_REVIEW',
        'producer':current,'measurement_source':SOURCE,'measurement_run':RUN,'evidence_commit':EVIDENCE,
        'archive_sha256':DIGEST,'archive_bytes':len(payload),'manifest_files_verified':members,
        'source_tree':(root/'source-tree.txt').read_text().strip(),
        'new_strategy_accounts':0,'refits':0,'parameter_trials':0,'treatment_2026_runs':0,
        'economic_acceptance':'NOT_MET','rows':reviews,
        'limitations':['Attribution is an arithmetic partition between two actual accounts, not a causal intervention or feasible profit bound.',
                      'Rank accuracy uses already-studied historical cohorts and is not an out-of-sample guarantee.',
                      'The original rejected pair and all raw outputs remain unchanged; this review cannot promote it.']})
    print(json.dumps({'reviewed_accounts':6,'new_accounts':0,'manifest_files':members}),flush=True)

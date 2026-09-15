"""Restore fixed pre-2026 observations on actual fills; do not run a new account."""
from pathlib import Path
import json,tarfile,hashlib,base64,io
import numpy as np,pandas as pd
from techquant.data import file_hash,load_market
from techquant.evidence import load_result
from research.quantity_obligation import SupportIntent
from research.observed_admission_completion import Owner as Parent,Parameters
from research.ledger_attribution import attribute

SOURCE='8ebcb5311f08f77d8e3077277126bd6bbd43504c'
RUN='34915730072'
ARCHIVE='4cc550c9ca4875cc10cd59b4f7e0770e0aaa421001f384484ce36b53aee994a2'


class ObservedSupport(SupportIntent):
    def _reduce_exposure(self,o,desired,price,cap,exposure):
        ids=np.flatnonzero(desired>1e-10)
        if 0<cap<exposure and len(ids)>1:
            distance=np.maximum(price-self.stop,.02*price)
            self.observations.append(dict(scope=self.scope,date=o.date,session=o.session,
                nav=o.nav,cash=o.cash,cap=cap,exposure_before=exposure,
                symbols=[self.market.symbols[j] for j in ids],actual_units=o.units[ids].tolist(),
                upper_units=desired[ids].tolist(),original_pro_rata_units=(desired[ids]*cap/exposure).tolist(),
                prices=price[ids].tolist(),support_risk_distance=distance[ids].tolist(),
                scores=[float(self.features.score[o.session,j]) if np.isfinite(self.features.score[o.session,j]) else None for j in ids],
                prior_obligation_ceiling=[float(self.reduction_ceiling[j]) if np.isfinite(self.reduction_ceiling[j]) else None for j in ids],reason=None))
        return super()._reduce_exposure(o,desired,price,cap,exposure)


def diagnose(parent,market,output):
    from techquant.policy import CloseObservation
    parent,output=Path(parent),Path(output)
    receipt=json.loads((parent/'receipt.json').read_text())
    if receipt['source_commit']!=SOURCE or str(receipt['run_id'])!=RUN:
        raise ValueError('recorded control origin mismatch')
    riskdir=output/'opening-risk';reducedir=output/'reduction-opportunity'
    riskdir.mkdir(parents=True,exist_ok=True);reducedir.mkdir(exist_ok=True)
    market=market.prefix('2025-12-31');risk_summary=[];reduction_summary=[];allrecords=[]
    for scope in ('union','chatgpt_5','joint_optical_leader_removal'):
        path=parent/'selection/runs'/('candidate0_'+scope);ident=json.loads((path/'identity.json').read_text())
        r=load_result(path,expected=ident);m=market.subset(ident['universe']);owner=Parent(m,Parameters());old=owner.inner
        p=ObservedSupport(m,old.params,config=old.config)
        p.features,p.atr,p.support,p.ready=old.features,old.atr,old.support,old.ready
        p.observations=[];p.scope=scope;owner.inner=p
        byday={str(d.date()):[] for d in m.calendar}
        for order in r.orders:
            if order['status']=='FILLED':byday[order['date']].append(order)
        units=np.zeros(len(m.symbols));intent={};records=[];maxerr=0.
        _,episodes,recon=attribute(m,r);ep={(e.symbol,e.entry):e for e in episodes.itertuples()}
        for i,d in enumerate(m.calendar):
            day=str(d.date())
            for order in byday[day]:
                j=m.symbols.index(order['symbol']);q=order['units'];prior=units[j]
                if order['side']=='BUY':
                    signal=intent[order['signal_date']];price=order['notional']/q
                    requested=max(0.,signal['target'][j]-signal['units'][j]);grant=requested*signal['distance'][j]
                    actual=q*max(price-signal['stop'][j],.02*price)
                    episode=ep.get((order['symbol'],day)) if prior<=1e-10 else None
                    records.append(dict(scope=scope,date=day,signal=order['signal_date'],symbol=order['symbol'],fresh=prior<=1e-10,units=q,signal_requested=requested,signal_price=signal['price'][j],paid_price=price,signal_stop=signal['stop'][j],risk_grant=grant,paid_support_distance_risk=actual,excess=actual-grant,ratio=actual/grant if grant>0 else None,gap=price/signal['price'][j]-1,campaign_return=episode.return_on_buys if episode else None,campaign_pnl=episode.pnl if episode else None,episode_closed=episode.exit!='OPEN' if episode else None))
                    units[j]+=q
                else:units[j]-=q
                if units[j]<1e-8:units[j]=0.
            price=np.nan_to_num(p.features.close[i],nan=0.);e=r.equity.loc[d]
            obs=CloseObservation.from_inventory(i,day,float(e.nav),float(e.cash),units,units*price/e.nav)
            dec=owner.decide(obs);err=float(np.max(np.abs(dec.weights-r.targets.loc[d].to_numpy())))
            maxerr=max(maxerr,err)
            if err>=1e-13:raise ValueError('default reduction hook changed a recorded control decision')
            if p.observations and p.observations[-1]['date']==day:
                p.observations[-1]['reason']=dec.reason.split('|')[0]
            stops=np.where(units>1e-10,p.stop,np.maximum(p.admission_stop(i),p.pending_stop))
            distance=np.maximum(price-stops,.02*price)
            intent[day]={'units':units.copy(),'target':dec.unit_targets.copy(),'stop':stops.copy(),'distance':distance.copy(),'price':price.copy()}
        df=pd.DataFrame(records);df.to_csv(riskdir/(scope+'.csv'),index=False,float_format='%.17g')
        fresh=df[df.fresh];viol=df.excess>1e-6
        summary=dict(scope=scope,buys=len(df),risk_overshoots=int(viol.sum()),risk_grant_sum=float(df.risk_grant.sum()),positive_excess_sum=float(df.excess.clip(lower=0).sum()),max_overshoot_ratio=float(df.ratio.max()),max_target_error=maxerr,ledger_maxerror=recon['max_reconciliation_error'])
        for mask,label in ((fresh.gap>0,'gap_up'),(fresh.gap<=0,'nonpositive_gap')):
            q=fresh[mask];closed=q[q.episode_closed==True]
            summary[label]=dict(n=len(q),closed=len(closed),mean_closed_return=float(closed.campaign_return.mean()),pnl_sum=float(q.campaign_pnl.sum()))
        risk_summary.append(summary)
        for row in p.observations:
            score=np.array([s if s is not None else 0 for s in row['scores']]);risk=np.array(row['support_risk_distance'])/row['prices'];best=int(np.argmax(score))
            row['stronger_has_lower_risk_fraction']=bool(risk[best]<=np.min(risk)+1e-12 and np.ptp(score)>0)
        reduction_summary.append(dict(scope=scope,positive_cap_joint_reductions=len(p.observations),stronger_has_lower_risk=sum(row['stronger_has_lower_risk_fraction'] for row in p.observations),max_target_error=maxerr,original_accounts_reexecuted=0))
        allrecords+=p.observations
    (riskdir/'summary.json').write_text(json.dumps(dict(origin_source=SOURCE,origin_run=RUN,cutoff='2025-12-31',new_portfolio_runs=0,status='DIAGNOSTIC_NOT_A_CAUSAL_RETURN_OR_REALIZED_LOSS_BOUND',summary=risk_summary),indent=2)+'\n')
    with tarfile.open(parent/'source.tar.gz') as bundle:
        source_module_sha256=hashlib.sha256(bundle.extractfile('research/support_budget.py').read()).hexdigest()
    (reducedir/'observations.json').write_text(json.dumps(dict(source=SOURCE,run=RUN,source_module_sha256=source_module_sha256,cutoff='2025-12-31',new_portfolio_runs=0,status='UNCHANGED_DECISION_OBSERVATION_NOT_ALTERNATIVE_RETURN',summary=reduction_summary,observations=allrecords),indent=2,allow_nan=False)+'\n')
    expected=json.loads(Path(__file__).with_name('records').joinpath('protected_retention_observations.json').read_text())
    checks={name:file_hash(output/name)==digest for name,digest in expected['files'].items()}
    (output/'preservation.json').write_text(json.dumps(dict(origin_source=SOURCE,origin_run=RUN,generator_sha256=file_hash(Path(__file__)),new_portfolio_runs=0,exact_observed_bytes=checks,all_matched=all(checks.values())),indent=2)+'\n')
    if not all(checks.values()):raise ValueError('recorded-control observations differ from original bytes')


def restore_engineering(source_archive,output):
    """Decode only explicit, hash-bound normalization; retain its transport identity."""
    with tarfile.open(source_archive) as source:
        prefix='research/records/';description=json.loads(source.extractfile(prefix+'admission_structure_verification.json').read())
        parts=[]
        for part in description['parts']:
            raw=source.extractfile(prefix+part['path']).read()
            if hashlib.sha256(raw).hexdigest()!=part['transport_sha256']:raise ValueError('transport hash mismatch')
            text=raw.decode()
            for correction in part['replacements']:
                if text.count(correction['find'])!=correction['count']:raise ValueError('normalization count mismatch')
                text=text.replace(correction['find'],correction['replace'])
            if hashlib.sha256(text.encode()).hexdigest()!=part['sha256']:raise ValueError('normalized part mismatch')
            parts.append(''.join(text.split()))
    archive=base64.b64decode(''.join(parts),validate=True)
    if hashlib.sha256(archive).hexdigest()!=description['archive_sha256']:raise ValueError('original engineering archive mismatch')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        files={m.name:bundle.extractfile(m).read() for m in bundle.getmembers() if m.isfile()}
    if set(files)!=set(description['members']):raise ValueError('engineering member set mismatch')
    for name,raw in files.items():
        if Path(name).name!=name or hashlib.sha256(raw).hexdigest()!=description['members'][name]:raise ValueError('engineering member mismatch')
        (output/name).write_bytes(raw)
    (output/'verification.json').write_text(json.dumps(dict(origin_source=SOURCE,original_archive_sha256=description['archive_sha256'],members_verified=len(files),normalization='EXPLICIT_LOSSLESS_HASH_VERIFIED',economic_acceptance='UNVERIFIED'),indent=2)+'\n')


def hosted(output,data,supplement):
    import urllib.request
    root=Path(output).parent/'retention-origin';root.mkdir(exist_ok=True);archive=root/'evidence.tar.gz'
    url='https://raw.githubusercontent.com/ychenracing/quant/research/evidence/nonlinear/'+SOURCE+'/'+RUN+'/nonlinear-evidence.tar.gz'
    with urllib.request.urlopen(url,timeout=60) as response:archive.write_bytes(response.read())
    if file_hash(archive)!=ARCHIVE:raise ValueError('retention origin archive mismatch')
    with tarfile.open(archive) as bundle:bundle.extractall(root,filter='data')
    parent=root/'nonlinear'
    for name,digest in json.loads((parent/'MANIFEST.json').read_text()).items():
        if file_hash(parent/name)!=digest:raise ValueError('retention origin manifest mismatch')
    m=load_market(data,supplement=supplement,sectors=json.loads(Path(__file__).with_name('catalog.json').read_text())['sectors'])
    destination=Path(output)/'retention_diagnosis';diagnose(parent,m,destination)
    restore_engineering(parent/'source.tar.gz',destination/'prior_engineering')
    # Repair only the order-sensitive manifest using preserved original bytes;
    # the old failed reconstruction and all old economic accounts remain untouched.
    from research.admission_geometry import preserve_manifest
    geometry=destination/'prior_admission_geometry';geometry.mkdir(exist_ok=True)
    for name in ('union_episodes.csv','chatgpt_5_episodes.csv','joint_optical_leader_removal_episodes.csv','summary.json'):
        (geometry/name).write_bytes((parent/'admission_diagnosis/admission-geometry'/name).read_bytes())
    expected=json.loads(Path(__file__).with_name('records').joinpath('admission_structure_observations.json').read_text())
    preserve_manifest(geometry,Path(__file__).with_name('records')/'admission_geometry_manifest.json',expected['files']['admission-geometry/MANIFEST.json'])

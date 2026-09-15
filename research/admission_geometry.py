"""Reconstruct source-pinned admission observations, never counterfactual returns."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from techquant.data import load_market, file_hash
from techquant.evidence import load_result
from research.ledger_attribution import attribute
from research.observed_admission_completion import Owner, Parameters

SOURCE = '4d0c9ec1ada94c4e23f85a329b880adbe0bd5aa1'
RUN = '34910777906'
ARCHIVE = '2c48197eb14888e340b95f193b7affde4026614dd87a1276bd723ae73e77e73c'


def generate(parent, market, output):
    parent, output = Path(parent), Path(output)
    receipt = json.loads((parent/'receipt.json').read_text())
    if receipt['source_commit'] != SOURCE or str(receipt['run_id']) != RUN:
        raise ValueError('admission observation origin mismatch')
    market = market.prefix('2025-12-31')
    attribution, geometry = output/'auction-attribution', output/'admission-geometry'
    attribution.mkdir(parents=True, exist_ok=True); geometry.mkdir(parents=True, exist_ok=True)
    summaries = []
    for scope in ('union','chatgpt_5','joint_optical_leader_removal'):
        for candidate in (0,1):
            path = parent/'selection/runs'/f'candidate{candidate}_{scope}'
            ident = json.loads((path/'identity.json').read_text())
            if ident['source']['commit'] != SOURCE:
                raise ValueError('account source differs from pinned observation source')
            result = load_result(path, expected=ident)
            m = market.subset(ident['universe'])
            days, episodes, reconciliation = attribute(m, result)
            days.to_csv(attribution/f'{scope}_{candidate}_days.csv', index=False)
            episodes.to_csv(attribution/f'{scope}_{candidate}_episodes.csv', index=False)
            if candidate:
                continue
            p = Owner(m, Parameters()).inner
            records = []
            for episode in episodes.to_dict('records'):
                i = m.calendar.get_loc(pd.Timestamp(episode['entry_signal']))
                j = m.symbols.index(episode['symbol']); lag = p.config.fast
                current = p.support[i,j]
                prior = p.support[i-lag,j] if i >= lag else np.nan
                known = bool(np.isfinite(current) and np.isfinite(prior))
                # Reproduce the original descriptive bucket exactly. The registered
                # treatment uses strict comparison, without a fitted price tolerance.
                lower = bool(known and current < prior-1e-12)
                records.append(dict(**episode,signal_support=float(current),
                    prior_support=float(prior) if np.isfinite(prior) else None,
                    history_known=known,lower_support=lower))
            df = pd.DataFrame(records)
            df.to_csv(geometry/(scope+'_episodes.csv'),index=False,float_format='%.17g')
            for lower in (False,True):
                q = df[df.lower_support==lower]; closed = q[q.exit!='OPEN']
                summaries.append(dict(scope=scope,lower_support=lower,episodes=len(q),
                    closed=len(closed),pnl_sum=float(q.pnl.sum()),
                    mean_closed_return_on_buys=float(closed.return_on_buys.mean()),
                    wins=int((q.pnl>0).sum()),
                    max_reconciliation_error=reconciliation['max_reconciliation_error']))
    report = {'original_source':SOURCE,'original_run':RUN,'cutoff':'2025-12-31',
        'new_portfolio_runs':0,
        'hypothesis':'Fresh upward momentum over a recently falling support base may be a weak rebound. Compare existing 20-session support to its value config.fast sessions earlier; use no new fitted threshold. This diagnostic does not establish the result of withholding admission.',
        'summary':summaries,'status':'OBSERVATIONAL_NOT_CAUSAL_OR_ACCEPTED'}
    (geometry/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (geometry/'MANIFEST.json').write_text(json.dumps(
        {p.name:file_hash(p) for p in geometry.iterdir() if p.name!='MANIFEST.json'},indent=2)+'\n')
    expected = json.loads(Path(__file__).with_name('records').joinpath(
        'admission_structure_observations.json').read_text())
    checks = {name:file_hash(output/name)==digest for name,digest in expected['files'].items()}
    (output/'preservation.json').write_text(json.dumps({'origin_source':SOURCE,'origin_run':RUN,
        'expected_manifest_sha256':file_hash(Path(__file__).with_name('records')/
            'admission_structure_observations.json'),
        'generator_sha256':file_hash(Path(__file__)),'new_portfolio_runs':0,
        'exact_observed_bytes':checks,'all_matched':all(checks.values()),
        'economic_acceptance':'UNVERIFIED'},indent=2)+'\n')
    if not all(checks.values()):
        raise ValueError('derived observations differ from preserved original bytes')
    return report


def hosted(output, data, supplement):
    import tarfile
    import urllib.request
    folder = Path(output).parent/'admission-observation-origin'; folder.mkdir(exist_ok=True)
    archive = folder/'evidence.tar.gz'
    url = ('https://raw.githubusercontent.com/ychenracing/quant/research/evidence/nonlinear/'
           +SOURCE+'/'+RUN+'/nonlinear-evidence.tar.gz')
    with urllib.request.urlopen(url, timeout=60) as response:
        archive.write_bytes(response.read())
    if file_hash(archive) != ARCHIVE:
        raise ValueError('pinned observation archive checksum mismatch')
    with tarfile.open(archive) as bundle:
        bundle.extractall(folder, filter='data')
    parent = folder/'nonlinear'
    for name,digest in json.loads((parent/'MANIFEST.json').read_text()).items():
        if file_hash(parent/name) != digest:
            raise ValueError('pinned observation manifest mismatch: '+name)
    catalog = json.loads(Path(__file__).with_name('catalog.json').read_text())
    market = load_market(data,supplement=supplement,sectors=catalog['sectors'])
    return generate(parent,market,Path(output)/'admission_diagnosis')

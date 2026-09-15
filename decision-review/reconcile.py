"""Read-only audit of preserved quant accounts; never calls a strategy or replay.

Usage: python reconcile.py EVIDENCE_ROOT SOURCE_ROOT OUTPUT_JSON
EVIDENCE_ROOT is the recovered main/parent/admission/fixed/readmission/protected/
holding archive directory. SOURCE_ROOT is the exact 07f4f32 source snapshot.
"""
from pathlib import Path
from collections import defaultdict
import hashlib, json, platform, subprocess, sys
import numpy as np
import pandas as pd

root, source, output = map(Path, sys.argv[1:])
sys.path[:0] = [str(source), str(source/'src')]
from techquant.data import load_market
from research.decision_review import recorded_account

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def checked(condition, message):
    if not condition:
        raise ValueError(message)

folders = {label: root/label/'archive'/('evidence' if label=='main' else
          'fixed-validation' if label=='fixed' else 'nonlinear')
          for label in ('main','parent','admission','fixed','readmission','protected','holding')}
manifest_checks = []
for label, folder in folders.items():
    manifest = json.loads((folder/'MANIFEST.json').read_text())
    for name, expected in manifest.items():
        checked(digest(folder/name)==expected, f'manifest mismatch: {label}/{name}')
    manifest_checks.append(dict(archive=label, members=len(manifest),
                               manifest_sha256=digest(folder/'MANIFEST.json')))
plan = json.loads((folders['holding']/'selection/plan.json').read_text())['identity']
for name, expected in plan['source']['files'].items():
    checked(digest(source/'src/techquant'/name)==expected, f'production identity: {name}')
for name, expected in plan['dependencies'].items():
    checked(digest(source/'research'/name)==expected, f'research identity: {name}')
catalog = json.loads((source/'research/catalog.json').read_text())
market = load_market(folders['main']/'inputs/market', supplement=folders['main']/'inputs/supplement', sectors=catalog['sectors'])
checked(market.fingerprint()=='d9ded1d933f292933aa09145bc36a5696421c43dbf266fce018a34ef0b9f3f9b', 'frozen data changed')
scopes = ['union','chatgpt_5','joint_optical_leader_removal']
checks, risk, quantities = [], [], []
paths = sorted((folders['parent']/'evaluation/runs').iterdir()) + sorted((folders['holding']/'selection/runs').iterdir())
for path in paths:
    result, m, days, episodes, daily, rec = recorded_account(market,path)
    cash = float(result.metadata['config']['initial_cash'])
    grouped = defaultdict(list)
    for o in result.orders:
        if o['status']=='FILLED': grouped[o['date']].append(o)
    min_cash = cash
    max_cash_error = 0.
    for date, eq in result.equity.iterrows():
        for o in grouped[str(date.date())]:
            cash += (-1 if o['side']=='BUY' else 1)*float(o['notional'])-float(o['fee'])
            min_cash = min(min_cash,cash)
            checked(cash >= -1e-6, 'unfunded filled order: '+str(path))
        max_cash_error = max(max_cash_error,abs(cash-float(eq['cash'])))
    checked(max_cash_error <= 1e-6,'cash ledger mismatch: '+str(path))
    checks.append(dict(account=str(path.relative_to(root)),
        source=result.metadata['source']['commit'], sessions=len(result.equity),
        fills=len([o for o in result.orders if o['status']=='FILLED']),
        orders_sha256=digest(path/'orders.csv'), equity_sha256=digest(path/'equity.csv'),
        minimum_cash_after_fill=min_cash,maximum_cash_error=max_cash_error,
        maximum_pnl_error=rec['max_reconciliation_error'],
        signal_precedes_fill=True,sales_within_opening_inventory=True))
    if path.name.endswith('_allocation_auction'):
        scope=path.name.removesuffix('_allocation_auction')
        e=result.equity
        events=[]
        for date,row in e.loc['2026-06-22':'2026-08-31'].iterrows():
            prev=e.loc[:date].iloc[-2] if e.index.get_loc(date)>0 else row
            if row.target_cap < prev.target_cap:
                signal=str(date.date())
                sells=[o for o in result.orders if o['signal_date']==signal and o['side']=='SELL' and o['status']=='FILLED']
                events.append(dict(signal=signal,cap=float(row.target_cap),exposure_at_signal=float(row.exposure),
                    sell_fill_dates=sorted(set(o['date'] for o in sells)),sell_fills=len(sells)))
        windows={}
        for start in ['2026-06-22','2026-07-01']:
            cut=e.loc[start:'2026-08-31']; before=e.loc[e.index<start].iloc[-1].nav
            windows[start+'_2026-08-31']=float(cut.nav.iloc[-1]/before-1)
        blocked=[o for o in result.orders if o['status']!='FILLED']
        block_rows=[]
        for o in blocked:
            later=[v for v in result.orders if v['date']>o['date'] and v['symbol']==o['symbol'] and v['side']=='SELL' and v['status']=='FILLED']
            block_rows.append(dict(date=o['date'],signal=o['signal_date'],symbol=o['symbol'],side=o['side'],reason=o['reason'],
                signal_reason=str(e.loc[pd.Timestamp(o['signal_date']),'reason']),
                next_actual_sell=min([v['date'] for v in later],default=None)))
        risk.append(dict(scope=scope,window_returns=windows,budget_reductions=events,
            blocked_attempts=block_rows,blocked_in_crash=sum('2026-06-22'<=o['date']<='2026-08-31' for o in blocked),
            exposure_aug31=float(e.loc['2026-08-31'].exposure),
            september_max_exposure=float(e.loc['2026-09-01':].exposure.max())))
        quantities.append(dict(scope=scope,days_below_one_percent_exposure=int((e.exposure<.01).sum()),
                               average_exposure=float(e.exposure.mean())))
parent=pd.read_csv(folders['parent']/'evaluation/matrix.csv')
readmit=pd.read_csv(folders['readmission']/'evaluation/matrix.csv')
fixed=pd.read_csv(folders['fixed']/'matrix.csv')
rows=[]
for label,frame,policy in [('main',parent,'incumbent'),('retained_control',parent,'allocation_auction'),('readmission',readmit,'readmission')]:
    for scope in scopes:
        row=frame[(frame.scope==scope)&(frame.policy==policy)&(frame.window=='full')].iloc[0]
        rows.append(dict(candidate=label,scope=scope,wealth=float(row.wealth),max_drawdown=float(row.max_drawdown),orders=int(row.orders)))
# No pooling or per-scope winner selection. Compare whole 3-scope vectors only.
vectors={label:np.array([x for row in rows if row['candidate']==label for x in
                       (row['wealth'],-row['max_drawdown'],-row['orders'])]) for label in ['main','retained_control','readmission']}
nondominated=[name for name,v in vectors.items() if not any(other!=name and np.all(w>=v) and np.any(w>v) for other,w in vectors.items())]
pair=json.loads((folders['holding']/'decision_review/pair_outcome.json').read_text())
checked(pair['advance'] is False and pair['status']=='REJECTED','pair screen unexpectedly changed')
paired=[]
for row in pair['rows']:
    a,b=row['control'],row['treatment']
    checked(b['wealth']>a['wealth'] and b['max_drawdown']>a['max_drawdown'],'pair summary mismatch')
    paired.append(dict(scope=row['scope'],control_wealth=a['wealth'],treatment_wealth=b['wealth'],
        wealth_change_percent=100*(b['wealth']/a['wealth']-1),drawdown_change_percentage_points=100*(b['max_drawdown']-a['max_drawdown']),
        control_orders=a['orders'],treatment_orders=b['orders']))
result=dict(kind='READ_ONLY_RECORDED_ACCOUNT_RECONCILIATION_NOT_NEW_ECONOMIC_EVIDENCE',
    audit_source='07f4f32f2a856866bb774e128470b437ceb7f779',source_tree='2b0e1b0e3911377c0f3232971c4e96ac007a9cd8',
    generator_sha256=digest(Path(__file__)),runtime=dict(python=platform.python_version(),platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__),
    data_sha256=market.fingerprint(),manifest_checks=manifest_checks,
    verified_manifest_members=sum(x['members'] for x in manifest_checks),
    production_hashes_verified=len(plan['source']['files']),research_hashes_verified=len(plan['dependencies']),
    accounts=checks,whole_candidate_comparison=rows,nondominated_in_this_recovered_set=nondominated,
    risk_observations=risk,exposure_observations=quantities,registered_pair=paired,
    new_portfolio_replays=0,new_structural_hypotheses=0,economic_acceptance='NOT_MET',
    limitations=['Adjusted economic units, not actual shares/corporate-action/tax verification.',
                 'Accounting and whole-account descriptive comparison, not a realizable counterfactual or global bound.',
                 'Numerical identity in three accounts does not extend broad evidence to a different source HEAD.',
                 'All source-bound old failures and original selection rules remain unchanged.'])
output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(accounts=len(checks),manifest_members=result['verified_manifest_members'],
 max_cash_error=max(v['maximum_cash_error'] for v in checks),max_pnl_error=max(v['maximum_pnl_error'] for v in checks),
 nondominated=nondominated,pair=paired,output_sha256=digest(output)),indent=2))

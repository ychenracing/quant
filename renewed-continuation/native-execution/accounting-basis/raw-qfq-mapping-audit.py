from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

THRESHOLD = 0.10
COMMON_FIVE = {"sz300308", "sz300502", "sz300394", "sh688008", "sh603986"}

def audit(root: Path) -> dict:
    events=[]
    symbols=[]
    qdir=root/'qfq'; rdir=root/'raw'
    for qpath in sorted(qdir.glob('*.csv')):
        symbol=qpath.stem
        if symbol.startswith(('sh000','sz399')) or not (rdir/qpath.name).exists():
            continue
        symbols.append(symbol)
        q=pd.read_csv(qpath, parse_dates=['date']).set_index('date')
        r=pd.read_csv(rdir/qpath.name, parse_dates=['date']).set_index('date')
        x=r[['close']].join(q[['close']], lsuffix='_raw', rsuffix='_qfq').dropna()
        x=x[(x.close_raw>0)&(x.close_qfq>0)]
        ratio=x.close_raw/x.close_qfq
        change=ratio/ratio.shift(1)-1
        for day in change.index[change.abs()>THRESHOLD]:
            events.append({
                'symbol':symbol,
                'date':day.strftime('%Y-%m-%d'),
                'raw_close':float(x.at[day,'close_raw']),
                'qfq_close':float(x.at[day,'close_qfq']),
                'raw_to_qfq_ratio_change':float(change.at[day]),
                'common_five':symbol in COMMON_FIVE,
            })
    return {
        'classification':'STATIC_DATA_REPRESENTATION_AUDIT_NOT_CORPORATE_ACTION_IDENTIFICATION',
        'method':'join frozen raw/qfq closes by date; flag >10% day-over-day changes in raw/qfq close ratio; this detects large mapping-regime discontinuities but does not identify dividend/split/rights event type',
        'threshold_abs_ratio_change':THRESHOLD,
        'stock_symbols_with_raw_and_qfq':len(symbols),
        'events':events,
        'event_count':len(events),
        'symbols_with_events':len({e['symbol'] for e in events}),
        'common_five_event_count':sum(e['common_five'] for e in events),
        'common_five_symbols_with_events':sorted({e['symbol'] for e in events if e['common_five']}),
        'interpretation':'Large raw/qfq mapping changes are present in the frozen sample, so adjusted-unit versus actual-share accounting is not merely a theoretical distinction. Exact corporate-action cash/share treatment cannot be reconstructed from this signal alone; dedicated action metadata would be required.'
    }

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('market',type=Path); p.add_argument('-o','--output',type=Path)
    a=p.parse_args(); result=audit(a.market); text=json.dumps(result,indent=2,sort_keys=True)+'\n'
    if a.output: a.output.write_text(text)
    else: print(text,end='')

"""Bounded, credential-free Tencent acquisition; never fill missing stock rows.

Prices are research-only qfq. Raw bars are retained separately for audits. The
provider's original response is hashed; chunk overlaps must agree. This script
is not imported by the backtest or run by ordinary CI.
"""
from __future__ import annotations
import concurrent.futures
import csv
import hashlib
import json
from pathlib import Path
import time
import urllib.request

SYMBOLS = 'sz300308 sz300502 sz300394 sh688008 sh603986 sz002409 sh688072 sh688300 sz300054 sh688205 bj920045 sz300776 sh688535 sh688249 sh688347 sz300666 sh600206 sh688409 sh688361 sz300604 sh688120 sh688082 sh688498 sz002281 sh601869 sz300223 sh688825 sh688256 sh688041 sz002371 sh688012 sh688037 sh688019 sh688268'.split()
OBSERVATIONS = ['sh000300', 'sh000682', 'sz399808']
WINDOWS = [('2023-01-01','2024-06-30'),('2024-06-01','2025-12-31'),('2025-12-01','2026-09-13')]
ENDPOINT = 'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get'
OUT = Path('market-evidence')


def request(symbol: str, start: str, end: str, adjust: str) -> tuple[list, str, dict]:
    url = f'{ENDPOINT}?param={symbol},day,{start},{end},500,{adjust}'
    for attempt in range(2):
        try:
            req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=18) as res:
                raw = res.read(5_000_001)
            if len(raw)>5_000_000: raise ValueError('oversized response')
            data=json.loads(raw)
            node=data.get('data',{}).get(symbol,{})
            rows=node.get('qfqday' if adjust else 'day',[])
            if not rows and adjust and node.get('day'):
                # Newly listed, never adjusted stocks may have only raw bars.
                rows=node['day']
            if not isinstance(rows,list): raise ValueError('daily rows are not a list')
            target=OUT/'responses'/f'{symbol}_{start}_{end}_{adjust or "raw"}.json'
            target.write_bytes(raw)
            qt=node.get('qt',{})
            return rows,hashlib.sha256(raw).hexdigest(),qt if isinstance(qt,dict) else {}
        except (OSError,ValueError) as exc:
            if attempt: raise
            time.sleep(0.4)
    raise AssertionError('unreachable')


def acquire(symbol: str) -> dict:
    meta={'symbol':symbol,'role':'observation' if symbol in OBSERVATIONS else 'technology_equity','windows':[]}
    try:
        for adjust in ([''] if symbol in OBSERVATIONS else ['qfq','']):
            merged={}
            differences=[]
            for start,end in WINDOWS:
                rows,digest,quote=request(symbol,start,end,adjust)
                meta['windows'].append({'start':start,'end':end,'adjust':adjust,'response_sha256':digest,'rows':len(rows)})
                if quote: meta['quote_identity']=quote
                for row in rows:
                    date=str(row[0])
                    if not '2023-01-01'<=date<='2026-09-13': continue
                    if date in merged:
                        old=merged[date]
                        delta=max(abs(float(a)-float(b)) for a,b in zip(old[1:5],row[1:5]))
                        if delta>0.011: differences.append({'date':date,'max_price_delta':delta})
                    merged[date]=row
            if differences: raise ValueError(f'overlap mismatch: {differences[:3]}')
            if not merged: raise ValueError(f'no {adjust or "raw"} rows')
            records=[]
            for date,row in sorted(merged.items()):
                o,c,h,l=(float(v) for v in row[1:5])
                if not (0<l<=min(o,c)+0.011 and max(o,c)<=h+0.011):
                    raise ValueError(f'invalid OHLC {date}: {row[1:5]}')
                provider_volume=float(row[5])
                # Source audit: STAR is already shares; other SSE/SZSE are lots.
                # BJ has no independent unit audit, so use the smaller share
                # interpretation and mark it as a conservative assumption.
                multiplier=1 if symbol.startswith(('sh688','bj')) else 100
                vol=0 if symbol in OBSERVATIONS else provider_volume*multiplier
                records.append({'date':date,'open':o,'high':h,'low':l,'close':c,'volume':vol})
            folder=OUT/('qfq' if adjust or symbol in OBSERVATIONS else 'raw')
            dest=folder/f'{symbol}.csv'
            with dest.open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=['date','open','high','low','close','volume'])
                writer.writeheader();writer.writerows(records)
            meta[adjust or 'raw']={'path':str(dest.relative_to(OUT)),'rows':len(records),'first':records[0]['date'],'last':records[-1]['date'],'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
        meta['status']='ok'
        meta['volume_unit']='shares; BJ raw-as-shares is a conservative unverified assumption' if symbol.startswith('bj') else 'shares; STAR raw, other equity raw*100'
    except Exception as exc:
        # Preserve failed symbols in the manifest; do not silently shrink a pool.
        meta['status']='failed';meta['error']=f'{type(exc).__name__}: {exc}'
    print(json.dumps({k:v for k,v in meta.items() if k not in {'quote_identity','windows'}},ensure_ascii=False),flush=True)
    return meta


def main() -> None:
    for folder in ['qfq','raw','responses']:(OUT/folder).mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(acquire,SYMBOLS+OBSERVATIONS))
    info={'provider':'Tencent','endpoint':ENDPOINT,'requested_start':'2023-01-01','requested_end':'2026-09-13','prices':'qfq research prices; not an actual-share corporate-action ledger','records':results}
    (OUT/'manifest.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SUCCESS',sum(r['status']=='ok' for r in results),'/',len(results))

if __name__=='__main__':main()

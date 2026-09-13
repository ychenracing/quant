"""Retrieve the one incomplete BSE series; all failures remain explicit."""
import csv
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

out=Path('market-evidence');out.mkdir(exist_ok=True)
report={'symbol':'bj920045','provider':'Eastmoney','status':'failed','records':[]}
try:
    for adjustment,label in [(1,'qfq'),(0,'raw')]:
        params={'secid':'0.920045','klt':'101','fqt':str(adjustment),'beg':'20251231','end':'20260913','lmt':'500','fields1':'f1,f2,f3,f4,f5,f6','fields2':'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61'}
        url='https://push2his.eastmoney.com/api/qt/stock/kline/get?'+urllib.parse.urlencode(params)
        req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0','Referer':'https://quote.eastmoney.com/'})
        with urllib.request.urlopen(req,timeout=25) as res:raw=res.read(3_000_001)
        if len(raw)>3_000_000:raise ValueError('oversized response')
        (out/f'bj920045_{label}_response.json').write_bytes(raw)
        node=json.loads(raw).get('data') or {}
        if node.get('code')!='920045':raise ValueError('provider identity mismatch')
        rows=[]
        for line in node.get('klines',[]):
            v=line.split(',');o,c,h,l,vol,amount=map(float,v[1:7])
            if not 0<l<=min(o,c)+.011 or max(o,c)>h+.011:raise ValueError('invalid OHLC')
            rows.append([v[0],o,h,l,c,vol*100,amount])
        if len(rows)<120 or rows[-1][0]!='2026-09-11':raise ValueError(f'incomplete history: {len(rows)} rows')
        with (out/f'bj920045_{label}.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['date','open','high','low','close','volume','amount']);w.writerows(rows)
        report['records'].append({'adjustment':label,'rows':len(rows),'first':rows[0][0],'last':rows[-1][0],'response_sha256':hashlib.sha256(raw).hexdigest(),'csv_sha256':hashlib.sha256((out/f'bj920045_{label}.csv').read_bytes()).hexdigest(),'name':node.get('name')})
    report['status']='ok'
except (OSError,ValueError,KeyError,TypeError) as exc:
    report['error']=f'{type(exc).__name__}: {exc}'
(out/'supplement_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))

"""Acquire public Tencent daily bars without copying a strategy implementation.

This is an optional research data acquisition command, not a trading dependency.
Each response is retained and hashed. Unit inference must agree with raw OHLC;
missing or conflicting observations are errors, never synthetic market prices.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.parse
import urllib.request

SYMBOLS = ('002281 002371 002409 300054 300223 300308 300394 300502 300604 300666 '
           '300776 600206 601869 603986 688008 688012 688019 688037 688041 688072 '
           '688082 688120 688205 688249 688256 688268 688300 688347 688361 688409 '
           '688498 688535 688825 920045').split()
END = '2026-09-11'


def request_one(task):
    code, year, adjustment, out = task
    symbol = ('sh' if code.startswith('6') else 'bj' if code.startswith(('4', '8', '9')) else 'sz') + code
    endpoint = 'https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get'
    query = urllib.parse.urlencode({'param': f'{symbol},day,{year}-01-01,{year}-12-31,640,{adjustment}'})
    url = endpoint + '?' + query
    for attempt in range(2):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            body = urllib.request.urlopen(req, timeout=18).read()
            text = body.decode('utf-8')
            payload = json.loads(text[text.index('{'):])
            entry = payload.get('data', {}).get(symbol, {})
            rows = entry.get(adjustment + 'day', entry.get('day', []))
            if not isinstance(rows, list):
                raise ValueError('unexpected daily schema')
            tag = f'{code}-{year}-{adjustment or "raw"}'
            (out / 'responses' / (tag + '.json')).write_bytes(body)
            return {'code': code, 'year': year, 'adjustment': adjustment, 'rows': rows,
                    'url': url, 'sha256': hashlib.sha256(body).hexdigest()}
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
    return {'code': code, 'year': year, 'adjustment': adjustment, 'error': error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    if out.exists():
        raise SystemExit('output already exists; do not overwrite an evidence snapshot')
    (out / 'responses').mkdir(parents=True)
    tasks = [(s, y, a, out) for s in SYMBOLS for y in range(2023, 2027) for a in ('', 'qfq')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(request_one, tasks))
    manifest = {'source': 'Tencent newfqkline, independently acquired 2026-09-13',
                'requested_start': '2023-01-03', 'requested_end': END,
                'price_mode': 'adjusted', 'volume_unit': 'shares', 'currency': 'CNY',
                'execution_verified': False, 'files': {}, 'failures': [],
                'notes': ['Diagnostic adjusted-equivalent shares; volume rescaled by raw/adjusted close.',
                          'Historical eligibility, official price bands, and corporate-action ledger uncertified.']}
    for code in SYMBOLS:
        try:
            series = {'': {}, 'qfq': {}}
            observations = [r for r in results if r['code'] == code]
            errors = [r for r in observations if 'error' in r]
            if errors:
                raise ValueError(f'{len(errors)} failed requests: {errors[0]["error"]}')
            for r in observations:
                for row in r['rows']:
                    if '2023-01-01' <= row[0] <= END:
                        if row[0] in series[r['adjustment']] and series[r['adjustment']][row[0]] != row:
                            raise ValueError('conflicting duplicate dates')
                        series[r['adjustment']][row[0]] = row
            if not series['qfq'] or set(series['']) != set(series['qfq']):
                raise ValueError('empty history or raw/adjusted date mismatch')
            records = []
            for date in sorted(series['qfq']):
                q, raw = series['qfq'][date], series[''][date]
                if len(raw) < 9:
                    raise ValueError(f'{date}: turnover amount is absent')
                opening, close, high, low = map(float, q[1:5])
                raw_close, raw_high, raw_low = map(float, (raw[2], raw[3], raw[4]))
                volume, amount = float(raw[5]), float(raw[8]) * 10000
                scales = [s for s in (1, 100) if volume > 0 and raw_low * 0.98 <= amount / (volume * s) <= raw_high * 1.02]
                if len(scales) != 1 or min(opening, close, high, low) <= 0:
                    raise ValueError(f'{date}: unverified price/volume unit')
                records.append([date, opening, high, low, close,
                                volume * scales[0] * raw_close / close, amount])
            text = io.StringIO(); writer = csv.writer(text, lineterminator='\n')
            writer.writerow(['date', 'open', 'high', 'low', 'close', 'volume', 'amount']); writer.writerows(records)
            body = text.getvalue().encode()
            (out / f'{code}.csv').write_bytes(body)
            manifest['files'][code] = {'path': f'{code}.csv', 'sha256': hashlib.sha256(body).hexdigest(),
                  'rows': len(records), 'start': records[0][0], 'end': records[-1][0]}
        except Exception as exc:
            manifest['failures'].append({'symbol': code, 'error': str(exc)})
    (out / 'requests.json').write_text(json.dumps([{k: v for k, v in r.items() if k != 'rows'} for r in results], indent=2))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'usable_symbols': len(manifest['files']), 'failures': manifest['failures']}, indent=2))


if __name__ == '__main__':
    main()

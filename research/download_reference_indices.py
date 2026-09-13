"""Optional read-only index inputs for external native comparators, not alpha."""
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request


def fetch(task):
    symbol, year = task
    query = urllib.parse.urlencode({'param': f'{symbol},day,{year}-01-01,{year}-12-31,640,'})
    url = 'https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get?' + query
    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    raw = urllib.request.urlopen(request, timeout=25).read()
    payload = json.loads(raw.decode()[raw.decode().index('{'):])
    rows = payload['data'][symbol]['day']
    return symbol, rows, {'url': url, 'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    root = Path('market-data'); root.mkdir(exist_ok=False)
    symbols = ['sh000300', 'sh000682', 'sz399808']
    manifest = {'use': 'external comparators only; not traded by techquant', 'requests': [], 'files': {}}
    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(fetch, [(s, y) for s in symbols for y in range(2023, 2027)]))
    for symbol in symbols:
        rows = {}
        for s, observations, provenance in results:
            if s == symbol:
                manifest['requests'].append(provenance)
                for row in observations:
                    if '2023-01-01' <= row[0] <= '2026-09-11':
                        rows[row[0]] = row
        if not rows:
            raise ValueError(f'missing index {symbol}')
        path = root / (symbol + '.csv')
        with path.open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['date', 'open', 'high', 'low', 'close', 'volume', 'amount'])
            for date, row in sorted(rows.items()):
                writer.writerow([date, row[1], row[3], row[4], row[2], row[5], float(row[8]) * 10000])
        manifest['files'][symbol] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'rows': len(rows), 'start': min(rows), 'end': max(rows)}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest['files'], indent=2))


if __name__ == '__main__':
    main()

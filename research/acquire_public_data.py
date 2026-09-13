"""Acquire a pinned public price archive for local research, without credentials.

This does not import the archived strategy or make any return claim. The archive
is retained as a provenance artifact, never installed or executed.
"""
from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile

REVISION = 'df45b4d7d9ea290ae953115afbb73140270537c2'
URL = f'https://codeload.github.com/ychenracing/uquant/zip/{REVISION}'

def main():
    out = Path('public-data-evidence')
    out.mkdir(exist_ok=True)
    request = urllib.request.Request(URL, headers={'User-Agent': 'quant-research/1.0'})
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read(150_000_001)
    if len(raw) > 150_000_000:
        raise ValueError('Archive exceeds the explicit size limit')
    archive = out / 'public-source.zip'
    archive.write_bytes(raw)
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        if not any('/data/' in name and name.endswith('.csv') for name in names):
            raise ValueError('Pinned public source has no price CSVs')
    info = {'source': 'ychenracing/uquant', 'revision': REVISION,
            'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
            'purpose': 'market data provenance only; source strategy is not a benchmark',
            'files': names}
    (out / 'manifest.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in info.items() if k != 'files'}, indent=2))

if __name__ == '__main__':
    main()

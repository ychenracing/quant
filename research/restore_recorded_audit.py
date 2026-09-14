"""Losslessly restore saved audit reports using their immutable raw-account base.

This decompresses existing bytes. It neither reruns an audit nor turns the
historical reports into verification of a newer source.
"""
from pathlib import Path
import base64
import gzip
import hashlib
import json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restore(record, origin, destination):
    record, origin, destination = Path(record), Path(origin), Path(destination)
    payload = json.loads(record.read_text())
    data = json.loads(gzip.decompress(base64.b64decode(payload['payload_base64'])))
    registry = json.loads((origin/'account_registry.json').read_text())
    keys = sorted(registry)
    destination.mkdir(parents=True, exist_ok=True)
    for saved in data:
        result = saved['template']
        if (digest(origin/'account_registry.json') != result['registry_sha256']
                or digest(origin/'matrix.csv') != result['matrix_sha256']):
            raise ValueError('historical account base does not match the saved audit')
        records = []
        for index, (pnl_error, cash_error, fills) in enumerate(saved['values'],result['start']):
            key = keys[index]
            account = registry[key]['path']
            root = origin/account
            records.append({'index':index,'account_key':key,'account':account,
                'policy':Path(account).name.split('_',1)[1],
                'identity_sha256':digest(root/'identity.json'),
                'raw_manifest_sha256':digest(root/'manifest.json'),
                'max_pnl_reconciliation_error':pnl_error,'max_cash_error':cash_error,
                'actual_fills':fills,'status':'VERIFIED_RECORDED_ACCOUNT'})
        result['records'] = records
        raw = (json.dumps(result,indent=2)+'\n').encode()
        if hashlib.sha256(raw).hexdigest() != saved['original_sha256']:
            raise ValueError('restoration is not byte-identical to the historical report')
        path = destination/saved['name']
        if path.exists() and path.read_bytes() != raw:
            raise ValueError('refuse to replace a different audit report')
        path.write_bytes(raw)
    return len(data)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record',type=Path,required=True)
    parser.add_argument('--origin',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    restore(args.record,args.origin,args.output)

"""Reproduce and durably package a finite research request on a hosted runner.

The input archive is pinned by commit and SHA256. Only raw input files are
extracted; historical performance is never used as current-source evidence.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request

PREFIX='https://raw.githubusercontent.com/ychenracing/quant/b54eaaf6487005a492b23ee5bce300f93f7e41cf/published/bbc002449093e458868a8748687df023958ba283/34794779702/'
PARTS=('de8657a1b07fbee0dd8d0b76348ec25ea6cd1732cdab8123d24746fbf642698a',
       '737850897a7093afc748e019e727d2976744f641d84e9284f282392f095e7c89',
       'd02df348c0be3e03028ba127c39b0b6cd9e211f5cf412128ec98e5ff1b9baacb')


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def get_part(cache:Path,index:int):
    """At most three transport attempts, atomic publication and mandatory hashes."""
    name=f'evidence.tar.gz.part-{index:03}'
    destination=cache/name
    if destination.is_file() and digest(destination)==PARTS[index]:
        return destination
    temporary=cache/(name+'.partial')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(PREFIX+name,timeout=45) as src,temporary.open('wb') as dst:
                shutil.copyfileobj(src,dst)
            if digest(temporary)!=PARTS[index]:
                raise ValueError('frozen input archive hash mismatch')
            temporary.replace(destination)
            return destination
        except (OSError,ValueError) as error:
            temporary.unlink(missing_ok=True)
            permanent=isinstance(error,urllib.error.HTTPError) and error.code not in {408,429,500,502,503,504}
            if permanent or attempt==2:raise
            print(f'frozen part {index}: retry after {type(error).__name__}',flush=True)
            time.sleep(attempt+1)
    raise AssertionError('unreachable bounded download state')


def inputs(cache):
    cache.mkdir(parents=True,exist_ok=True)
    def get(index):
        return get_part(cache,index)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        parts=list(pool.map(get,range(len(PARTS))))
    archive=cache/'input.tar.gz'
    with archive.open('wb') as dst:
        for part in parts:
            with part.open('rb') as src:shutil.copyfileobj(src,dst)
    if digest(archive)!='6f7864e8f60001790806dcedb2fb7abc5960fb992c936ee9ea2be02069f3f353':
        raise ValueError('complete frozen input archive hash mismatch')
    with tarfile.open(archive) as source:
        members=[m for m in source.getmembers() if m.isfile() and
                 (m.name.startswith('evidence/inputs/market/') or m.name.startswith('evidence/inputs/supplement/'))]
        if not members:raise ValueError('archive does not contain frozen inputs')
        source.extractall(cache,members=members,filter='data')
    return cache/'evidence/inputs/market',cache/'evidence/inputs/supplement'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--data',type=Path)
    p.add_argument('--supplement',type=Path);p.add_argument('--families',nargs='+',choices=('expectation','leadership','admission'),required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    result={'source_commit':os.environ['QUANT_SOURCE_COMMIT'],'generator':'hosted_reproduction',
            'input_archive_commit':'b54eaaf6487005a492b23ee5bce300f93f7e41cf','families':{},'economic_acceptance':'UNVERIFIED'}
    failed=False
    try:
        data,supplement=(args.data,args.supplement) if args.data else inputs(args.output.parent/'cache')
        for family in args.families:
            root=args.output/family;root.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,'-m',f'research.{family}_study','--data',str(data),'--supplement',str(supplement)]
            with (root/'execution.log').open('w') as log:
                for task in ('selection','evaluation'):
                    call=command+['--output',str(root/task)]
                    if task=='evaluation':call+=['--selection',str(root/'selection/selection.json')]
                    subprocess.run(call,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=240)
            result['families'][family]='MEASURED_NOT_ACCEPTED'
    except Exception as exc:
        failed=True;result['failure']=repr(exc)
    finally:
        (args.output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
        files={str(f.relative_to(args.output)):digest(f) for f in sorted(args.output.rglob('*')) if f.is_file()}
        (args.output/'MANIFEST.json').write_text(json.dumps(files,indent=2)+'\n')
        with tarfile.open(args.output.parent/'research-evidence.tar.gz','w:gz') as target:
            target.add(args.output,arcname='evidence')
        (args.output.parent/'research-evidence.sha256').write_text(digest(args.output.parent/'research-evidence.tar.gz')+'  research-evidence.tar.gz\n')
    if failed:raise SystemExit(1)


if __name__=='__main__':main()

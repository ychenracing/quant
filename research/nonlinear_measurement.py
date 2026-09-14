"""Measure the existing frozen nonlinear candidates and retain their forecasts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from techquant.data import file_hash, load_market
from research import nonlinear
from research.finite_study import Study
from research.expectation_study import write_json
from research.inventory_study import package


class EvidenceStudy(Study):
    def __init__(self, root):
        super().__init__('nonlinear')
        self.root=root
        self.saved_forecasts=set()

    def identity(self):
        result=super().identity()
        directory=Path(__file__).parent
        result['measurement']={p:file_hash(directory/p) for p in ('nonlinear_measurement.py',
            'nonlinear_contract.json','requirements.txt','catalog.json','inventory_study.py')}
        return result

    def forecasts(self):
        for key,prediction in nonlinear._CACHE.items():
            name=prediction.data_sha256+'_h'+str(prediction.horizon)
            if name in self.saved_forecasts:
                continue
            root=self.root/'forecasts'/name
            context={'study':self.identity(),'cache_key':json.loads(json.dumps(key)),
                     'data_sha256':prediction.data_sha256,'horizon':prediction.horizon,
                     'universe':list(prediction.symbols),'forecast_sha256':prediction.fingerprint()}
            if root.exists():
                receipt=json.loads((root/'receipt.json').read_text())
                if receipt['identity']!=context or receipt['arrays_sha256']!=file_hash(root/'arrays.npz'):
                    raise ValueError('saved forecast evidence identity mismatch')
            else:
                root.mkdir(parents=True)
                np.savez_compressed(root/'arrays.npz',**{p:getattr(prediction,p) for p in
                    ('expected','tail','ready','price','ema10','ema20','ema60','momentum5','ret1')})
                write_json(root/'receipt.json',dict(identity=context,fits=prediction.fits,
                    arrays_sha256=file_hash(root/'arrays.npz')))
            self.saved_forecasts.add(name)

    def saved(self,*args,**kwargs):
        result=super().saved(*args,**kwargs)
        self.forecasts()
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--supplement',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    study=EvidenceStudy(args.output);started=time.monotonic()
    try:
        catalog=json.loads(Path(__file__).with_name('catalog.json').read_text())
        market=load_market(args.data,supplement=args.supplement,sectors=catalog['sectors'])
        if market.fingerprint()!='d9ded1d933f292933aa09145bc36a5696421c43dbf266fce018a34ef0b9f3f9b':
            raise ValueError('frozen input identity mismatch')
        study.select(market,catalog,args.output/'selection')
        study.evaluate(market,catalog,args.output/'selection/selection.json',args.output/'evaluation')
        write_json(args.output/'status.json',dict(identity=study.identity(),
            status='MEASURED_NOT_ECONOMIC_ACCEPTANCE',economic_acceptance='UNVERIFIED'))
    except Exception as exc:
        write_json(args.output/'failure.json',dict(error=repr(exc),identity=study.identity()))
        raise
    finally:
        study.forecasts()
        write_json(args.output/'execution.json',dict(identity=study.identity(),seconds=time.monotonic()-started))
        package(args.output)


if __name__=='__main__':
    main()

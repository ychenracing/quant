"""Allocate an already-required cut without increasing marked support risk."""
from dataclasses import asdict,dataclass
from pathlib import Path
import numpy as np
from techquant.config import Config
from techquant.data import Market,file_hash
from research.observed_admission_completion import Owner as Parent,Parameters as ParentParameters
from research.quantity_obligation import SupportIntent,preserve_trace,verify_trace


@dataclass(frozen=True)
class Parameters:
    prefer_supported_strength: bool = True

    def __post_init__(self):
        if type(self.prefer_supported_strength) is not bool:
            raise ValueError('prefer_supported_strength must be a registered boolean')


def grid():
    return [Parameters(False),Parameters(True)]


def retain_supported_strength(upper,proportional,prices,distances,scores):
    upper,base,price,distance,score=(np.asarray(v,dtype=float) for v in
                                   (upper,proportional,prices,distances,scores))
    if (upper.ndim!=1 or any(v.shape!=upper.shape for v in (base,price,distance,score))
            or not all(np.isfinite(v).all() for v in (upper,base,price,distance))
            or np.any(upper<0) or np.any(base<0) or np.any(base>upper)
            or np.any(distance<0) or np.any((upper>0)&(price<=0))):
        raise ValueError('invalid protected retention resources')
    result=base.copy();ids=np.flatnonzero(upper>0)
    if len(ids)!=2 or not np.isfinite(score[ids]).all() or score[ids[0]]==score[ids[1]]:
        return result
    strong=ids[int(np.argmax(score[ids]))];weak=ids[int(np.argmin(score[ids]))]
    if distance[strong]/price[strong]>distance[weak]/price[weak]:
        return result
    value=min((upper[strong]-base[strong])*price[strong],base[weak]*price[weak])
    if value<=0:
        return result
    result[strong]+=value/price[strong];result[weak]-=value/price[weak]
    result=np.clip(result,0.,upper)
    # Only floating-point tolerance is permitted; never create resource authority.
    scale=max(1.,float(base@price),float(base@distance))
    if (abs(float((result-base)@price))>scale*1e-12
            or float((result-base)@distance)>scale*1e-12):
        raise AssertionError('retention changed the required liquidation or raised risk')
    return result


class ProtectedSupport(SupportIntent):
    def _reduce_exposure(self,o,desired,price,cap,exposure):
        proportional=super()._reduce_exposure(o,desired,price,cap,exposure)
        distance=np.maximum(price-self.stop,.02*price)
        scores=self.features.score[o.session]
        selected=retain_supported_strength(desired,proportional,price,distance,scores)
        self.retention_trace.append({'kind':'PROTECTED_RETENTION','date':o.date,
            'session':o.session,'symbols':list(self.market.symbols),'nav':o.nav,
            'actual_units':o.units.tolist(),'individual_upper_units':desired.tolist(),
            'proportional_units':proportional.tolist(),'selected_units':selected.tolist(),
            'prices':price.tolist(),'support_risk_distance':distance.tolist(),
            'scores':[float(s) if np.isfinite(s) else None for s in scores],
            'prior_ceiling':[float(x) if np.isfinite(x) else None for x in self.reduction_ceiling],
            'exit_pending':self.exit_pending.tolist(),'cap':cap,
            'proportional_notional':float(proportional@price),'selected_notional':float(selected@price),
            'proportional_risk':float(proportional@distance),'selected_risk':float(selected@distance),
            'applied':not np.array_equal(selected,proportional)})
        return selected


class Owner(Parent):
    def __init__(self,market:Market,parameters:Parameters,*,config:Config|None=None):
        if type(parameters) is not Parameters:
            raise ValueError('only the registered retention comparison is supported')
        super().__init__(market,ParentParameters(),config=config)
        self.retention_parameters=parameters
        if parameters.prefer_supported_strength:
            original=self.inner
            inner=ProtectedSupport(market,original.params,config=original.config)
            inner.features,inner.atr,inner.support,inner.ready=(original.features,original.atr,
                                                            original.support,original.ready)
            inner.retention_trace=self.trace
            self.inner=inner

    def identity(self):
        return {'name':'protected_retention','parameters':asdict(self.retention_parameters),
            'implementation_sha256':file_hash(Path(__file__)),
            'contract_sha256':file_hash(Path(__file__).with_name('protected_retention_contract.json')),
            'parent_definition':super().identity(),'data_sha256':self.market.fingerprint(),
            'status':'RESEARCH_NOT_ACCEPTED'}

"""Joint sizing of qualified additions inside the original protective owner."""
from dataclasses import asdict,dataclass
from pathlib import Path
import numpy as np
from techquant.config import Config
from techquant.data import Market,file_hash
from research.observed_admission_completion import Owner as Parent,Parameters as ParentParameters
from research.quantity_obligation import Owner as QuantityOwner,SupportIntent,preserve_trace,verify_trace
from research.joint_resources import allocate_additions


@dataclass(frozen=True)
class Parameters:
    joint: bool = True

    def __post_init__(self):
        if type(self.joint) is not bool:raise ValueError('joint must be a registered boolean')


def grid():return [Parameters(False),Parameters(True)]


class JointSupport(SupportIntent):
    def _allocate(self,o,desired,price,initial,allowed,cap):
        if not np.allclose(desired,o.units,rtol=0,atol=1e-10):
            raise AssertionError('joint additions cannot offset a protective reduction')
        held=o.units>1e-10
        stops=np.where(held,self.stop,np.maximum(initial,self.pending_stop))
        distance=np.maximum(price-stops,.02*price)
        density=np.divide(distance,price,out=np.ones_like(price),where=price>0)
        risk_limit=self._risk_limit(o,cap)
        room=max(0.,risk_limit-float(o.units@distance))
        cash=max(0.,min(.99*o.cash,cap*o.nav-float(o.units@price)))
        symbol_cap=max(self.config.single_cap,1/len(held))
        sector_cap=self.config.sector_cap if len(set(self.features.sectors))>1 else 1.
        groups=self.features.sectors
        sector_room={g:max(0.,sector_cap*o.nav-float((o.units*price)[np.array(groups)==g].sum()))
                     for g in sorted(set(groups))}
        upper=np.maximum(0.,symbol_cap*o.nav-o.units*price)
        eligible=self._funding_eligible(o,held,allowed,price,stops)
        slots=max(0,min(self.params.positions,len(held))-int(held.sum()))
        values=allocate_additions(self.features.score[o.session],density,upper,groups,sector_room,
                    held,eligible,self.market.symbols,cash=cash,risk=room,slots=slots,
                    nav=o.nav,held_band=self.config.trade_band)
        addition=np.divide(values,price,out=np.zeros_like(values),where=price>0)
        active=addition>1e-10
        self.pending_stop[active]=np.maximum.reduce([self.pending_stop[active],initial[active],self.stop[active]])
        self.joint_trace.append({'kind':'JOINT_RESOURCE_FUNDING','session':o.session,'date':o.date,
            'symbols':list(self.market.symbols),'actual_cash':o.cash,'nav':o.nav,
            'actual_units':o.units.tolist(),'cash_authority':cash,'risk_authority':risk_limit,
            'remaining_risk':room,'risk_density':density.tolist(),'notional_additions':values.tolist(),
            'qualified_indices':[int(j) for j in np.flatnonzero(eligible)],'available_slots':slots,
            'held_fresh_competition':bool(np.any(eligible&held) and np.any(eligible&~held) and slots>0),
            'funded_legs':int(active.sum()),'funded_stop_risk':float(addition@distance),
            'note':'requests only; actual execution is owned by the unchanged engine'})
        if values.sum()>cash+1e-7 or float(addition@distance)>room+1e-7:
            raise AssertionError('joint additions exceeded original cash or stop-risk room')
        return desired+addition


class Owner(Parent):
    def __init__(self,market:Market,parameters:Parameters,*,config:Config|None=None):
        if type(parameters) is not Parameters:raise ValueError('only the fixed joint funding pair is supported')
        super().__init__(market,ParentParameters(),config=config)
        self.joint_parameters=parameters
        if parameters.joint:
            original=self.inner;inner=JointSupport(market,original.params,config=original.config)
            inner.features,inner.atr,inner.support,inner.ready=(original.features,original.atr,
                                                             original.support,original.ready)
            inner.joint_trace=self.trace;self.inner=inner

    def decide(self,o):
        if not self.joint_parameters.joint:return super().decide(o)
        # One joint funding pass replaces only the separate fresh-completion pass.
        # Economic protection latches and executable declaration rounding remain.
        return QuantityOwner.decide(self,o)

    def identity(self):
        root=Path(__file__).parent
        return {'name':'joint_resource_funding','parameters':asdict(self.joint_parameters),
            'implementation_sha256':file_hash(Path(__file__)),
            'primitive_sha256':file_hash(root/'joint_resources.py'),
            'contract_sha256':file_hash(root/'joint_funding_contract.json'),
            'parent_definition':super().identity(),'data_sha256':self.market.fingerprint(),
            'status':'RESEARCH_NOT_ACCEPTED'}

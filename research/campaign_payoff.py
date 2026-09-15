"""Learn arithmetic cash payoffs only after an entire campaign has settled.

This learner never creates execution labels from a future fixed-window mark.
Its reference campaigns are selected observations, not unbiased counterfactuals.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class SettledCampaign:
    formation: int
    settled: int
    symbol: int
    payoff: float


def learn_campaign_payoff(features, campaigns, *, refit: int):
    """Issue one fixed ridge estimate using only settled reference episodes.

    NaN means original-priority fallback. The mean cash payoff retains magnitude,
    including large wins; a probability of winning is not the learned objective.
    Each actual campaign appears once, regardless of its fills or duration.
    """
    x = np.asarray(features, dtype=float)
    if (type(refit) is not int or refit < 1 or x.ndim != 3
            or min(x.shape) < 1 or np.isinf(x).any()):
        raise ValueError('need a nonempty feature cube and positive integer refit')
    rows = list(campaigns)
    seen = set()
    for r in rows:
        if (type(r) is not SettledCampaign
                or any(type(v) is not int for v in (r.formation,r.settled,r.symbol))
                or r.formation < 0 or r.settled <= r.formation+1
                or not 0 <= r.symbol < x.shape[1]
                or type(r.payoff) not in (int,float) or not math.isfinite(r.payoff)):
            raise ValueError('invalid settled campaign or execution chronology')
        key=(r.formation,r.symbol)
        if key in seen:
            raise ValueError('duplicate campaign origin')
        seen.add(key)
    rows.sort(key=lambda r:(r.settled,r.formation,r.symbol))
    predictions=np.full(x.shape[:2],np.nan)
    fits=[]; cursor=0; available=[]; beta=None
    dimensions=x.shape[2]
    center=np.zeros(dimensions);scale=np.ones(dimensions);mean=0.
    for i in range(len(x)):
        while cursor < len(rows) and rows[cursor].settled <= i:
            r=rows[cursor]
            # A settled origin must already exist in this causal feature prefix.
            if np.isfinite(x[r.formation,r.symbol]).all():
                available.append(r)
            cursor+=1
        if i % refit == 0:
            beta=None
            if (len(available)>=2*dimensions
                    and len({r.symbol for r in available})>=2):
                xx=np.array([x[r.formation,r.symbol] for r in available])
                yy=np.array([r.payoff for r in available])
                center=xx.mean(axis=0);scale=xx.std(axis=0,ddof=0)
                scale=np.where(scale>1e-12,scale,1.)
                xx=(xx-center)/scale;mean=float(yy.mean())
                coefficient=np.linalg.solve(xx.T@xx/len(yy)+np.eye(dimensions),
                                             xx.T@(yy-mean)/len(yy))
                if np.isfinite(coefficient).all() and np.any(coefficient!=0):
                    beta=coefficient
            fits.append({'session':i,'samples':len(available),
                'distinct_symbols':len({r.symbol for r in available}),
                'first_formation_session':min((r.formation for r in available),default=None),
                'last_formation_session':max((r.formation for r in available),default=None),
                'latest_settlement_session':max((r.settled for r in available),default=-1),
                'coefficients':None if beta is None else beta.tolist(),
                'feature_center':None if beta is None else center.tolist(),
                'feature_scale':None if beta is None else scale.tolist(),
                'mean_settled_payoff':None if beta is None else mean})
        if beta is not None:
            valid=np.isfinite(x[i]).all(axis=1)
            predictions[i,valid]=((x[i,valid]-center)/scale)@beta+mean
    return predictions,fits

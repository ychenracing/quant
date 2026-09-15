"""Progressive positive-resource funding, not a return-maximization oracle."""
from __future__ import annotations
import numpy as np


def progressive_funding(matrix, limits, direction):
    """Fill simultaneously; freeze only legs consuming each saturated resource."""
    a,b,w=(np.asarray(x,dtype=float) for x in (matrix,limits,direction))
    if (a.ndim!=2 or b.shape!=(a.shape[0],) or w.shape!=(a.shape[1],)
            or not a.size or not all(np.isfinite(x).all() for x in (a,b,w))
            or np.any(a<0) or np.any(b<0) or np.any(w<=0) or np.any(a.sum(axis=0)<=0)):
        raise ValueError('funding needs finite nonnegative bounded resources and positive directions')
    scale=np.maximum(1.,b)
    constraints=a/scale[:,None];capacity=b/scale
    value=np.zeros(len(w));active=np.ones(len(w),dtype=bool);w=w/w.max()
    for _ in range(len(w)+1):
        if not active.any():break
        vector=np.where(active,w,0.);usage=constraints@vector
        available=np.maximum(0.,capacity-constraints@value)
        step=np.divide(available,usage,out=np.full(len(b),np.inf),where=usage>0)
        amount=float(step.min())
        if not np.isfinite(amount):raise ValueError('unbounded funding leg')
        value+=amount*vector
        bound=(usage>0)&(step<=amount+1e-12*max(1.,amount))
        frozen=(constraints[bound]>0).any(axis=0)&active
        if not frozen.any():raise AssertionError('progressive funding made no finite progress')
        active[frozen]=False
    if active.any() or np.any(constraints@value>capacity+1e-11):
        raise AssertionError('joint funding exceeds its declared resource')
    return value


def allocate_additions(scores,density,headroom,groups,sector_room,held,eligible,symbols,
                       *,cash,risk,slots,nav,held_band):
    """Keep original membership priority; jointly size only material additions.

    A dropped sub-material leg is never reconsidered at the same close. Removing
    a new leg releases its reserved slot to the next originally eligible name.
    """
    score,distance,upper=(np.asarray(x,dtype=float) for x in (scores,density,headroom))
    held,eligible=np.asarray(held),np.asarray(eligible)
    n=len(score) if score.ndim==1 else 0
    if (not n or any(x.shape!=(n,) for x in (distance,upper,held,eligible))
            or held.dtype!=bool or eligible.dtype!=bool or len(groups)!=n or len(symbols)!=n
            or len(set(symbols))!=n or type(slots) is not int or slots<0
            or not np.isfinite([cash,risk,nav,held_band]).all()
            or min(cash,risk,held_band)<0 or nav<=0 or held_band>=1
            or not np.isfinite(distance).all() or np.any(distance<=0)
            or not np.isfinite(upper).all() or np.any(upper<0)
            or any(not np.isfinite(sector_room.get(g,np.nan)) or sector_room[g]<0 for g in groups)
            or np.any(eligible&(~np.isfinite(score)|(score<=0)))):
        raise ValueError('invalid original funding resources or qualification mask')
    result=np.zeros(n)
    if cash<=0 or risk<=0:return result
    order=sorted(np.flatnonzero(eligible),key=lambda j:(-score[j],symbols[j]))
    excluded=set()
    for _ in range(len(order)+1):
        chosen=[];new=0
        for j in order:
            if j in excluded:continue
            if not held[j]:
                if new>=slots:continue
                new+=1
            chosen.append(j)
        if not chosen:return result
        k=len(chosen)
        matrix=[np.ones(k),distance[chosen],*np.eye(k)]
        limits=[cash,risk,*upper[chosen]]
        for g in sorted(set(groups)):
            matrix.append(np.array([groups[j]==g for j in chosen],dtype=float))
            limits.append(sector_room[g])
        values=progressive_funding(np.array(matrix),limits,score[chosen]/distance[chosen])
        floors=np.array([held_band if held[j] else .01 for j in chosen])*nav
        failing=[j for j,v,f in zip(chosen,values,floors,strict=True) if v<f or v<=1e-10]
        if not failing:
            result[chosen]=values
            return result
        excluded.add(failing[-1])
    raise AssertionError('materiality selection failed to terminate')

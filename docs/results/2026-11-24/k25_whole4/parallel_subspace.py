"""Multigoal extension of old-axis projection, with no instruction-label routing."""
import numpy as np

RCOND=1e-6


def helmert(n):
    h=np.zeros((n,n-1),dtype=np.float64)
    for j in range(1,n):
        scale=np.sqrt(j*(j+1))
        h[:j,j-1]=1/scale
        h[j,j-1]=-j/scale
    return h


def fields(parent,late):
    p,f=np.asarray(parent,dtype=np.float64),np.asarray(late,dtype=np.float64)
    pc,fc=p-p.mean(0),f-f.mean(0)
    pn,fn=np.linalg.norm(pc[...,:7]),np.linalg.norm(fc[...,:7])
    return dict(parent_norm=float(pn),late_norm=float(fn),ratio=float(pn/max(fn,1e-30)),
        mu_shift_over_parent=float(np.linalg.norm((p.mean(0)-f.mean(0))[...,:7])/max(pn,1e-30)))


def project(parent,late,actual,alpha):
    p,f,q=(np.asarray(a,dtype=np.float64) for a in (parent,late,actual))
    assert p.shape==f.shape and p.ndim==3 and p.shape[0]>=2 and q.shape==p.shape[1:] and p.shape[-1]>=7
    assert all(np.isfinite(a).all() for a in (p,f,q)) and np.isfinite(alpha) and alpha>0
    pc=p-p.mean(0)
    residual=q-f.mean(0)
    basis=np.einsum('ntd,nk->tdk',pc,helmert(len(p)))
    active=basis[:,:7,:].reshape(-1,len(p)-1)
    real=residual[:,:7].reshape(-1)
    u,s,vh=np.linalg.svd(active,full_matrices=False)
    retained=(s>RCOND*s.max()) & (s>0)
    weights=np.divide(u.T@real,s,out=np.zeros_like(s),where=retained)
    coefficients=vh.T@weights
    correction=np.einsum('tdk,k->td',basis,coefficients)
    short=correction[:,:7].reshape(-1)
    remainder=real-short
    rn,pn=np.linalg.norm(real),np.linalg.norm(pc[:,:,:7])
    axes=pc[:,:,:7].reshape(len(p),-1)
    cosines=(axes@real)/np.maximum(np.linalg.norm(axes,axis=1)*rn,1e-30)
    out=p.mean(0)+alpha*correction
    diag=dict(rank=int(retained.sum()),singular_values=s.tolist(),coefficients=coefficients.tolist(),
        condition=float(s[0]/s[retained][-1]) if retained.any() else 0.,
        parent_norm=float(pn),late_norm=float(np.linalg.norm((f-f.mean(0))[:,:,:7])),
        residual_norm=float(rn),projected_norm=float(np.linalg.norm(short)),
        scaled_projection_over_parent=float(alpha*np.linalg.norm(short)/max(pn,1e-30)),
        off_axis_fraction=float(np.linalg.norm(remainder)/max(rn,1e-30)),
        orthogonality_error=float(np.linalg.norm(active.T@remainder)/max(np.linalg.norm(active)*rn,1e-30)),
        old_axis_cosines=cosines.tolist())
    return out,diag

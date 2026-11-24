"""Full K15-style actual-query residual at the precision used by parallel4."""
import numpy as np
import parallel_subspace as reference


def whole(parent,late,actual,alpha):
    p,f,q=(np.asarray(x,dtype=np.float64) for x in (parent,late,actual))
    projected,diag=reference.project(p,f,q,alpha)
    mu=p.mean(0)
    residual=q-f.mean(0)
    output=mu+alpha*residual
    removed=output-projected
    pn=max(diag['parent_norm'],1e-30)
    diag.update(executed_residual_norm7=float(alpha*np.linalg.norm(residual[:,:7])),
        scaled_full_over_parent=float(alpha*np.linalg.norm(residual[:,:7])/pn),
        scaled_removed_over_parent=float(np.linalg.norm(removed[:,:7])/pn),
        removed_norm_full32=float(np.linalg.norm(removed)))
    return output,diag

import inspect
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/k25'))
import parallel_subspace as m


class SubspaceProjection(unittest.TestCase):
    def test_two_old_goals_reduce_to_scalar_parallel_formula_including_padding(self):
        rng=np.random.default_rng(14);p=rng.normal(size=(2,5,32));f=rng.normal(size=p.shape);q=f[1]
        mu=p.mean(0);d=(p[0]-p[1])/2;r=q-f.mean(0)
        coefficient=(r[:,:7]*d[:,:7]).sum()/(d[:,:7]**2).sum()
        out,diag=m.project(p,f,q,2.959)
        np.testing.assert_allclose(out,mu+2.959*coefficient*d,rtol=1e-12,atol=1e-12)
        self.assertEqual(diag['rank'],1)

    def test_three_old_goals_reconstruct_their_own_outputs(self):
        rng=np.random.default_rng(26);p=rng.normal(size=(3,6,32))
        for q in p:
            out,diag=m.project(p,p,q,1.)
            np.testing.assert_allclose(out,q,rtol=1e-12,atol=1e-12)
            self.assertEqual(diag['rank'],2)

    def test_permuting_reference_labels_does_not_change_projection(self):
        rng=np.random.default_rng(24);p=rng.normal(size=(3,5,32));f=rng.normal(size=p.shape);q=rng.normal(size=(5,32))
        a,_=m.project(p,f,q,1.5);b,_=m.project(p[[2,0,1]],f[[2,0,1]],q,1.5)
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)

    def test_rank_deficiency_and_zero_axis_remain_finite(self):
        p=np.zeros((3,2,32));p[:, :,0]=np.array([-1,0,1])[:,None]
        q=np.zeros((2,32));q[:,1]=5
        out,diag=m.project(p,p,q,1.)
        np.testing.assert_allclose(out,0,atol=1e-12);self.assertEqual(diag['rank'],1)
        out,diag=m.project(np.zeros_like(p),np.zeros_like(p),q,1.)
        self.assertTrue(np.isfinite(out).all());self.assertEqual(diag['rank'],0)

    def test_no_goal_label_is_an_argument_to_the_projector(self):
        self.assertEqual(list(inspect.signature(m.project).parameters),['parent','late','actual','alpha'])


if __name__=='__main__':unittest.main()

from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/k25'))
import whole_residual as m


class FullResidualControl(unittest.TestCase):
    def fixture(self):
        p=np.zeros((3,5,32));p[0,:,0]=1.;p[1,:,1]=1.;p[2,:,:2]=-1.
        f=np.zeros_like(p);q=np.zeros((5,32));q[:,2]=2.;q[:,7:]=3.
        return p,f,q

    def test_whole_preserves_active_off_axis_signal_and_padding(self):
        p,f,q=self.fixture();out,d=m.whole(p,f,q,1.5)
        projected,_=m.reference.project(p,f,q,1.5)
        np.testing.assert_array_equal(projected,0.)
        np.testing.assert_array_equal(out,1.5*q)
        self.assertGreater(d['scaled_removed_over_parent'],0.)

    def test_only_removed_component_differs_when_residual_also_contains_old_axes(self):
        p,f,q=self.fixture();q[:,0]=.5;q[:,1]=-.25
        out,_=m.whole(p,f,q,2.)
        projected,_=m.reference.project(p,f,q,2.)
        expected=np.zeros_like(q);expected[:,2]=4.;expected[:,7:]=6.
        np.testing.assert_allclose(out-projected,expected,rtol=1e-12,atol=1e-12)

    def test_zero_actual_residual_gives_parent_mean(self):
        rng=np.random.default_rng(15);p=rng.normal(size=(3,5,32));f=rng.normal(size=p.shape)
        out,_=m.whole(p,f,f.mean(0),1.635)
        np.testing.assert_allclose(out,p.mean(0),rtol=1e-12,atol=1e-12)


if __name__=='__main__':unittest.main()

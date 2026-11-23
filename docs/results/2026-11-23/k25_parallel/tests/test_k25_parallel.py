import ast
from collections import defaultdict
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/k25"))
import parallel_protocol as p


class LockedProtocol(unittest.TestCase):
    def test_only_one_method_four_models_all_paired_starts(self):
        keys=set()
        for profile in p.PROFILES:
            for lane in range(8):
                new=p.expected(profile,lane)
                self.assertFalse(keys.intersection(new))
                keys.update(new)
        self.assertEqual(len(keys),160)
        self.assertEqual(p.WINDOW,(0,150))
        self.assertEqual(p.HORIZON,400)

    def test_k21_uses_upper_median_not_average_of_middle_pair(self):
        self.assertEqual(p.upper_median([1.,2.,4.,7.]),4.)


@unittest.skipUnless(os.environ.get("K25_REAL_PROJECTION_TESTS") == "1","Original tensor formula checked in project server environment")
class OriginalProjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch
        path=Path(__file__).resolve().parent.parent/"history/k22.py"
        tree=ast.parse(path.read_text())
        lab=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == "Lab2")
        method=next(n for n in lab.body if isinstance(n,ast.FunctionDef) and n.name == "combine2")
        ns={}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(path),"exec"),ns)
        cls.method=staticmethod(ns["combine2"])

    def fixture(self,polarity):
        t=self.torch
        axis=t.zeros(1,50,32)
        axis[...,0]=1.
        axis[...,7:]=100.
        residual=t.zeros_like(axis)
        residual[...,0]=polarity*2.
        residual[...,7:]=-900.
        vals={"RA":axis,"RB":-axis,"FA":residual,"FB":-residual,"FQ":residual}
        rt=SimpleNamespace(query=lambda role,prefix,x,time:vals[prefix],vdtype=t.float32)
        obj=SimpleNamespace(rt=rt,torch=t,alpha=3.)
        pre={tuple(key):key for key in vals}
        return obj,pre,axis

    def test_scalar_uses_only_executed_dims_but_scales_full_old_axis(self):
        obj,pre,axis=self.fixture(1.)
        out=self.method(obj,"par",None,pre,"Q",None,None,defaultdict(list))
        self.assertTrue(self.torch.equal(out,6.*axis))

    def test_actual_residual_can_select_negative_old_direction(self):
        obj,pre,axis=self.fixture(-1.)
        out=self.method(obj,"par",None,pre,"Q",None,None,defaultdict(list))
        self.assertTrue(self.torch.equal(out,-6.*axis))


if __name__ == "__main__":
    unittest.main()

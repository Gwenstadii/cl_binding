import hashlib
import os
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/k25"))
from anchor_tensor_hash import tensors_hash

torch = None
if os.environ.get("K25_REAL_TENSOR_TESTS") == "1":
    import torch


@unittest.skipIf(torch is None,"Real tensor checks run in the project server environment")
class TensorStorageHash(unittest.TestCase):
    def test_bfloat16_hashes_storage_without_numpy_dtype_conversion(self):
        tensor = torch.tensor([1.,-2.5,0.],dtype=torch.bfloat16)
        expected = hashlib.sha256(b"v"+struct.pack("<3H",0x3f80,0xc020,0)).hexdigest()
        self.assertEqual(tensors_hash({"v":tensor}),expected)

    def test_float64_delta_fingerprint_keeps_legacy_bytes(self):
        values = {"b":torch.tensor([.3,-.6],dtype=torch.float64),"a":torch.tensor([[1.,2.]],dtype=torch.float64)}
        expected = hashlib.sha256(b"".join(n.encode()+v.numpy().tobytes() for n,v in sorted(values.items()))).hexdigest()
        self.assertEqual(tensors_hash(values),expected)


if __name__ == "__main__":
    unittest.main()

import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from stage1_dual_then_back_run import validate_config


class BackPlanTests(unittest.TestCase):
    def test_fixed_plan_and_reject_wrong_parent_budget_replay_or_partition(self):
        config=json.loads((Path(__file__).resolve().parents[1]/'configs/stage1_dual_right_then_back_v1.json').read_text())
        validate_config(config)
        for section,key,value in [('sequential','initial_checkpoint','outputs/r0/dual_left_front_20261023_01/checkpoints/012000'),
                                  ('sequential','initial_model_sha256','wrong'),('sequential','replay',True),
                                  ('training','max_steps',18000),('evaluation','paired_indices',list(range(30,50))),
                                  ('data','normalization_source','datasets/stage1_single_back_train_aligned_v1')]:
            changed=copy.deepcopy(config);changed[section][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_config(changed)


if __name__=='__main__':
    unittest.main()

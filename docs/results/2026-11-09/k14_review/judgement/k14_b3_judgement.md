状态数 26（左 11，右 15）；T0 逐步复现 J：26/26

| 组 | 终态 | 左终态 | 右终态 | 分叉后曾成功 | 左 | 右 |
|---|---:|---:|---:|---:|---:|---:|
| T0_cand | 13 | 7 | 6 | 17 | 11 | 6 |
| T1_z | 12 | 6 | 6 | 17 | 11 | 6 |
| T2_rotgrip | 13 | 7 | 6 | 17 | 11 | 6 |
| T3_nonhoriz | 12 | 6 | 6 | 17 | 11 | 6 |
| T4_xy | 12 | 6 | 6 | 17 | 11 | 6 |
| T5_full | 14 | 7 | 7 | 18 | 11 | 7 |
| T1r_z | 12 | 6 | 6 | 17 | 11 | 6 |
| T2r_rotgrip | 12 | 7 | 5 | 16 | 11 | 5 |
| T3r_nonhoriz | 13 | 7 | 6 | 17 | 11 | 6 |

| 比较 | 指标 | 差值 | 95% CI | 明确更好 | 接近 | 增/减 |
|---|---|---:|---|---|---|---|
| T1_z − T0_cand | term | -0.038 | [-0.120, +0.000] |  | 是 | 0/1 |
| T1_z − T0_cand | ever | +0.000 | [+0.000, +0.000] |  | 是 | 0/0 |
| T2_rotgrip − T0_cand | term | +0.000 | [-0.111, +0.111] |  | 是 | 1/1 |
| T2_rotgrip − T0_cand | ever | +0.000 | [+0.000, +0.000] |  | 是 | 0/0 |
| T3_nonhoriz − T0_cand | term | -0.038 | [-0.120, +0.000] |  | 是 | 0/1 |
| T3_nonhoriz − T0_cand | ever | +0.000 | [+0.000, +0.000] |  | 是 | 0/0 |
| T4_xy − T0_cand | term | -0.038 | [-0.167, +0.083] |  | 是 | 1/2 |
| T4_xy − T0_cand | ever | +0.000 | [-0.111, +0.111] |  | 是 | 1/1 |
| T5_full − T0_cand | term | +0.038 | [-0.125, +0.200] |  | 是 | 3/2 |
| T5_full − T0_cand | ever | +0.038 | [-0.083, +0.167] |  | 是 | 2/1 |
| T1_z − T1r_z | term | +0.000 | [+0.000, +0.000] |  | 是 | 0/0 |
| T1_z − T1r_z | ever | +0.000 | [+0.000, +0.000] |  | 是 | 0/0 |
| T2_rotgrip − T2r_rotgrip | term | +0.038 | [-0.087, +0.167] |  | 是 | 2/1 |
| T2_rotgrip − T2r_rotgrip | ever | +0.038 | [+0.000, +0.120] |  | 是 | 1/0 |
| T3_nonhoriz − T3r_nonhoriz | term | -0.038 | [-0.167, +0.083] |  | 是 | 1/2 |
| T3_nonhoriz − T3r_nonhoriz | ever | +0.000 | [-0.111, +0.111] |  | 是 | 1/1 |
| T5_full − T3_nonhoriz | term | +0.077 | [-0.074, +0.217] |  | 是 | 3/1 |
| T5_full − T3_nonhoriz | ever | +0.038 | [-0.083, +0.167] |  | 是 | 2/1 |
| T5_full − T4_xy | term | +0.077 | [+0.000, +0.185] |  | 是 | 2/0 |
| T5_full − T4_xy | ever | +0.038 | [+0.000, +0.120] |  | 是 | 1/0 |
| T5_full − T1_z | term | +0.077 | [-0.074, +0.217] |  | 是 | 3/1 |
| T5_full − T1_z | ever | +0.038 | [-0.083, +0.167] |  | 是 | 2/1 |
| T5_full − T2_rotgrip | term | +0.038 | [-0.125, +0.200] |  | 是 | 3/2 |
| T5_full − T2_rotgrip | ever | +0.038 | [-0.083, +0.167] |  | 是 | 2/1 |

```json
{
 "verdicts": {
  "channel_z_carries_content": [],
  "channel_rotgrip_carries_content": [],
  "channel_nonhoriz_carries_content": [],
  "xy_carries_content": [],
  "not_channel_separable": false
 },
 "diagnostics": {
  "T0_cand": {
   "old_cos_full": 0.7138005422694342,
   "norm_over_old": 1.0000000143492664,
   "cap_hits": 0,
   "clip_mean": 0.48832126947468313,
   "min_z_rel_median": 0.06866634144398931
  },
  "T1_z": {
   "delta_norm_over_old": 0.17585420567413856,
   "old_cos_full": 0.7275386723024505,
   "norm_over_old": 1.0032484795366015,
   "cap_hits": 0,
   "clip_mean": 0.4764098411839445,
   "min_z_rel_median": 0.0835063541466009
  },
  "T2_rotgrip": {
   "delta_norm_over_old": 0.6932965981549231,
   "old_cos_full": 0.9175513273683088,
   "norm_over_old": 0.9253661330257144,
   "cap_hits": 0,
   "clip_mean": 0.3882897948273914,
   "min_z_rel_median": 0.0688383351702877
  },
  "T3_nonhoriz": {
   "delta_norm_over_old": 0.7421120547015091,
   "old_cos_full": 0.9358659957136427,
   "norm_over_old": 0.9266674063823841,
   "cap_hits": 0,
   "clip_mean": 0.3885232852602533,
   "min_z_rel_median": 0.08233314493516497
  },
  "T4_xy": {
   "delta_norm_over_old": 0.3760540321469307,
   "old_cos_full": 0.782042610006673,
   "norm_over_old": 1.064863219084563,
   "cap_hits": 0,
   "clip_mean": 0.4655525309220096,
   "min_z_rel_median": 0.08066933879452476
  },
  "T5_full": {
   "old_cos_full": 1.0000000108932627,
   "norm_over_old": 1.0,
   "cap_hits": 0,
   "clip_mean": 0.3727242227364762,
   "min_z_rel_median": 0.0809156422537688
  },
  "T1r_z": {
   "rand_cos_to_delta": 0.00017838466322116022,
   "rand_norm_over_delta": 1.000000007876328,
   "delta_norm_over_old": 0.16287754176762598,
   "old_cos_full": 0.6646643131971359,
   "norm_over_old": 1.014414570587022,
   "cap_hits": 0,
   "clip_mean": 0.48050130125366547,
   "min_z_rel_median": 0.0848643666761657
  },
  "T2r_rotgrip": {
   "rand_cos_to_delta": 0.0007611321557541722,
   "rand_norm_over_delta": 1.0000000061733383,
   "delta_norm_over_old": 0.7156956489597048,
   "old_cos_full": 0.5696077101996967,
   "norm_over_old": 1.2314849974853652,
   "cap_hits": 0,
   "clip_mean": 0.44822164473084203,
   "min_z_rel_median": 0.08253643071302119
  },
  "T3r_nonhoriz": {
   "rand_cos_to_delta": 0.001161572141931034,
   "rand_norm_over_delta": 1.0000000057475908,
   "delta_norm_over_old": 0.7080789137503196,
   "old_cos_full": 0.5708509563335351,
   "norm_over_old": 1.2270280083705638,
   "cap_hits": 0,
   "clip_mean": 0.46886905471843776,
   "min_z_rel_median": 0.08222874766501609
  }
 }
}
```

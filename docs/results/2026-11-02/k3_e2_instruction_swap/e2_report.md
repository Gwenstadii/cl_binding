# K3-E2 结果：换测试指令下的新目标捕获

回合 420/420，每格 10 个开发初态（0-9），固定噪声流。捕获 = 首次进入的目标恰为新目标；任何目标 = 至少进入一个目标格；抬起 = 物体中心抬高超过 3cm。

| 家族 | 模型 | 指令 | 指令文本 | 捕获 | 任何目标 | 抬起 | 首次目标分布 | 新目标终态成功 |
|---|---|---|---|---:|---:|---:|---|---:|
| scene2 | R0 | A | pick up the book and place it in the left compartment of the caddy | 0.0 | 1.0 | 1.0 | {'left': 10} | 0.0 |
| scene2 | R0 | B | pick up the book and place it in the front compartment of the caddy | 0.0 | 0.9 | 0.9 | {'front': 9, 'none': 1} | 0.0 |
| scene2 | R0 | N | pick up the book and place it in the right compartment of the caddy | 0.0 | 1.0 | 1.0 | {'left': 10} | 0.0 |
| scene2 | R0 | P | move the book into the caddy's right-hand slot | 0.0 | 0.3 | 0.6 | {'none': 7, 'left': 2, 'front': 1} | 0.0 |
| scene2 | R0 | S | pick up the book and place it in the back compartment of the caddy | 0.0 | 0.9 | 1.0 | {'front': 7, 'left': 2, 'none': 1} | 0.0 |
| scene2 | R0 | U | the weather is sunny today | 0.0 | 0.3 | 0.4 | {'none': 7, 'left': 2, 'front': 1} | 0.0 |
| scene2 | R0 | Z | zq7 kv3 | 0.0 | 0.2 | 0.5 | {'none': 8, 'front': 2} | 0.0 |
| scene2 | FT4 | A | pick up the book and place it in the left compartment of the caddy | 1.0 | 1.0 | 1.0 | {'right': 10} | 0.6 |
| scene2 | FT4 | B | pick up the book and place it in the front compartment of the caddy | 1.0 | 1.0 | 1.0 | {'right': 10} | 0.6 |
| scene2 | FT4 | N | pick up the book and place it in the right compartment of the caddy | 1.0 | 1.0 | 1.0 | {'right': 10} | 0.7 |
| scene2 | FT4 | P | move the book into the caddy's right-hand slot | 0.7 | 0.7 | 0.8 | {'none': 3, 'right': 7} | 0.4 |
| scene2 | FT4 | S | pick up the book and place it in the back compartment of the caddy | 1.0 | 1.0 | 1.0 | {'right': 10} | 0.4 |
| scene2 | FT4 | U | the weather is sunny today | 0.0 | 0.0 | 0.4 | {'none': 10} | 0.0 |
| scene2 | FT4 | Z | zq7 kv3 | 0.0 | 0.0 | 0.5 | {'none': 10} | 0.0 |
| scene1b | R0 | A | pick up the book and place it in the left compartment of the caddy | 0.0 | 1.0 | 1.0 | {'left': 10} | 0.0 |
| scene1b | R0 | B | pick up the book and place it in the front compartment of the caddy | 0.0 | 0.8 | 0.9 | {'front': 8, 'none': 2} | 0.0 |
| scene1b | R0 | N | pick up the book and place it in the right compartment of the caddy | 0.0 | 0.7 | 0.9 | {'left': 3, 'front': 4, 'none': 3} | 0.0 |
| scene1b | R0 | P | move the book into the caddy's right-hand slot | 0.0 | 0.7 | 0.7 | {'none': 3, 'left': 2, 'front': 5} | 0.0 |
| scene1b | R0 | S | pick up the book and place it in the back compartment of the caddy | 0.0 | 1.0 | 1.0 | {'front': 10} | 0.0 |
| scene1b | R0 | U | the weather is sunny today | 0.0 | 0.0 | 0.0 | {'none': 10} | 0.0 |
| scene1b | R0 | Z | zq7 kv3 | 0.0 | 0.0 | 0.0 | {'none': 10} | 0.0 |
| scene1b | FT4 | A | pick up the book and place it in the left compartment of the caddy | 0.6 | 0.6 | 0.7 | {'right': 6, 'none': 4} | 0.0 |
| scene1b | FT4 | B | pick up the book and place it in the front compartment of the caddy | 0.5 | 0.5 | 0.6 | {'none': 5, 'right': 5} | 0.2 |
| scene1b | FT4 | N | pick up the book and place it in the right compartment of the caddy | 0.5 | 0.5 | 0.6 | {'none': 5, 'right': 5} | 0.1 |
| scene1b | FT4 | P | move the book into the caddy's right-hand slot | 0.0 | 0.0 | 0.0 | {'none': 10} | 0.0 |
| scene1b | FT4 | S | pick up the book and place it in the back compartment of the caddy | 0.5 | 0.5 | 0.7 | {'none': 5, 'right': 5} | 0.2 |
| scene1b | FT4 | U | the weather is sunny today | 0.0 | 0.0 | 0.0 | {'none': 10} | 0.0 |
| scene1b | FT4 | Z | zq7 kv3 | 0.0 | 0.0 | 0.0 | {'none': 10} | 0.0 |
| goal | R0 | A | put the bowl on the stove | 0.0 | 0.9 | 0.9 | {'none': 1, 'left': 9} | 0.0 |
| goal | R0 | B | put the bowl on top of the cabinet | 0.0 | 0.7 | 0.8 | {'none': 3, 'front': 7} | 0.0 |
| goal | R0 | N | put the bowl on the plate | 0.0 | 1.0 | 1.0 | {'left': 10} | 0.0 |
| goal | R0 | P | set the bowl down onto the dish | 0.0 | 1.0 | 1.0 | {'left': 10} | 0.0 |
| goal | R0 | S | open the top drawer and put the bowl inside | 0.1 | 0.6 | 0.8 | {'none': 4, 'front': 4, 'left': 1, 'right': 1} | 0.0 |
| goal | R0 | U | the weather is sunny today | 0.0 | 0.9 | 0.9 | {'none': 1, 'left': 9} | 0.0 |
| goal | R0 | Z | zq7 kv3 | 0.0 | 0.5 | 0.7 | {'none': 5, 'left': 5} | 0.0 |
| goal | FT4 | A | put the bowl on the stove | 1.0 | 1.0 | 1.0 | {'right': 10} | 1.0 |
| goal | FT4 | B | put the bowl on top of the cabinet | 0.9 | 0.9 | 0.9 | {'none': 1, 'right': 9} | 0.8 |
| goal | FT4 | N | put the bowl on the plate | 0.8 | 0.8 | 1.0 | {'right': 8, 'none': 2} | 0.8 |
| goal | FT4 | P | set the bowl down onto the dish | 1.0 | 1.0 | 1.0 | {'right': 10} | 1.0 |
| goal | FT4 | S | open the top drawer and put the bowl inside | 0.8 | 0.8 | 0.8 | {'none': 2, 'right': 8} | 0.8 |
| goal | FT4 | U | the weather is sunny today | 0.9 | 0.9 | 1.0 | {'right': 9, 'none': 1} | 1.0 |
| goal | FT4 | Z | zq7 kv3 | 0.8 | 0.8 | 0.8 | {'none': 2, 'right': 8} | 0.6 |

## 配对对比（同家族、同模型、同初态，捕获的符号检验）

- scene2|R0|N_vs_U: n=10 b=0 c=0 p=None
- scene2|R0|N_vs_Z: n=10 b=0 c=0 p=None
- scene2|R0|N_vs_P: n=10 b=0 c=0 p=None
- scene2|R0|N_vs_S: n=10 b=0 c=0 p=None
- scene2|R0|A_vs_B: n=10 b=0 c=0 p=None
- scene2|FT4|N_vs_U: n=10 b=10 c=0 p=0.002
- scene2|FT4|N_vs_Z: n=10 b=10 c=0 p=0.002
- scene2|FT4|N_vs_P: n=10 b=3 c=0 p=0.25
- scene2|FT4|N_vs_S: n=10 b=0 c=0 p=None
- scene2|FT4|A_vs_B: n=10 b=0 c=0 p=None
- scene1b|R0|N_vs_U: n=10 b=0 c=0 p=None
- scene1b|R0|N_vs_Z: n=10 b=0 c=0 p=None
- scene1b|R0|N_vs_P: n=10 b=0 c=0 p=None
- scene1b|R0|N_vs_S: n=10 b=0 c=0 p=None
- scene1b|R0|A_vs_B: n=10 b=0 c=0 p=None
- scene1b|FT4|N_vs_U: n=10 b=5 c=0 p=0.0625
- scene1b|FT4|N_vs_Z: n=10 b=5 c=0 p=0.0625
- scene1b|FT4|N_vs_P: n=10 b=5 c=0 p=0.0625
- scene1b|FT4|N_vs_S: n=10 b=0 c=0 p=None
- scene1b|FT4|A_vs_B: n=10 b=1 c=0 p=1.0
- goal|R0|N_vs_U: n=10 b=0 c=0 p=None
- goal|R0|N_vs_Z: n=10 b=0 c=0 p=None
- goal|R0|N_vs_P: n=10 b=0 c=0 p=None
- goal|R0|N_vs_S: n=10 b=0 c=1 p=1.0
- goal|R0|A_vs_B: n=10 b=0 c=0 p=None
- goal|FT4|N_vs_U: n=10 b=1 c=2 p=1.0
- goal|FT4|N_vs_Z: n=10 b=2 c=2 p=1.0
- goal|FT4|N_vs_P: n=10 b=0 c=2 p=0.5
- goal|FT4|N_vs_S: n=10 b=2 c=2 p=1.0
- goal|FT4|A_vs_B: n=10 b=1 c=0 p=1.0

## 预登记判定

```json
{
 "H_vis_every_instruction_captures": {
  "values": {
   "scene2|A": 1.0,
   "scene2|B": 1.0,
   "scene2|N": 1.0,
   "scene2|P": 0.7,
   "scene2|S": 1.0,
   "scene2|U": 0.0,
   "scene2|Z": 0.0,
   "scene1b|A": 0.6,
   "scene1b|B": 0.5,
   "scene1b|N": 0.5,
   "scene1b|P": 0.0,
   "scene1b|S": 0.5,
   "scene1b|U": 0.0,
   "scene1b|Z": 0.0,
   "goal|A": 1.0,
   "goal|B": 0.9,
   "goal|N": 0.8,
   "goal|P": 1.0,
   "goal|S": 0.8,
   "goal|U": 0.9,
   "goal|Z": 0.8
  },
  "passing": 11,
  "total": 21,
  "all": false
 },
 "H_sim_form_like_captures_unrelated_does_not": {
  "similar_ge_0.8": {
   "values": {
    "scene2|N": 1.0,
    "scene2|P": 0.7,
    "scene2|S": 1.0,
    "scene1b|N": 0.5,
    "scene1b|P": 0.0,
    "scene1b|S": 0.5,
    "goal|N": 0.8,
    "goal|P": 1.0,
    "goal|S": 0.8
   },
   "passing": 5,
   "total": 9,
   "all": false
  },
  "unrelated_le_0.3": {
   "values": {
    "scene2|U": 0.0,
    "scene2|Z": 0.0,
    "scene1b|U": 0.0,
    "scene1b|Z": 0.0,
    "goal|U": 0.9,
    "goal|Z": 0.8
   },
   "passing": 4,
   "total": 6,
   "all": false
  }
 },
 "P3_R0_capture_le_0.2": {
  "values": {
   "scene2|A": 0.0,
   "scene2|B": 0.0,
   "scene2|N": 0.0,
   "scene2|P": 0.0,
   "scene2|S": 0.0,
   "scene2|U": 0.0,
   "scene2|Z": 0.0,
   "scene1b|A": 0.0,
   "scene1b|B": 0.0,
   "scene1b|N": 0.0,
   "scene1b|P": 0.0,
   "scene1b|S": 0.0,
   "scene1b|U": 0.0,
   "scene1b|Z": 0.0,
   "goal|A": 0.0,
   "goal|B": 0.0,
   "goal|N": 0.0,
   "goal|P": 0.0,
   "goal|S": 0.1,
   "goal|U": 0.0,
   "goal|Z": 0.0
  },
  "passing": 21,
  "total": 21,
  "all": true
 },
 "P4_replication": {
  "R0_first_under_A": {
   "scene2": {
    "left": 10
   },
   "scene1b": {
    "left": 10
   },
   "goal": {
    "none": 1,
    "left": 9
   }
  },
  "R0_first_under_B": {
   "scene2": {
    "front": 9,
    "none": 1
   },
   "scene1b": {
    "front": 8,
    "none": 2
   },
   "goal": {
    "none": 3,
    "front": 7
   }
  },
  "FT4_capture_A_B": {
   "scene2": [
    1.0,
    1.0
   ],
   "scene1b": [
    0.6,
    0.5
   ],
   "goal": [
    1.0,
    0.9
   ]
  }
 },
 "P5_lift_under_UZ": {
  "scene2": {
   "U": 0.4,
   "Z": 0.5
  },
  "scene1b": {
   "U": 0.0,
   "Z": 0.0
  },
  "goal": {
   "U": 1.0,
   "Z": 0.8
  }
 },
 "family_pattern": {
  "scene2": [
   1.0,
   1.0,
   1.0,
   0.7,
   1.0,
   0.0,
   0.0
  ],
  "scene1b": [
   0.6,
   0.5,
   0.5,
   0.0,
   0.5,
   0.0,
   0.0
  ],
  "goal": [
   1.0,
   0.9,
   0.8,
   1.0,
   0.8,
   0.9,
   0.8
  ]
 }
}
```


# K25-parallel-result：三指令不同组合与四指令评测总汇

更新：2026-11-26。本文汇总已经完成的parallel评测、模型身份、固定增益和运行配置；本次仅整理已有证据，没有启动训练、仿真或新增模型前向。

**主要结论：** 三指令的LR→Front、LF→Right两条路线，在4个模型条件下均恢复了部分旧指令分流，但最终放置与干预时长明显相关；四指令中，parallel能恢复部分旧任务，却不能同时保留Back。最新F4-B24在线复验中，原生Back19/20，短窗parallel10/20、全程1/20；四句同初态均原生成功仅8/20、1/20。

本文主体覆盖K25系列6个续训模型条件、每个两种窗口，共12种“模型×窗口”设定；F4-B24另有一次同设定的新在线复验。主体引用**800条实际parallel回合记录**：三指令320、Back-3000四指令160、F4-B24历史160、新在线160。它们复用20个保存初态，不能当成800个独立物理样本。父模型／原生F对照、标定前缀和完整残差实验不计入这800条。K22原始跨任务族实验单列为历史来源，不重复计入。

## 1. 历史指标与新在线指标的含义

| 本文记法 | 判定 | 回合是否成功即停 | 用途 |
| --- | --- | --- | --- |
| 几何首达G | 书抬高>3厘米，连续5步进入最早的唯一目标区域，且该区域为请求目标 | 否，历史回合跑满400步 | 路线分流诊断，不能代替原生任务成功 |
| 原生曾成功E | 请求目标原生BDDL谓词在400步内至少一次为真 | 历史轨迹没有提前停；这是已完成轨迹的事件计数 | 与当前成功事件口径对应，但不能称为新在线提前终止实验 |
| 固定终态T | 第400步请求目标原生谓词为真 | 否 | 后续继续执行后的终态诊断，不要求额外释放／保持 |
| 新在线原生成功S | 每个真实动作后检查请求目标原生谓词，首次为真立即结束，最长400步 | 是，包括10步执行块内部 | 当前主任务指标 |

额外“最后50步至少45步满足、最后10步不能两指夹持”等释放保持门禁曾用于Back选档，不等于上表T，更不是当前S。Back多数示范在仍夹持时结束，Back竖槽姿态对后续动作敏感，不能由低T或旧门禁失败推出“Back没有学会”。详见[Back判据更正](K25-Back-LIBERO.md)。

以下成对／三旧／四句“成组”均指：同一保存初态分别复位执行各句，再取成功集合交集；不是在一个回合依次执行多句。所有计数分母均为20，失败保留在分母。

## 2. 模型清单与训练路线

LR-F-1k等仅为本文表格简称；F4-B24是已登记代号。LR表示Left＋Right，LF表示Left＋Front。前三指令实验只评两句旧指令，不能据此宣称三条指令全部保留。

| 简称 | 训练历史／续训终点 | 匹配父模型P | 旧参考O | 已评请求 |
| --- | --- | --- | --- | --- |
| LR-F-1k | LR 24000 → Front既有分支1000 | P-LR：R1 joint LR 24000 | Left、Right | Left、Right |
| LF-R-1k | LF 12000 → Right既有分支1000 | P-LF：R0 joint LF 12000 | Left、Front | Left、Front |
| LR-F-12k | LR 24000 → Front原K21/K22终点12000 | P-LR | Left、Right | Left、Right |
| LF-R-12k | LF 12000 → Right原K21/K22终点12000 | P-LF | Left、Front | Left、Front |
| Back-3000 | LR 24000 → Front replay_full1000 → Back3000 | P3：K19 replay_full Front1000 | Left、Front、Right | 四句全部 |
| F4-B24 | 同P3 → 另一条Back标准数据续训24000 | P3 | Left、Front、Right | 四句全部 |

**1000与12000条件不是同一续训轨迹的两个时间点。** 不能把跨分支差异解释为只改变训练步数；两条路线也同时改变旧任务组合及匹配父模型。Back-3000和F4-B24同样是不同Back训练配方，不是仅延长同一条训练链。未评Right＋Front→Left等其他组合。

### 2.1 实际检查点与SHA256

目录项的SHA指其`model.safetensors`；单文件项指列出的任务权重文件。155张任务权重需装载在匹配冻结基座上，不能单独当作完整模型。所有路径为服务器绝对路径。

| 身份 | 实际路径 | SHA256 |
| --- | --- | --- |
| P-LR | `/home/lzx/cl_binding/.server/r1_left_right_20261104_01/train_r1/checkpoints/024000` | `225322ba28d0ad152fcced19950c43e3e40a435f1835cf64a664226da7830003` |
| LR-F-1k | `/home/lzx/cl_binding/.server/k18_training_cause_20261111_01/train/checkpoints/001000/model.safetensors` | `082df3050cb7efde0ab8f27e56a234af90d56477bc04e668644b68b71206d41b` |
| P-LF | `/home/lzx/cl_binding_stage2a/outputs/r0/dual_left_front_20261023_01/checkpoints/012000` | `15df8ed2839c281ba5d55b411b43a15177013b4d12b0e5cb98baefdcd60313f7` |
| LF-R-1k | `/home/lzx/cl_binding/.server/k25_order_restore_20261122_01/lf_right_train/001000.safetensors` | `366cb4413a26e8a1df333a50ea6b6d50be182ccee01da0de88fb2b520cce8c32` |
| LR-F-12k | `/home/lzx/cl_binding/.server/r1_then_front_v1/train/checkpoints/012000` | `8ff1bf8f372d728a3218101a3cc93fb6fb9d484e8fefb39c48d81a0c8ab2c515` |
| LF-R-12k | `/home/lzx/cl_binding_stage2a/outputs/stage2a/stage2a_ft4_er_20261024_01_ft4/checkpoints/012000` | `c67c3fc45763dd4ebd63d24354bd9b32aae830edb67892d44f199d1dd08ca54f` |
| P3 | `/home/lzx/cl_binding/.server/k19_retention_20261111_01/runs/20261121_replay_full/001000/trainable.safetensors` | `f8bd197dc20886ee14873429c90a7b51ea00ccc2656c28fc542b625b1261772c` |
| Back-3000 | `/home/lzx/cl_binding/.server/k26_multigoal_20261121_01/back/003000/trainable.safetensors` | `ccee300a0cfc2ae9755eb6a149e0e1dee1d1f57e6a1677584162bab61788ad83` |
| F4-B24任务权重 | `/home/lzx/cl_binding/.server/k25_back_retrain_20261124_01/training/024000/trainable.safetensors` | `d75b4df483e850c895ec203f48ff4ee4dd274ca2bfe7c7c0d20bdcb69bfccc47` |
| F4-B24完整推理包 | `/home/lzx/cl_binding/.server/k25_native_event_20261125_01/models/F4-B24` | `e872dd3567a06e929ebb8a4e8264e49103b4311acff45349316bcde2c691c52c` |

P3、Back-3000和F4-B24的冻结基座均为P-LR；归一化继承R1，经K19不变。F4-B24推理包500张状态张量已核验等于选定155张任务权重＋原冻结来源，不经过重新训练。LR的1000分支从P-LR加载、LF的1000分支从P-LF加载再覆盖选定权重；两个12k完整终点的动作／观测归一化与各自P逐值核对一致。

### 2.2 已封存的关键训练配置

以下记录用于辨认模型，不表示本次重新执行训练。历史终点未在当前汇总中核实的优化器字段不推测补齐。

| 模型／阶段 | 已核实配置 | 原始来源 |
| --- | --- | --- |
| LR-F-1k的Front续训 | K18既有Front-only分支的1000步权重；配置seed20261021、batch6、lr1e-4、warmup100；归一化继承LR | [该分支配置](results/2026-11-11/k18_training_cause/code/config.json)、[实际选档身份](results/2026-11-23/k25_parallel/protocol.json) |
| LF-R-1k的Right续训 | seed20261021；1000步；batch6；lr1e-4；warmup100；优化器重置；不回放；继承LF归一化 | [训练清单](results/2026-11-22/k25_order_restore/lf_right_training_manifest.json) |
| Back-3000 | seed20261123；3000步；batch6；新AdamW；lr1e-4，warmup100后保持；clip10；Back-only，不回放 | [训练清单](results/2026-11-24/k25_parallel4/history/k26_training_manifest.json)、[实际训练程序](results/2026-11-24/k25_parallel4/history/k26_train_back.py) |
| F4-B24 | seed20261124；24000步；batch6；新AdamW；peak lr5e-5／floor5e-6；warmup300后连续余弦衰减；clip10；不回放 | [完整配置](results/2026-11-24/k25_back_retrain/config.json) |

两个Back模型均用原40条对齐Back示范，数据路径`/home/lzx/cl_binding_stage2a/datasets/stage1_single_back_train_aligned_v1`，split SHA256 `a55e1c1c8d0796329fc21b196b4d0a0a9348a2a63efebe86679a3a2a0e1b54f0`。F4-B24每批3条全示范采样＋3条真实末30%阶段采样，episode-uniform；没有合成松手尾帧。动作对齐为post-action observation→next action（action offset1，末观察丢弃），图像旋转180°、128×128、20fps。Back-3000采用既有balanced_indices全帧采样，没有F4-B24的尾段混采设置。

## 3. parallel公式与运行超参数

每次去噪均在相同当前观察、动作草稿x和流时间t上比较模型响应。两旧任务时：

```text
μP = [vP(A)+vP(B)]/2；dP = [vP(A)−vP(B)]/2
μF = [vF(A)+vF(B)]/2；rF(q) = vF(q)−μF
c1 = <rF(q)7, dP,7> / max(||dP,7||², 1e−30)
vparallel(q) = μP + α c1 dP
```

内积展平50×7维，输出作用于完整50×32响应，再积分、取7维、反归一化和裁剪。实际调用原K22 `Lab2.combine2("par")`，系数／组合为float32，再转原生速度dtype。三旧任务时用明确的多任务扩展：

```text
O = {Left, Front, Right}；μP、μF分别为各模型在O上响应的均值
BP,32 = centered(VP) · Helmert(3)；BP,7展平为350×2
vparallel(q) = μP + α BP,32 · pinv(BP,7) · rF(q)7
```

三旧投影使用float64 SVD、固定`rcond=1e−6`，再转原生dtype；同一两维系数提升至完整32维。中心化后最多二维是代数结果，不代表模型内部任务表征只有二维。两旧与三旧实现的精度／求解器不同，已分别记录，不能声称两者代码完全相同。

修复中P只接收固定旧参考；真实完整q送入F，未按目标标签强制系数符号、调用P的Back方向或绕回原生F。原生P基线自身会接收真实q。这里仍依赖P提供共同响应与旧方向，是双模型控制诊断。

| 配置项 | 值／约束 |
| --- | --- |
| 模型／场景 | SmolVLA；LIBERO Scene2书放caddy四格 |
| 规范指令 | `pick up the book and place it in the {left/front/right/back} compartment of the caddy` |
| 输入／动作 | 图像128×128；状态8维；动作有效7维、填充32维 |
| 动作与积分 | chunk_size50；每次执行10步；flow_steps10（Euler） |
| 运行精度 | CUDA autocast bfloat16；两旧投影float32／三旧SVD float64 |
| 复位／噪声 | 固定800–819初态、保存复位状态和逐块噪声；同初态跨句／跨窗口配对 |
| 窗口 | 1–150：151步后交回F同一句；1–400：持续parallel；新在线成功则当步停止 |
| 评估上限 | 400真实动作步；不按抓取成功筛选回合 |
| 硬件／环境 | 历史运行：2×RTX3090；Python3.12.15；PyTorch2.10.0+cu128；CUDA12.8 |
| 关键依赖 | lerobot0.6.2；transformers5.5.4；numpy2.2.6；MuJoCo3.8.1；robosuite1.4.0 |

历史初态文件SHA256为`0c233a96513e7ecde914761fb7d41b8c11e5de457a492854d407b0f996438969`。运行时需显式恢复`n_action_steps=10`，项目通用加载器曾将它改成1；F4-B24新在线清单明确记录此项。软件版本来自[三指令运行快照](results/2026-11-23/k25_parallel/runtime.json)及[四指令运行快照](results/2026-11-24/k25_parallel4/runtime.json)，不将历史版本快照冒称为每一轮重新采集。

### 3.1 固定增益与标定

| 条件 | α | 来源 |
| --- | --- | --- |
| LR-F-1k | 1.24663519859314 | 独立开发范数比，上中位数封顶10 |
| LF-R-1k | 1.278009176254272 | 独立开发范数比，上中位数封顶10 |
| LR-F-12k | 2.959 | 沿用K14/K21历史固定2.959 |
| LF-R-12k | 2.948583602905273 | 复用原K21开发标定 |
| Back-3000 | 1.63491654569347 | 独立开发三旧中心化响应范数比，上中位数封顶10 |
| F4-B24（历史及新在线） | 0.7814008479195439 | 同模型独立标定；新在线原值复用 |

新标定规则：开发初态0–9，由原生F执行新任务160步前缀；在30、40、…、150步的首个Euler输入记录父／续训区别范数比，每模型130项，排序取第66项，上限10。两旧用半差范数；三旧用中心化旧响应的Frobenius范数。只按范数标定，不根据正式成功率选α，也未在投影后再做等范数放大。LR-F-12k的2.959是历史固定值，不伪称本轮按统一规则重新算出。

F4-B24的α<1，本身是缩小；parallel并非一律“区别放大”。增益标定原则相同不意味着每个目标的有效方向强度相同。固定α也不代表找到了最优α。

## 4. 三指令：两条路线、四个模型、两个窗口

此表全部来自已完成的400步轨迹。单句和成对单元格均为 **G／E／T（几何首达／原生曾成功／第400步终态）**，各/20。E是既有summary的ever字段，不是本次新在线S。

| 模型 | 窗口末步 | Left G/E/T | 另一旧句 | 该句G/E/T | 同初态成对G/E/T |
| --- | --- | --- | --- | --- | --- |
| LR-F-1k | 150 | 20／20／16 | Right | 19／14／14 | 19／14／10 |
| LR-F-1k | 400 | 20／20／18 | Right | 19／6／6 | 19／6／5 |
| LF-R-1k | 150 | 18／18／15 | Front | 18／12／10 | 16／11／9 |
| LF-R-1k | 400 | 18／18／14 | Front | 17／4／4 | 15／4／4 |
| LR-F-12k | 150 | 20／20／17 | Right | 19／17／15 | 19／17／13 |
| LR-F-12k | 400 | 20／20／19 | Right | 19／17／17 | 19／17／16 |
| LF-R-12k | 150 | 17／17／15 | Front | 16／6／5 | 13／5／4 |
| LF-R-12k | 400 | 17／17／15 | Front | 15／3／3 | 12／2／2 |

新任务Front或Right未在这批三指令parallel回合中评测，故“成对恢复”不等于三个任务同时恢复。E和G可不一致：先进入一个格子与真正满足放置谓词是不同事件。

### 4.1 既有父模型与原生F参照

下表复用封存参照，不计为本次新回合；格式仍为G／E／T。

| 路线／条件 | 参照 | Left G/E/T | 另一旧句 | 该句G/E/T | 成对G/E/T |
| --- | --- | --- | --- | --- | --- |
| lr_front_1000 | parent | 20／20／20 | Right | 18／17／17 | 18／17／17 |
| lr_front_1000 | native | 17／17／12 | Right | 3／0／0 | 3／0／0 |
| lf_right_1000 | parent | 17／17／16 | Front | 20／15／14 | 17／14／13 |
| lf_right_1000 | native | 0／0／0 | Front | 0／0／0 | 0／0／0 |

两个12k终点没有这批800–819初态上的新增原生F基线，不能用另一批初态的0/20补齐。LR-F-1k与LF-R-1k的父模型与其对应12k条件相同。

只延长窗口后，成对T依次为10→5、9→4、13→16、4→2；只有LR-F-12k改善。160组前150步的动作、状态和去噪记录已逐项一致，说明这是后段控制方式的影响，不能解释为前段抓取改善。

## 5. 四指令：Back-3000与F4-B24历史结果

本节全部为已执行至400步的历史轨迹，单句格式G／E／T。父P3未训练Back，Back列仅为未训练参照。P3在F4-B24历史实验中复用原80条，不重复当作新模型能力核验。

| 模型 | 条件 | Left G/E/T | Front G/E/T | Right G/E/T | Back G/E/T | 三旧G/E/T | 四句G/E/T |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Back-3000 | parent | 20／20／18 | 19／15／10 | 20／13／13 | 0／2／0 | 19／11／6 | 0／2／0 |
| Back-3000 | native | 17／18／10 | 17／0／0 | 0／0／0 | 0／16／1 | 0／0／0 | 0／0／0 |
| Back-3000 | parallel150 | 20／20／16 | 19／14／14 | 19／11／11 | 0／8／4 | 18／8／6 | 0／3／2 |
| Back-3000 | parallel400 | 20／20／16 | 19／16／13 | 20／11／11 | 0／1／0 | 19／9／5 | 0／0／0 |
| F4-B24 | parent | 20／20／18 | 19／15／10 | 20／13／13 | 0／2／0 | 19／11／6 | 0／2／0 |
| F4-B24 | native | 20／17／12 | 20／0／0 | 20／8／8 | 0／19／4 | 20／0／0 | 0／0／0 |
| F4-B24 | parallel150 | 20／20／18 | 18／15／15 | 20／16／15 | 0／10／3 | 18／13／12 | 0／8／2 |
| F4-B24 | parallel400 | 20／20／17 | 19／15／13 | 20／13／13 | 0／1／0 | 19／10／8 | 0／1／0 |

F4-B24的E列已与2026-11-25[原生事件重评分](results/2026-11-25/k25_native_event/rescore.json)逐项核对；重评分没有产生新回合。Back-3000的E直接取其封存summary原生谓词ever计数，同样不是在线提前终止。

Back的G全部为0，原因是旧“首次唯一几何到区”记录常先到Front；这不否定它随后满足Back原生谓词。按E看，Back-3000原生16/20、F4-B24原生19/20；两个模型全程parallel均只有1/20。因此不能再以Back几何首达0或低T概括其学习能力。

## 6. F4-B24：最新原生成功即停在线复验

2026-11-26完成320个全新在线回合，四条件×四句×20初态。保持同模型、同α=0.7814008479195439、同初态及噪声；使用S口径，每个真实动作后检查，首次成功立即结束。下表单元格为S，各/20，无固定400步终态列。

| 条件 | Left | Front | Right | Back | 三旧成组 | 四句成组 |
| --- | --- | --- | --- | --- | --- | --- |
| parent | 20 | 15 | 13 | 2 | 11 | 2 |
| native | 17 | 0 | 8 | 19 | 0 | 0 |
| parallel150 | 20 | 15 | 16 | 10 | 13 | 8 |
| parallel400 | 20 | 15 | 13 | 1 | 10 | 1 |

短窗相对原生F：Left新增3／损失0，Front15／0，Right10／2，Back1／10；全程分别3／0、15／0、7／2、0／18。两个窗口都损失原生Right成功的809、816，故净成功数不能代替配对得失。

只将短窗延长至全程：Left不变；Front新增809、损失810，总数仍15；Right损失3例且无新增；Back损失9例且无新增。短窗Back的10例成功中，仅814在干预期第141步发生，另9例在交回F后的152–213步发生。不能把10例全部归功于parallel独立完成Back。

320条新轨迹均核对停止步与请求谓词；204次成功、116次超时。186次成功位于执行块内部，已当步终止；80组窗口前缀一致；320个新执行前缀与历史同条件记录的float32动作、原生标记一致，物体位置误差≤1e-7。后者用于核验，不为控制器提供历史动作。新S与历史E计数一致，是复验一致性，不是独立留出泛化证据。

## 7. 证据索引、实际运行与计数去重

| 批次／来源 | 实际新增回合 | 其中parallel | 本汇总用途 |
| --- | --- | --- | --- |
| 2026-11-23 K25-parallel | 160 | 160 | 三指令四模型短窗 |
| 2026-11-24 K25-parallel4：原模型延长 | 160 | 160 | 三指令四模型全程 |
| 2026-11-24 K25-parallel4：Back-3000 | 320 | 160 | 四指令P/F/两个窗口 |
| 2026-11-24 K25-whole4 | 160 | 0 | 完整残差；原parallel复用，不重复计数 |
| 2026-11-24 K25-whole4-new | 400 | 160 | F4-B24原生80、parallel160、完整残差160；P另复用80 |
| 2026-11-25 K25-native-event | 0 | 0 | 既有轨迹重评分 |
| 2026-11-26 K25-parallel4-correct | 320 | 160 | F4-B24新在线P/F/两个窗口 |

上表每批的“实际新增回合”还包含非parallel方法及参照，不能相加后称为parallel样本数。标定开发前缀、Back资格检查、K22旧实验与技术中断尝试另计。

| 内容 | 报告 | 锁定配置／逐项结果 |
| --- | --- | --- |
| 三指令短窗 | [K25-parallel](K25-parallel.md) | [protocol](results/2026-11-23/k25_parallel/protocol.json)、[summary](results/2026-11-23/k25_parallel/summary.json) |
| 三指令全程＋Back-3000四指令 | [K25-parallel4](K25-parallel4.md) | [full_protocol](results/2026-11-24/k25_parallel4/full_protocol.json)、[four_protocol](results/2026-11-24/k25_parallel4/four_protocol.json)、[summary](results/2026-11-24/k25_parallel4/summary.json) |
| F4-B24历史 | [K25-whole4-new](K25-whole4-new.md) | [protocol](results/2026-11-24/k25_whole4_new/protocol.json)、[summary](results/2026-11-24/k25_whole4_new/summary.json) |
| F4-B24重评分 | [Back更正](K25-Back-LIBERO.md) | [rescore](results/2026-11-25/k25_native_event/rescore.json) |
| F4-B24新在线 | [K25-parallel4-correct](K25-parallel4-correct.md) | [manifest](results/2026-11-26/k25_parallel4_correct/manifest.json)、[summary](results/2026-11-26/k25_parallel4_correct/summary.json)、[audit](results/2026-11-26/k25_parallel4_correct/audit.json) |
| 本次整理的机器可读索引 | 本文 | [index.json](results/2026-11-26/k25_parallel_result/index.json)：源文件SHA、模型身份及各表记录 |

服务器目录依次为：

```text
/home/lzx/cl_binding/.server/k25_parallel_20261123_01
/home/lzx/cl_binding/.server/k25_parallel4_20261124_01
/home/lzx/cl_binding/.server/k25_whole4_new_20261124_01
/home/lzx/cl_binding/.server/k25_native_event_20261125_01
/home/lzx/cl_binding/.server/k25_parallel4_correct_20261126_01
```

实际入口分别为`parallel_reproduction.py`、`parallel_full400.py`、`parallel_four.py`、`whole4_new.py`和`parallel4_correct.py`；多进程队列入口及完整命令见各批README和runtime清单。加载独立环境`source /home/lzx/cl_binding/.server/k1_screen_20261101_01/code/env.sh`后需重新切回相应实验目录；环境脚本会改变工作目录。模型、数据与缓存留在服务器，报告不复制权重。

## 8. 历史来源附录：K22的parallel结果

为保留方法来源，这里另列[2026-11-13 K21/K22报告](K21_K22_生死测试与残差拆解_20261113.md)已存档的parallel计数。它们不是K25在800–819上的复现，初态、任务族及接手方式不同，不能与主体拼接成配对比较。此附录仅按原报告汇总成对几何首达；该报告未完整提供逐线最终放置与精确α表，不推测补齐。

| 原线 | 任务／父子关系 | 抓稳快照接手60步：parallel／窗口P | 从复位前150步：parallel／全程P |
| --- | --- | --- | --- |
| r0_s2 | Scene2 LF→Right；与本文LF-R-12k同父子身份 | 20／19 | 16／18 |
| c_s2a | Scene2 LF→Right，独立seed20261102 | 18／17 | 20／19 |
| c_s2b | Scene2 LF→Right，独立seed20261103 | 16／11 | 14／12 |
| scene1b | Scene1b LF→Right | 16／16 | 12／17 |
| goal | LIBERO-Goal 灶台＋柜顶→盘子 | 快照不足，未完成该条件 | 11／12 |
| c_goal | 同Goal，独立seed20261102 | 快照不足，未完成该条件 | 14／15 |
| r1_front | Scene2 LR→Front；与本文LR-F-12k同父子身份 | 18／18 | 19／19 |

原K22从复位窗口为1–150，之后交还F至400；快照组先由F生成符合抓稳条件的共同状态，接手60步后交回F，具有快照筛选。其α沿用K21开发旧／新区别幅度比上中位数、上限10，r1_front固定2.959，r0_s2精确值2.9485836029052734；其余线只有原报告约2.95–7.22范围，本汇总未核实其精确值。算法源码及原父／子路径见[封存k21.py](results/2026-11-23/k25_parallel/history/k21.py)、[k22.py](results/2026-11-23/k25_parallel/history/k22.py)、[k22p3.py](results/2026-11-23/k25_parallel/history/k22p3.py)。这也不支持把“无行为调参”改写成“没有增益”。

## 9. 目前可用于论文的结论边界

1. 在已测两种三指令组合上，父μ＋旧方向投影可以把续训模型的残留条件信号变成部分旧目标分流；单句、成对及放置完成需要分别报告。三指令只测旧两句，缺少新指令保持率。
2. 增益、方向和共同响应的匹配依模型而变；延长窗口不是统一有效的补救。1000与12k分支差异不构成只改变训练时长的因果证据。
3. 四指令的强Back基线F4-B24为19/20，新在线parallel仍使Back降至10/20或1/20，因此当前方法没有实现四任务兼容恢复。
4. 相对原生F，parallel同时换μ、限制方向并缩放区别；本汇总不能唯一归因于投影删掉Back信息，也不能把偏轴成分统一称为有害噪声。[完整残差同条件对照](K25-whole4-new.md)属于另一个方法，其结果不能冒记为parallel。
5. K25主体没有新增改写句、反向符号、独立训练种子或新物理留出集；K22跨任务族证据需保留自身协议限制。两者均依赖匹配父模型，不是单模型修复的证明。

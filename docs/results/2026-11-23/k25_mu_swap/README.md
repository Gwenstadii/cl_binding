# K25：LF→Right 的共同响应替换

主报告：[K25-mu.md](../../../K25-mu.md)。本轮直接比较续训第 1000 步共同响应 μF 与 LF 父模型第 0 步共同响应 μP。V 锚点固定为本线第 30 步，α 固定为 5.294038631688923；两条旧指令为 Left／Front，新增任务为 Right。这里的第 0 步是旧任务联合训练后的父模型，不是未微调基础模型。

## 锁定的对照

所有组固定检查点、规范句、噪声、归一化、800–819 初态、400 控制步和评分。主 K25 对照的条件残差系数 β=1，改变 μ 来源时保留同一 V 修正算法、强度和符号。每十步重新规划，在十次去噪中用相同当前观察、动作草稿 x 和时间 t 重算：

```text
μF = [vF(Left) + vF(Front)] / 2
μP = [vP(Left) + vP(Front)] / 2
rF(q) = vF(q) − μF
ΔdV = dF[V30] − dF

K25：       μb + rF(q) + α × polarity × decoded_signF(q) × ΔdV
只换 μ：    μP + rF(q)
K14 式：    μb + 2.959 × rF(q)
```

μb 为 μF 或 μP。父模型始终只接收固定规范 Left、Front 两句，其输出取平均；真实 q 只送给续训执行模型。K14 式条件不加 V 修正，2.959 沿用历史固定值，未在本批重新标定。反向只翻转 V 追加项，不翻转晚期原有残差或共同项。时间窗为 81–180 和 1–400，JSON 内使用零基半开区间。

本轮为**两个模型来源参与的机制诊断**。父模型平均不提供当前句子的条件区别，但仍是外部模型响应；即使有效，也不能称为已经修好一个独立续训模型。

## 数量与统计单位

每个初态分别复位执行两条旧句，每句 15 个条件：

- 原生 F 一次；晚期 μ＋无修正在两个窗口下都是同一原生策略，只执行一次，不重复计数。
- 两个窗口各七组：父 μ 无 V；晚期／父 μ × V 正／反；晚期／父 μ × 2.959 倍真实条件残差。

共 **600 次实际 400 步执行**。其中 **200 条为原生和原 K25 条件的完整复跑，400 条为新条件**。200 条与 K25-V 原始档案核对全部 400 步的状态、实际动作、签名、读出和评分；这些复跑不增加独立样本量。另引用 K25-anchor 的 40 条完整父模型旧任务资格记录作为上下文，不计入 600 次执行。

统计单位仍为同一批 **20 个物理初态**。首达、终态放置、曾成功、抓取及同初态两句均正确分别统计；所有失败保留。没有新的措辞、独立留出集、独立训练种子或修复条件的新 Right 保持率评估。

## 记录与核验

- `analysis_plan.json` 在队列启动前锁定问题、主比较、控制及解释边界。
- `protocol.json` 记录模型和六份直接源码哈希、共享依赖、预处理器、固定网格与版本。`runtime.json` 记录实际环境、30 个复用项目模块、队列源码及分析计划哈希。
- `unit_00`–`unit_39` 各对应一个初态×真实指令，每个保存 15 条密集轨迹。每条包含 400 步实际 7D 动作、书本／夹爪 XYZ、接触、放置和区域，以及 40 个动作块哈希和 41 个状态签名。
- `*_COMPLETE.json` 保存逐文件哈希、实际权重指纹、运行前后权重不变、五条参考完整复跑、父模型来源、同输入均值／区别闸门及执行 dtype 均值舍入误差。
- 响应闸门分别验证：改变 μ 只产生共享差；同输入的条件区别保持；正反 V 修正均值相同；2.959 作用于续训实际句子的残差；β=1 且零注入时原生输出逐值一致。均值／区别的代数检验使用 float64，执行仍沿用原 BF16 速度场类型；有限精度误差另存。
- `summary.json` 和补充表保留全部配置、正反实际去向、μ 来源配对、在父 μ 上添加 V 的配对收益、残差放大收益、逐初态获得／失去的编号。
- `history/` 原样保存 K14／K15 说明和实际公式代码，`sources.json` 给出来源与字节哈希。历史相对链接按原目录解释。K15 的 17/20 来自 R1→Front-12000、已抓稳的共同快照及 60 步干预；本轮为 LF→Right-1000、完整复位与不同窗口，不将两个分流数字直接当作相同设置的复现。
- `artifact_manifest.json` 绑定新报告、源码、轨迹、日志、统计和图表。模型、示范数据与缓存仍留服务器。
- 实际运行源码在本目录按原字节保留。仓库 `scripts/k25` 的两份维护模块只清理了末尾多余空行，未改变代码；`source_normalization.json` 同时记录原件与维护版本的 SHA256。下列复算命令使用冻结原件。

## 复算

在仓库根目录，统计复算只需 Python 标准库和已经归档的 K25-V、K25-order-restore、K25-anchor：

```powershell
python -m unittest discover -s tests -p test_k25_mu_swap.py -v
python -B docs/results/2026-11-23/k25_mu_swap/summarize_mu_swap.py docs/results/2026-11-23/k25_mu_swap --check
python docs/results/2026-11-23/k25_mu_swap/report_generation.py
```

绘图另需 Matplotlib 与 NumPy：

```powershell
python scripts/k25/plot_mu_swap.py docs/results/2026-11-23/k25_mu_swap docs/figures/2026-11-23/k25_mu_swap
```

实际执行目录为 `/home/lzx/cl_binding/.server/k25_mu_swap_20261123_01`。使用现有离线环境，无新训练：

```bash
source /home/lzx/cl_binding/.server/k1_screen_20261101_01/code/env.sh
mu_run=/home/lzx/cl_binding/.server/k25_mu_swap_20261123_01
export PYTHONPATH="$mu_run:$PYTHONPATH"
"$PY" -m unittest discover -s "$mu_run/tests" -p test_k25_mu_swap.py -v
"$PY" "$mu_run/mu_swap.py" prepare
"$PY" "$mu_run/capture_runtime.py"
nohup "$PY" -u "$mu_run/run_mu_swap.py" --workers 4 > "$mu_run/queue.log" 2>&1 < /dev/null &
"$PY" "$mu_run/summarize_mu_swap.py" "$mu_run"
```

初始每 GPU 两个进程；显存检查后，`capacity.json` 将上限改为总共八个、每 GPU 四个进程，队列日志保留变更。并发安排不依据成功结果。重新执行需独立目录，准备入口与轨迹入口均拒绝覆盖已有方案或分片。

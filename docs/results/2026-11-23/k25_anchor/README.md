# K25：第 0／30 步锚点资格审计

主报告为 [K25-anchor.md](../../../K25-anchor.md)。`0` 指每条学习组合开始新任务续训前、已经训练过两条旧任务的父模型，不是未微调的 SmolVLA 基础模型。两条组合仍为 Left＋Right→Front、Left＋Front→Right，晚期执行端固定第 1000 步。

0／30 是新任务续训的相对步数；第 0 步分别是 R1-24000、R0-12000。`initialization_audit.json` 将两条原训练初始化清单与本轮第 0 步模型哈希核对；历史完成标记也确认两个第 30 步文件的步数及重载资格。

## 三层资格与网格

1. **完整供体策略**：父模型 0、第 30 步模型各自从复位执行 A／B／N 三句，共 240 回合。第 1000 步的 120 条既有原生记录另列作参照。
2. **八张 V 的兼容性**：第 1000 步模型只将八张 V 换成第 0／30 步值，全程原生执行两句旧任务，共 160 回合。此对照允许 V 影响共同项，不能与只加区别修正混为一谈。
3. **保留晚期 μ 的修复**：0／30 锚点 × 1／5.294038631688923 倍 × 81–180／1–400 窗口 × 正反符号 × 两句旧任务 × 20 初态 × 两条学习线，共 1280 个条件回合。其中第 30 步、5.294 倍的 320 条已由 K25-V 实际执行，本轮以源哈希与同输入动作探针核验后引用；其余 960 条新执行。

总计 **1360 条本轮网格回合、320 条引用，1680 条网格记录**；120 条晚期原生参照另列。1360 条中，中断前完成 804 条，恢复运行补齐 556 条；另执行 8 条完整技术复验，合计实际完成 1368 次 400 步执行。技术复验不增加网格样本量。网格中包含 40 条已有 LF 父模型条件复跑，其结果及第 80 步签名与原档核对。相同的 20 个 800–819 物理初态在所有条件中配对复用；回合数不等于独立样本量，也不是新的留出集。

`analysis_plan.json` 在队列开始前锁定分析目标。所有锚点、剂量、窗口与符号均完整报告，不为不同目标事后拼接最好条件。1 倍检验自然历史改动的作用，5.294 倍检验原方法剂量的迁移，不在本批重标定。

## 控制与源码

- `protocol.json`：六个检查点 SHA256、八张 V 的来源、源码与历史协议哈希、冻结部分重建核验、版本、固定网格。
- `cases.json`：与 K25-V 完全相同的复位状态、观察签名、噪声及几何参数。JSON 时间窗为零基半开区间。
- `*_*.jsonl.gz`：1360 条完整网格轨迹；包含 400 步书本和夹爪坐标、实际 7D 动作、接触、放置与区域标记，以及每十步动作哈希、状态签名与读出。804 条保留记录与中断前解压后的 JSON 行逐字节相同。
- `*_COMPLETE.json`：逐分片文件 SHA256、0／30／1000 的实际张量指纹、0／30 的八张 V 差指纹、四组局部重算／权重恢复／μ 守恒／新目标零注入闸门，以及旧条件探针与父模型复跑数量。
- `summary.json`：三层资格分别计分；含首达、终态、曾成功、抓取、两旧任务成对成绩、逐初态获益／损失及反向转移。
- `report_generation.py`：从固定统计再生主报告和全部补充表，使用标准库；每个配置都输出，不按成绩筛选。
- `anchor_check.py`、`anchor_protocol.py`、`run_anchor_check.py` 与统计脚本为运行快照。共享 `v_timing.py`、`v_timing_protocol.py` 和 `summarize_v_timing.py` 沿用已锁定版本；更早的 28 个实际项目依赖仍在上一轮的 `k25_order_restore/dependencies` 中逐一核验。
- `prepare.log`、分片日志、`queue.log` 和 `queue_COMPLETE.json` 保存恢复运行。`interrupted/` 原样保留第一次运行的方案、源码、804 条完整记录、运行来源与日志；其中 `startup_import_error.log` 记录产生回合前补齐共享协议模块的错误，`lf_right_1000_3.log` 记录产生完整轨迹后，在完成标记中把 BF16 张量转为 NumPy 时的错误。其他工作进程随后停止，未完成回合不计样本。部分原始 gzip 缺少正常结束标记，恢复入口只接受完整且可复核的 JSON 行。
- `recovery_manifest.json` 固定八个中断文件、804 条行的哈希和原方案／源码哈希。恢复运行使用同一物理网格，保留全部完整回合，不按成败挑选；补齐 556 条缺失条件，并在八个原活动分片各完整复跑一条已保存的区别修正回合。八次均要求 400 步轨迹、动作哈希、状态签名、读出和评分逐值相同。`*_COMPLETE.json` 单列保留、补跑和技术复验数量。
- `anchor_tensor_hash.py` 直接按张量的存储字节计算哈希，修复 BF16 的完成记录序列化；不改变推理计算。`anchor_recovery.py` 负责无结果筛选的保留与核验。服务器 `tests.log` 保存五项通过的测试，包括真实 BF16 存储和旧 float64 差值指纹兼容性。
- `artifact_manifest.json` 保存报告、图表、源码和原始记录的字节哈希。模型、训练示范与缓存仍留在服务器。
- `weights_inventory.json` 与 `anchor_inventory.py` 独立核对八张 V 的名称、形状、差值指纹、范数及参数空间方向；范数不能直接当成行为贡献比例。`runtime.json` 记录 30 个已锁定复用项目模块、实际版本及 GPU。
- `history/` 原样保存四份用于追溯第 30 步来源的历史说明，`sources.json` 记录原工作区路径和字节哈希；源文档内的相对链接仍按原目录解释。

完整第 30 步策略的重建只替换训练中实际可变的 155 个张量。LR 的其余 345 个张量与父模型逐值相同；LF 第 30 步文件为部分检查点，训练完成记录核验冻结部分未变，因此继承同一父模型。所有权重赋值发生在新的 autocast 上下文之前，避免旧的类型转换缓存干扰资格比较。

主修复仍只有一个 F 生成动作。历史 V 差在相同当前观察、动作草稿和去噪时刻重算。规范真实句子完整输入；目标标签只用于事后评分。完整供体原生运行是独立资格参照，不参与修复回合执行。

## 复算

纯统计只需 Python 标准库和此前两个归档。在仓库根目录运行：

```powershell
python -m unittest discover -s tests -p 'test_k25_anchor*.py' -v
python scripts/k25/summarize_anchor.py docs/results/2026-11-23/k25_anchor --check
python docs/results/2026-11-23/k25_anchor/report_generation.py
```

复算验证全部新轨迹与引用轨迹、网格、动作哈希、两种符号的首次相同读出、延后组干预前与晚期原生控制一致，以及既有父模型条件。绘图另需 Matplotlib、NumPy：

```powershell
python scripts/k25/plot_anchor.py docs/results/2026-11-23/k25_anchor docs/figures/2026-11-23/k25_anchor
```

首次运行目录为 `/home/lzx/cl_binding/.server/k25_anchor_20261123_01`，恢复运行目录为 `_02`。实际恢复与执行命令：

```bash
source /home/lzx/cl_binding/.server/k1_screen_20261101_01/code/env.sh
anchor_run=/home/lzx/cl_binding/.server/k25_anchor_20261123_02
export PYTHONPATH="$anchor_run:$PYTHONPATH"
K25_REAL_TENSOR_TESTS=1 "$PY" -m unittest discover -s "$anchor_run/tests" -p 'test_k25_anchor*.py' -v
"$PY" "$anchor_run/anchor_check.py" prepare --resume-from /home/lzx/cl_binding/.server/k25_anchor_20261123_01
"$PY" "$anchor_run/capture_runtime.py"
nohup "$PY" -u "$anchor_run/run_anchor_check.py" --workers 8 > "$anchor_run/queue.log" 2>&1 < /dev/null &
"$PY" "$anchor_run/summarize_anchor.py" "$anchor_run"
```

重跑需新的目录、服务器既有离线环境和原检查点，并保留相邻的历史运行目录。准备入口拒绝覆盖已锁定方案，工作入口拒绝覆盖分片。

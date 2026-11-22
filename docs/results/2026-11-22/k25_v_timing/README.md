# K25-V：只改变 V 修正的时间窗

主报告：[K25-V.md](../../../K25-V.md)。本实验接续相邻的 `k25_order_restore` 归档，固定其中两条第 1000 步续训模型及各自的第 30→1000 步八张 V 权重差。

## 网格与统计单位

两条学习组合为 Left＋Right→Front 和 Left＋Front→Right。每条组合测试两条旧指令、20 个配对物理初态（800–819）、原生及七个时间窗的正反修正，网格共 1200 个 400 步回合。七个时间窗按一基控制步表示为 1–100、41–140、81–180、161–260、1–400、81–400、161–400；JSON 中保存零基半开区间。增益全部固定为 5.294038631688923，不在本批数据上重新选择。

四个 100 步窗口用于等时长比较。持续到 400 步的条件同时改变作用时长，须单独解释。同起点的短窗口和持续窗口在撤除时刻之前必须逐值相同；继续与停止的差异才来自撤除之后。

每个目标、模型、条件的分母是 20 个初态。1200 是执行回合数，不是独立样本量。其中 200 回合重放上一批已有条件：80 原生、80 条 Front 续训线的延后正反条件、40 条 Right 续训线的原强度延后正向条件。它们用于检验复现一致性，不算新的独立证据。其余条件仍复用这 20 个已见初态；本轮是针对时间混杂的机制检验。

## 数据

- `protocol.json`、`cases.json`：预先锁定的时间网格、权重和源码 SHA256、初态、噪声、场景几何及版本。
- `*_*.jsonl.gz`：每片的逐回合数据。每个回合含 400 步书本 XYZ、夹爪 XYZ、实际裁剪后 7D 动作、双指接触、区域与放置标记；还含每十步的物理／控制器／图像签名、动作哈希和干预块的读出距离及修正系数。
- `*_COMPLETE.json`：分片行数资格、压缩文件哈希、八张 V 差的实际张量指纹、单模型及均值守恒闸门、已核对的旧条件键。
- `summary.json`：由密集轨迹重算的首达、终态、曾放好、抓取、干预前已确认事件、干预后首次到达、配对转移、共用前缀和正反对照审计。
- `queue.log`、分片日志、`queue_COMPLETE.json`、`capacity.json`：实际运行、完成及并发设置。资源并发不改变模型、噪声或时间掩码。
- `v_timing.py`、`v_timing_protocol.py`、`summarize_v_timing.py`、`run_v_timing.py`：执行及复算入口的源码快照。上一轮的 `order_restore.py`、`baseline_self_restore.py` 和 `dependencies/` 是固定依赖，哈希可从相邻归档核验。
- `runtime.json`：实际导入的项目模块与上一轮快照逐一核验，包版本及环境另行记录。上一轮队列后来新增 `reserve` 分支，属于本轮未导入、未执行的工作流脚本，差异单独保留，不混作当前推理依赖。
- `artifact_manifest.json`：归档文件及报告、脚本、图表的字节哈希；不包含模型、演示或缓存。

`contacts` 位 0／1 分别表示左右指接触。`regions`、`success` 的位 0–3 依次为 left、front、right、back。首达要求书本抬高超过 3 厘米且连续五步位于目标水平区域；时间记稳定段的第一步。干预前事件只有五个确认步都在干预前才计入。例如第 78–82 步的段不能声称第 80 步已经确认。终态是第 400 步 LIBERO 放置判据，不额外证明松爪后的长期稳定。

“抓取至介入前的旧向减新向接近量”是书本相对两个格中心的水平距离缩短之差；从稳定抓取的确认步计算，单位 mm。它只描述几何趋势，不能独自判定策略的意图。未到错格也不能排除已经偏向错误方向。

规范旧句同时参与参考构造。其零距离匹配不能作为语言泛化证据，参考响应重合时按原算法的 A、B、N 次序处理距离并列。正确目标仅在评分时使用，不传入动作采样器。单模型在当前状态重算同一条 V 修正规则；不同时间窗会改变后续状态，因此实际数值 Δd 不保证相同。

## 本地复算

只需要 Python 标准库；不需要模型、CUDA 或仿真器。在仓库根目录运行：

```powershell
python -m unittest discover -s tests -p test_k25_v_timing.py -v
python scripts/k25/summarize_v_timing.py docs/results/2026-11-22/k25_v_timing --check
```

`--check` 会重算全部 1200 条密集轨迹、检查准确网格、压缩文件和动作哈希、前缀逐值一致、首次正反读出相同且系数相反，以及上一批 200 条条件的结果和第 80 步状态签名。它不会覆盖归档统计。绘图另需 Matplotlib、NumPy：

```powershell
python scripts/k25/plot_v_timing.py docs/results/2026-11-22/k25_v_timing docs/figures/2026-11-22/k25_v_timing
```

## 服务器执行

本轮目录为 `/home/lzx/cl_binding/.server/k25_v_timing_20261122_01`，依赖上一轮的相邻目录 `k25_order_restore_20261122_01`。需要服务器现有的离线环境、检查点与 LIBERO 素材。本轮未训练新模型，未重标定增益。

```bash
source /home/lzx/cl_binding/.server/k1_screen_20261101_01/code/env.sh
timing_run=/home/lzx/cl_binding/.server/k25_v_timing_20261122_01
"$PY" "$timing_run/v_timing.py" prepare
nohup "$PY" -u "$timing_run/run_v_timing.py" --workers 4 --lanes 8 > "$timing_run/queue.log" 2>&1 < /dev/null &
"$PY" "$timing_run/summarize_v_timing.py" "$timing_run"
```

准备和仿真入口拒绝覆盖既有方案及分片。重跑应新建运行目录，保留相邻依赖关系，重新记录哈希。`capacity.json` 可在运行时限制并发，实际变化见 `queue.log`。

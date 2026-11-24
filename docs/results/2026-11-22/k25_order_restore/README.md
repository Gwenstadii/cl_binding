# K25 单模型修复：续训组合、晚期检查点与措辞补验

日期：2026-11-22。主报告见 [K25-order-validation.md](../../../K25-order-validation.md)。

## 复算

统计复算只需要 Python 标准库，不需要权重、CUDA 或仿真器。在仓库根目录运行：

```powershell
python scripts/k25/summarize_order_restore.py select docs/results/2026-11-22/k25_order_restore --check
python scripts/k25/summarize_order_restore.py summary docs/results/2026-11-22/k25_order_restore --check
```

`--check` 逐项重算并与归档比较，不覆盖已锁定的选择与结果。绘图另外需要 Matplotlib 和 NumPy：

```powershell
python scripts/k25/plot_order_restore.py docs/results/2026-11-22/k25_order_restore/summary.json docs/figures/2026-11-22/k25_order_restore.png
```

## 数据与控制

- `protocol.json`：初始方案、八个开发初态、二十个新测试初态、检查点哈希、剂量和选定规则。
- `cases.json`：完整物理初态、观察签名、复位与重规划噪声种子。测试编号 800–819，没有按抓取或搬运成绩筛选。
- `selection.json`：只依据开发集选择的强度。`selection_source.py` 保存做出选择时的原始统计脚本。
- `standard_protocol.json`：开发集三个剂量均未恢复 Left＋Front→Right 后，另保留原 5.294 倍直接迁移对照。该强度在查看此组合正式结果前固定。
- `dev_*`、`main_*`、`phrase_*`：每个工作分片的逐回合 JSONL 和完成标记。`main` 从复位开始，修正窗口为控制步 81–180，全程 400 步；`phrase` 使用完整改写句。
- `standard_*`：Left＋Front→Right 的原 5.294 倍正向对照；`parent_*`：其父模型在同批测试初态上的原生资格参照。父模型不参与修复动作生成。
- `reserve_protocol.json`、`reserve_cases.json`、`reserve_*`：看到主测试集的 Left 代价后，另用未看过的 900–919 初态确认较保守的 1 倍修正。剂量从八个开发初态选出，四个条件是原生、1 倍正向、1 倍反向及 5.294 倍正向；该后续实验与最初方案分别报告。
- `lf_right_training*`：新增的 1000 步 Right 续训日志、配置与冻结核验；权重留在服务器。
- `summary.json`：完整网格审计、按目标的首达和终态、同初态成对成绩、逐回合转移、正反首次读出与状态核对。
- `runtime.json`、`dependency_sha256.json`、`dependencies/`：实际环境和项目代码依赖快照。依赖快照供追溯，执行入口是仓库 `scripts/k25/order_restore*.py`。
- `baseline_k25_final_summary.json`、`baseline_method.json`、`baseline_self_restore.py`：既有 K25-final 的统计、配置和实现快照；本轮已完整复算一致，原有 780 回合不计作新增实验。

相同初态在条件、检查点、措辞和学习组合之间复用。主测试集与保守剂量确认集各有 20 个不同物理初态；每条指令在各自集合内的统计单位为 20。回合总数、控制块和去噪次数都不能替代独立样本数。`A/B` 的目标名称依学习组合变化：Front 续训线是 Left/Right，Right 续训线是 Left/Front。

## 仿真重跑

实际目录是 `/home/lzx/cl_binding/.server/k25_order_restore_20261122_01`。需要服务器原有数据、权重、LIBERO 素材和归档模块；本目录不包含模型或演示数据。代码依赖快照保留原项目相对位置，第三方包版本见 `runtime.json`。

在既有离线运行环境中依次执行：

```bash
source /home/lzx/cl_binding/.server/k1_screen_20261101_01/code/env.sh
task_run=/home/lzx/cl_binding/.server/k25_order_restore_20261122_01

"$PY" "$task_run/order_restore.py" train
"$PY" "$task_run/order_restore.py" prepare
"$PY" "$task_run/run_order_queue.py" dev --workers 8
"$PY" "$task_run/summarize_order_restore.py" select "$task_run"
"$PY" "$task_run/order_restore_standard.py" prepare
"$PY" "$task_run/run_order_queue.py" main --workers 8
"$PY" "$task_run/run_order_queue.py" standard --workers 8
"$PY" "$task_run/run_order_queue.py" parent --workers 8
"$PY" "$task_run/order_restore_conservative.py" prepare
"$PY" "$task_run/run_order_queue.py" reserve --workers 8
"$PY" "$task_run/summarize_order_restore.py" summary "$task_run"
```

这些是本轮命令顺序，已有完成输出不会被训练和方案锁定入口覆盖。要重新仿真，应建立新的运行目录并记录新哈希；纯统计复算使用上面的 `--check`。

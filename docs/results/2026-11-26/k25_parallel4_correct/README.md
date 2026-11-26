# K25-parallel4-correct 证据

2026-11-26，在F4-B24上新执行320个在线回合。20个初态800–819 × 四句 × 四种执行条件。请求目标原生BDDL谓词首次为真即成功并结束，最多400步。

主报告：[K25-parallel4-correct](../../../K25-parallel4-correct.md)。

- `manifest.json`：模型、控制条件、代码及输入SHA256。内嵌模型登记的`evidence.mode`描述此前的模型选取证据；本轮运行性质由顶层清单、320条新轨迹与完成标记确定。
- `case_0.jsonl.gz`至`case_19.jsonl.gz`：每片16条JSON回合，含实际动作、物体／末端位置、接触、原生谓词位图、停止步和投影诊断。
- `case_*_COMPLETE.json`、`COMPLETE.json`：逐片及全部运行完成证明。
- `summary.json`：逐目标成功、停止步、同初态成组及相对原生F的配对得失。
- `audit.json`：权重身份、轨迹完整性、前缀及窗口核验。
- `artifact_manifest.json`：服务器归档的71个文件的字节数和SHA256。本说明与`.gitattributes`为下载后新增，不在该清单内；下载后71项均已逐一核对通过。
- `F4-B24.json`、`cases.json`、`reused_calibration.json`：模型登记、固定初态和原有130项标定。未重新标定或训练。
- `parallel4_correct.py`、`native_event_protocol.py`：实际服务器执行源码快照；工作区入口见`../../../../scripts/k26/parallel4_correct.py`。
- `controller.log`、`worker_*.log`、`status.json`：运行日志及最终状态。

运行依赖既有项目服务器布局。原parallel数学实现对应仓库`scripts/k25/parallel_four.py`及其`parallel_subspace.py`依赖；实际加载路径与原源码SHA256记录于清单。先加载独立环境，再切换目录，避免环境脚本改变工作目录：

```bash
source /home/lzx/cl_binding/.server/k1_screen_20261101_01/code/env.sh
cd /home/lzx/cl_binding/.server/k25_parallel4_correct_20261126_01
# 原执行入口；controller只允许在无manifest的新目录中启动，避免覆盖完成记录。
"$PY" parallel4_correct.py controller
"$PY" parallel4_correct.py summarize
```

所有模型、数据和环境缓存留在服务器，证据中不含这些文件。320个新回合复用历史20个初态，不构成新留出集。

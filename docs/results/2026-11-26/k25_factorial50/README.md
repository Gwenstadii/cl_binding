# K25-factorial50 启动归档

2026-11-26，评测已授权，尚未完成。详见[方案与待验收报告](../../../K25-factorial50-result.md)。

- `protocol.json`：6个检查点、5组方法、两个窗口、固定增益、模型与输入SHA。
- `cases.json`：50个共享保存初态，800–819复用，820–849新增。
- `reuse_manifest.json`：1200条完整旧记录的来源与SHA，新的9000条总记录还需补跑7800条。
- `STARTED.json`：服务器后台主控启动凭据，不是完成证明。
- `launch_verification.json`：启动后8路运行、6模型门禁和新回合已实际产生的核验快照，不是最终验收。

服务器：`/home/lzx/cl_binding/.server/k25_factorial50_20261126_01`。
工作区入口：`scripts/k26/factorial50.py`和`launch_factorial50.sh`；对应副本部署为服务器`factorial50.py`与`launch.sh`。

完成后应读取服务器`COMPLETE.json`，核验所有分片及每句50次，再下载`summary.json`与自动生成的同名结果MD更新主报告。本目录当前不含完成后的统计，不把启动标记当作实验结果。

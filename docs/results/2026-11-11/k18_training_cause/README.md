# K18 结果与复现记录

本目录记录 R1（Left＋Right）→ Front 的早期训练复现、固定输入响应、闭环评估和 AdamW 一步监督归因。原始模型与优化器文件保存在服务器对应检查点目录。实验使用的服务器运行目录为 `/home/lzx/cl_binding/.server/k18_training_cause_20261111_01`，运行环境为 `/home/lzx/cl_binding_stage2a`。

- [summary.json](summary.json) 汇总固定快照、验证示范、闭环和训练分叉。
- [replay_audit.json](replay_audit.json) 核验 6000 步训练前缀、模型哈希、优化器、调度器与随机状态。
- [native_audit.json](native_audit.json) 核验 300 个闭环记录、共同抓持边界、K14 两个端点的逐回合比较，以及附加通道探针复算的一致性。
- [front_phases.json](front_phases.json) 记录 Front 训练数据的几何阶段标签和动作对齐。
- [phase_audit.png](phase_audit.png) 显示抽查的六条原始示范关键帧。
- [k18_timeline.png](k18_timeline.png) 根据已验收读数绘制时间线、闭环结果和动量对照。
- `records/` 保存每次探针、每个闭环回合的结果，以及每个训练分叉的实际更新结果。闭环记录包含首个区域、首次时刻、曾成功和终态。完整动作轨迹未导出。
- `train/` 和 `logs/` 保存逐步训练日志和任务日志。服务器现场保留了运行时的早期自检日志。
- `code/` 保存此次配置、原训练复现入口、固定输入与验证探针、分叉归因、核验和汇总脚本。
- [evidence_sha256.json](evidence_sha256.json) 列出从服务器归档的证据文件哈希。报告图和本说明在本地生成。

可在本目录用 Python 3.10 或更新版本从原始记录重新验收汇总。

```powershell
python code/summarize.py records --complete
python code/plot.py summary.json k18_timeline.png
```

固定抓持快照及几何判据继承 K14。模型装载与原生控制继承 K15 的运行环境。训练配方和历史代码审计见 [K18 计划](../../../2026-11-10_K18_从Front训练更新追溯目标替代_实验计划.md)。新增锚点按 0–3000 步的探针时间线选择，阶段×通道交叉拆分及 H/Z/R/G 增益为本轮追加的探索性读数。

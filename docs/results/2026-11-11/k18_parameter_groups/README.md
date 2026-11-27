# K18-C 参数更新定位与闭环证据

服务器运行目录为 `/home/lzx/cl_binding/.server/k18_parameter_groups_20261111_01`。本目录保存从 R1 续训 Front 的真实权重改动分组对照。原权重、优化器和固定快照沿用 K18-A/B。K18-C 的两档闭环都使用原指令原生生成动作。

- [summary.json](summary.json) 汇总固定输入读数、配对区间、原生闭环、注意力细分和旋转来源归因。
- [protocol.json](protocol.json) 记录主分析选择规则。[secondary_protocol.json](secondary_protocol.json) 记录观察第 100 步失败构成后追加的第 1000 步闭环。
- `audit_*.json` 记录 155 个可训练张量的互斥分组、各张量更新范数、两端权重哈希和 345 个其余张量一致性的核验。
- `probe_*.json` 保存每窗 22 个参数配置的固定输入读数。每个配置使用同一批 120 个画面／动作草稿／去噪时刻组合。
- `attention_split_*.json` 保存每窗 14 个配置的探索性子组探针。
- `rotation_bridge_*.jsonl` 保存 3 个锚点的 24 批完整／去旋转 AdamW 对照及注意力交换结果。
- `native_*_w*.jsonl` 保存 480 个新闭环回合。首个区域与任务成功判据按全部 400 步计算。轨迹按每 10 步导出，用于查看阶段进展。
- [reference_1000.jsonl](reference_1000.jsonl) 是从 K18-A 复用的 60 条第 1000 步基线。第 30 步基线复用本轮主窗闭环。
- `code/` 保存实验、聚合、作图脚本。`logs/` 保存执行日志。[evidence_sha256.json](evidence_sha256.json) 记录服务器归档文件哈希，作图脚本与图在本地追加。
- [参数定位及闭环结果图](k18_parameter_groups.png)

可在本目录重新汇总开发数据。区间按 20 个初态配对重抽样，闭环和额外探针沿用同一批开发初态。

```powershell
python code/summarize.py .
python code/plot.py summary.json k18_parameter_groups.png
```

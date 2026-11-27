# K19 实验证据

服务器运行目录为 `/home/lzx/cl_binding/.server/k19_retention_20261111_01`。权重、训练状态和 R1 教师库保存在服务器。

- [summary.json](summary.json) 汇总开发筛选、独立确认、固定输入与 Front 验证示范读数、层组定位和实际优化器对照。
- [protocol.json](protocol.json) 记录原始设计，[replay_extension.json](replay_extension.json) 与 [soft_extension.json](soft_extension.json) 记录开发期追加的动态回放。
- [selection.json](selection.json) 记录开发筛选及确认候选。候选在新初态评估之前锁定。
- [holdout.json](holdout.json) 记录 20 个新物理初态和共同抓持前缀。候选筛选只使用已有开发状态。
- `mu_layers_*.json` 记录两个时间窗的真实参数增量分组替换，每组保留 120 个配对探针。
- `gradient_*.jsonl` 记录 4 个历史锚点每点 8 批，包含原始梯度方向及三种实际 AdamW 更新结果。
- `dev_w*.jsonl`、`confirm_w*.jsonl` 记录逐回合首次、曾成功、终态。旧任务从相同抓持状态接手，Front 从复位开始。
- `runs/` 和 [training_audit.json](training_audit.json) 记录训练日志及匹配 Front 批次的核验。教师源由 [teacher_READY.json](teacher_READY.json) 记录。
- [evidence_sha256.json](evidence_sha256.json) 记录归档哈希。作图脚本与图在本地生成。

```powershell
python code/summary_k19.py .
python code/plot.py summary.json k19_endpoints.png
```

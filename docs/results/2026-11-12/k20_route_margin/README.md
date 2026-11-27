# K20 分流余量实验

服务器运行目录为 `/home/lzx/cl_binding/.server/k20_route_margin_20261112_01`。训练权重、优化器状态和分流查询库保存在服务器。正式报告记录两条训练流、对照组以及实验的适用范围。

- [summary.json](summary.json) 汇总两个场景组的首次到区、终态、逐场景变化、前四段实际位移、固定动作响应和训练一致性。
- [protocol.json](protocol.json) 记录实际采用的训练与验收设置。[protocol_initial.json](protocol_initial.json) 保留早期范围草案。
- [calibration.json](calibration.json) 收录 112 个校准闭环回合、实际位移、代理分数和拟合系数。
- `dev_*.jsonl`、`confirm_*.jsonl` 和 [baseline.jsonl](baseline.jsonl) 保存新增评估和配对参考回合。两个参考模型的 Front 回合重跑以补齐阶段读数。
- `transport_*.jsonl` 保存相同接手状态上前 10、20、30、40 步的实际书本位移。
- `runs/` 包含 1000 步训练日志和检查点签名。固定输入响应与 Front 示范去噪记录在 `probe_*.json`。
- [evidence_sha256.json](evidence_sha256.json) 记录归档与服务器查询库的哈希。

从本目录复算汇总。

```powershell
python code/summary_k20.py .
```

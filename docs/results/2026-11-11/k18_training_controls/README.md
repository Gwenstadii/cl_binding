# K18-D 训练反事实与独立闭环

服务器运行目录为 `/home/lzx/cl_binding/.server/k18_training_controls_20261111_01`。各组权重和终点 AdamW 状态保存在其 `runs/` 下。

- [protocol.json](protocol.json) 记录四组设计、三随机流、掩码和范数匹配规则。
- [holdout.json](holdout.json) 记录在观察任何新分支结果前锁定的 20 个新物理初态和共享抓持前缀。`candidates.json` 和 `screen_*.json` 保留生成和筛选结果。
- [summary.json](summary.json) 记录 780 个主闭环回合的首次、曾成功和终态、逐初态配对转移、固定输入 (mu/d)、K/V 更新投影，以及验证示范误差。
- [fit_summary.json](fit_summary.json) 记录 1000 步去旋转与示范误差接近的预设完整监督检查点之间的追加比较。
- [front_validation_phases.json](front_validation_phases.json) 记录十条独立 Front 示范的动作对齐和粗阶段标签。
- [replay_weight_audit.json](replay_weight_audit.json) 记录历史随机流完整监督的四个检查点与 K18-A 逐张量相等。
- `runs/` 保存全部 12000 步的训练日志及各检查点校验标记。`native_w*.jsonl` 和 `fit_w*.jsonl` 保存逐回合结果。
- `probe_*.json` 保存每组检查点的 120 个固定输入和 90 个验证示范加噪输入的原始读数。
- [evidence_sha256.json](evidence_sha256.json) 记录服务器归档文件哈希。作图脚本和图在本地生成。
- [K18-D 训练与闭环结果图](k18_training_controls.png)

可用 Python 从原始记录重新汇总主结果。

```powershell
python code/summarize.py .
python code/plot.py summary.json k18_training_controls.png
```

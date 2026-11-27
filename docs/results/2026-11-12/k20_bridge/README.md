# K20 的 μ/d 桥接实验

服务器运行目录为 `/home/lzx/cl_binding/.server/k20_bridge_20261112_01`。两个新训练检查点和优化器状态保存在服务器。实验复用原 K20 的抓持状态、校准和配对对照。

- [summary.json](summary.json) 记录闭环首次到区和终态、逐回合转移、真实位移分解、训练一致性和固定动作响应。
- [protocol.json](protocol.json) 记录训练约束、评估初态及干预范围。
- `patch_*.jsonl` 保存同草稿动作输出互换的 240 个闭环和前四段物体位移，`native_*.jsonl` 保存投影训练模型的 120 个原生闭环。
- `probe_*.json` 保存固定去噪输入读数和新模型的实际首四段位移。
- `runs/` 保存两条训练流的逐步日志和检查点签名。
- [role_verification.json](role_verification.json) 记录 K20 权重和两句原生动作的逐值一致性检查。
- [evidence_sha256.json](evidence_sha256.json) 记录本目录与原 K20 参考文件的哈希。

在当前目录复算完整汇总。

```powershell
python code/summary_bridge.py . ../k20_route_margin
```

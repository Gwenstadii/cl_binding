# 原生成功事件重评分

主说明：[K25-Back-LIBERO](../../../K25-Back-LIBERO.md)。模型：[F4-B24登记](../../../../configs/k25_models/F4-B24.json)。

本目录是已有完整轨迹的离线重评分：260条原标准Back开发轨迹、400条原whole4-new对照轨迹及80条复用父模型轨迹，共740条。没有训练、模型前向或新增仿真。92个源分片的SHA256、请求目标首次原生成功步数、成功后应终止的步数及旧400步终态均在`rescore.json`中。

位掩码通过每条轨迹的原生`ever`与逐步谓词交叉核对，唯一对应Left1、Front2、Right4、Back8。部分回合不补成20例，也不用几何区域首达代替原生成功。

`PROTOCOL.json`记录新主指标与官方实现出处；`late_confirmation`只归档先前已经完成的旧释放扩展分支确认标记，不是本轮新增评估，也不用于跨初态强行排名。F4-B24来自原40示范标准分支24000步，并保留与此前whole4／parallel4相同的权重和处理器。

复核命令（在项目根目录，用单独输出文件保留本归档）：

```text
python scripts/k26/rescore_native_event.py --output .tmp/native_event_recheck.json
python -m unittest discover -s tests -p test_native_event_protocol.py -v
```

模型完整权重仅在服务器，未加入Git。完整推理封装的500张状态张量已经与原冻结权重／155张续训权重逐张验证相等；没有复制不匹配的R1优化器状态。既有所有训练继续保持停止。

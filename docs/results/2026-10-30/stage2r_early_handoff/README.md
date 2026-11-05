# 提前交接与换指令：完成归档

本批192条分支，48个交接状态×两个接手模型×左前两条接手指令。40／80／120指已执行控制步；所有分支在原第400步结束。

- `launcher.json`、`dispatch.json`、`registered_config.json`：派发、冻结配置、阶段耗时。
- `evaluation/`：全部原始逐步动作、位置、目标谓词、重规划噪声与工程签名。
- `audit.json`与`audit_local_v1.json`：服务器与本地完整日志审计，内容一致。
- `movement_independent_v1.json`：另一套滑窗实现重算主移动口径与双指令配对。
- `results.csv`：48行逐条件汇总，每条件4分支；`instruction_pairs.csv`：24行双指令汇总。
- `interpretation_summary.json`及`failure_types.json`：开发条件的派生描述，不当作独立场景总体成功率。
- `source_context/`：前批已封存原生轨迹的描述，不是本轮新增模型回合。

只有两个原始物理初态、两个噪声和一个历史适应过程。120步许多状态已经到过目标，须看pretarget_eligible；新目标按接手指令评分，不能沿用前段指令。

本批没有训练、CA补丁或新权重下载。

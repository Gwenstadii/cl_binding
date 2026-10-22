# Does Continual Adaptation Erode Acquired Instruction–Goal Selection?

项目仓库：[Gwenstadii/cl_binding](https://github.com/Gwenstadii/cl_binding)。当前目标是在双 RTX 3090 上完成 VLA 持续学习与语言条件目标选择的最小起步验证。开源代码已固定版本；已完成工程冒烟及6000步R0开发运行，但R0尚未合格，尚未开展FT/ER现象对照或取得正式研究结论。

## 当前进度与接手入口（2026-10-22）

正在进行[单任务起点与四指令探测](docs/2026-10-22_单任务起点与四指令探测.md)：只训练左格40条示范，按预定候选的左格开发成绩选点，达到8/10后冻结；再在同一组独立初态测试前／左／右／后指令，并与训练前基座比较。它先检验单任务操作起点与目标偏置，不将未学过任务的失败自动解释为持续学习遗忘。

先读[科研交接说明](docs/handoff/2026-10-21/先读这个_科研交接.md)，再看[验收与诊断](docs/handoff/2026-10-21/证据/r0_20261021_174358/R0运行验收与诊断.md)和[结果图](docs/handoff/2026-10-21/证据/r0_20261021_174358/r0_diagnostics.png)。服务器环境和数据已在，无须重复安装或下载。

最新结果见[修正重训阶段汇报](docs/2026-10-22_R0修正重训_阶段汇报.md)：原配对003000完整开发10/30、006000为16/30；修正时序并同预算复跑后分别为10/30、11/30，均未达到逐目标8/10，未获得合格R0。新一轮已正常结束，共47分19秒。下一步建议做单条成功示范的闭环拟合诊断。此前的[时序核验与执行记录](docs/2026-10-22_时序核验与修正后重训.md)和[冒烟复核与下一步计划](docs/2026-10-22_冒烟复核与下一步计划.md)保留为历史快照。

- 旧三目标120条训练/30条验证；归一化只使用旧训练集，后格不参与R0训练或统计。
- 6000步开发训练完成；12个checkpoint保存/严格重载均通过。GPU0训练、GPU1评估。
- 原每点9回合开发筛查，最好5/9，最后4/9；现已补两个检查点的完整10初态开发，R0仍未达标。独立20初态资格尚未运行。
- 专家动作回放10/12曾成功，right两条失败仍保留。不能将当前低成功率解释为持续学习遗忘。
- FT、普通ER、配对现象验证、差分蒸馏和覆盖选样均尚未开始；建议修改单列供接手者讨论，没有擅自更改资格标准。

原始运行记录日期为2026-10-21；历史文档中的“未推送GitHub”等描述属于当时快照。本次代码和交接按[协作交付流程](CONTRIBUTING.md)通过分支/PR交付。**PR合并前，单独pull main拿不到PR内新增代码。**

`essay/` 仅供本地阅读，不加入 Git；模型、数据、缓存和训练输出也不随源码上传。

服务器位置、独立环境安装、已有权重复用和后续下载命令见 [scutlab 环境与资源下载](docs/scutlab_环境与资源下载.md)。

真实数据转换、5 步训练/保存重载和四任务短程闭环已在双 3090 上验证。按 [下一步冒烟的服务器操作指引](docs/下一步冒烟的服务器操作指引.md) 逐步重跑；这些工程检查不代表正式任务能力或论文实验结果。

## 已取得的两个仓库

| 目录 | 上游 | 固定提交 | 用途 |
|---|---|---|---|
| [third_party/lerobot](third_party/lerobot) | [huggingface/lerobot](https://github.com/huggingface/lerobot) | `afbf3f14a13bdad3e5990e2bed629c6539605658` | SmolVLA 模型、数据工具和原生训练/评估入口 |
| [third_party/LIBERO-reference](third_party/LIBERO-reference) | [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | `8f1084e3132a39270c3a13ebe37270a43ece2a01` | 任务定义、配置、演示下载程序及环境实现参考 |

详细来源、筛选规则和文件统计见 [UPSTREAM_REPOS.json](UPSTREAM_REPOS.json)。LeRobot 的 [Apache-2.0 许可证](third_party/lerobot/LICENSE) 与 LIBERO 的 [MIT 许可证](third_party/LIBERO-reference/LICENSE) 均保留；使用、修改和发表时按对应上游要求署名与引用。

这两个目录是从官方仓库浅克隆、按代码范围展开的**固定版本快照**。模型权重、数据文件、视频、图片和大体积仿真素材没有展开。LeRobot 的源代码、脚本、配置和文本说明，以及 LIBERO 的代码与 BDDL 任务定义已保留；被省略的测试素材不属于已完成的测试验证。

## 为什么可以直接放进自己的 GitHub 仓库

`third_party` 下保留的是普通文件，没有嵌套 `.git` 指针，也没有依赖 Git submodule。你将整个项目推送后，服务器普通 `git clone` 就能取得这里已保存的代码，不需要额外的 submodule 初始化。

原始浅克隆元数据仅在本机的 `.upstream_git/` 中保留，用于核查来源；它已被忽略，不随项目上传。上游提交号由版本清单保存。以后不要直接在普通快照目录执行 `git pull`；升级时应重新取得指定提交、审查差异并更新版本清单。

项目使用根目录 Git 历史管理这些普通源码文件；服务器克隆本项目即可取得已保存的代码。上游资源与本地阅读资料由 `.gitignore` 排除，升级代码时同步维护来源清单。

## 已核对的四个最小实验任务

目录：[LIBERO-90 BDDL](third_party/LIBERO-reference/libero/libero/bddl_files/libero_90)。以下四份定义均存在：

```text
STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy.bddl
STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy.bddl
STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy.bddl
STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.bddl
```

前、左、右是三个旧条件，后格是 N1。定义存在只说明题目可查，不代表演示、初态与运行时已齐备，更不代表模型已会执行。

## 模型和数据留到服务器

- 主训练基座：`lerobot/smolvla_base`，服务器已取得，固定版本见资源清单。
- 可选工程参照：服务器 `$HOME/hf_models/smolvla_libero` 已有完整文件，可直接复用；它不直接充当干净的 R0-small。
- 上述四条件的原始演示已在服务器补齐；每目标一条轨迹的冒烟数据已转换，旧三目标和后格 N1 分开存储。
- 服务器已通过 `hf-libero` 安装初态并取得仿真素材，四条件复位和双相机渲染已通过。`LIBERO-reference` 仍仅作参考，不直接安装为同名运行时。
- LeRobot 运行时优先采用其固定依赖中的 `hf-libero` 路线；在服务器核对运行资源、实际导入路径和四条件支持。不要同时把参考仓库安装进去覆盖同名 `libero` 包。

代码克隆不会自动获得上述文件。模型与演示数据应在服务器单独下载、固定 revision，并放在被忽略的数据目录或服务器共享存储。官方整理的 `lerobot/libero` 是 40 任务数据，不能默认替代这里需要的 LIBERO-90 四条件。

## 下一步的执行顺序

1. 接手先执行[只读检查脚本](docs/handoff/2026-10-21/接手只读检查.sh)，核对服务器、分支、提交和结果。不要重跑旧训练目录。
2. 先解决R0旧能力不稳定；[单轨迹诊断候选](docs/handoff/2026-10-21/证据/r0_20261021_174358/下一轮单任务诊断_待确认.md)及[筛选规则等讨论项](docs/handoff/2026-10-21/先读这个_科研交接.md#7-建议同学了解并讨论的调整)均未执行，需先锁定方案。
3. 开发选点后才做一次独立R0资格，再从同一合格R0分别开展FT-N1与普通ER-N1；受限ER缓存和公平对照仍需实现/核验。
4. 新任务确有学习增益后，做R0/FT/ER前后配对目标选择诊断。现象成立再进入机制与方法消融。

数据审计、多目标事件评分和共同初态评估已实现。完整开发/资格入口存在，但入口可运行不等于模型已合格。原计划180次基础执行不包含本轮108次开发筛查。

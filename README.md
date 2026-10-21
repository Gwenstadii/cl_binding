# Does Continual Adaptation Erode Acquired Instruction–Goal Selection?

项目仓库：[Gwenstadii/cl_binding](https://github.com/Gwenstadii/cl_binding)。当前目标是在双 RTX 3090 上完成 VLA 持续学习与语言条件目标选择的最小起步验证。开源代码已固定版本；环境与资源状态以服务器检查记录为准，目前没有新训练结果。

`essay/` 仅供本地阅读，不加入 Git；模型、数据、缓存和训练输出也不随源码上传。

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

- 主训练基座：`lerobot/smolvla_base`，当前未下载。
- 可选工程参照：`lerobot/smolvla_libero`，当前未下载，也不直接充当干净的 R0-small。
- 上述四条件的原始演示与转换后数据：当前未下载。
- LIBERO 的初态、纹理、网格等运行资源：当前未下载。`LIBERO-reference` 因此仅作参考，不能直接当成已完整安装的仿真环境。
- LeRobot 运行时优先采用其固定依赖中的 `hf-libero` 路线；在服务器核对运行资源、实际导入路径和四条件支持。不要同时把参考仓库安装进去覆盖同名 `libero` 包。

代码克隆不会自动获得上述文件。模型与演示数据应在服务器单独下载、固定 revision，并放在被忽略的数据目录或服务器共享存储。官方整理的 `lerobot/libero` 是 40 任务数据，不能默认替代这里需要的 LIBERO-90 四条件。

## 下一步的执行顺序

1. 在双 3090 Linux 服务器建立独立环境，从固定版本 LeRobot 安装需要的依赖；先核对驱动、PyTorch 和视频解码兼容性。
2. 下载模型、四条件演示和必要仿真资源，核查数据转换与相机/动作含义。
3. 一张卡跑训练链路冒烟，另一张卡核对环境复位、多目标判据和闭环推理。
4. 重新从主基座联合学好三个旧条件，建立合格 R0-small，再分别续训 FT-N1 和普通 ER-N1。
5. 按原计划完成 180 次基础执行：资格 60 次，加三份模型的三旧一新考试 120 次；开发调试、故障重跑与独立确认另计。

本次没有额外克隆 openpi、OpenVLA-OFT 或 LIBERO-CF；它们按主线证据在后续加入。原生训练/评估入口已经取得，但受限回放、四条件数据转换与前后配对的多目标终点诊断仍须接通，不能直接视为现成训练管线。

# scutlab：环境、已有资源与后续下载

日期：2026-10-21。项目：[Gwenstadii/cl_binding](https://github.com/Gwenstadii/cl_binding)。

## 服务器位置与资源检查

项目目录为 `$HOME/cl_binding`，独立环境为 `$HOME/cl_binding/.venv`。使用 `ssh scutlab` 连接后，在该项目内操作。

已确认两张 RTX 3090，各 24GB，驱动 570.133.20。旧 Conda 注册表仍有记录，但对应安装目录已不存在，不能直接激活复用。因此在新项目内创建 Python 3.12 环境，不改旧项目或全局 shell 配置。

已检查的资源：

| 资源 | 检查结果 | 处理 |
|---|---|---|
| `$HOME/hf_models/smolvla_libero` | 有约 1.218GB 的策略权重、配置和前后处理器 | 复用作工程参照，不重新下载，不直接当干净旧能力起点 |
| `lerobot/smolvla_base` | 在已核查模型目录与默认缓存中未找到 | 正式 R0-small 仍需取得主基座 |
| 旧 LIBERO 原始演示 | 现有四任务属于此前实验，不含此次 SCENE2 前/左/右/后格 | 保留旧数据，只补本次缺失四文件 |
| `$HOME/.libero/config.yaml` | 指向已不存在的旧安装路径 | 本项目隔离使用 `.server/libero/config.yaml`，不覆盖用户旧配置 |

这里的“未找到”限定于本次检查过的用户目录和缓存，不声称扫描了整台服务器的所有账户。

## 独立环境的安装与验收

```bash
ssh scutlab
cd ~/cl_binding
bash scripts/setup_scutlab.sh
```

若 CUDA 大包单连接很慢，可用服务器已有的 aria2 分段续传；与普通安装二选一执行，不要同时启动两份安装：

```bash
CL_BINDING_MULTIPART=1 bash scripts/setup_scutlab.sh > .server/setup.log 2>&1
```

分段下载只涉及 Python/CUDA 环境依赖，使用官方 wheel 的 SHA-256 校验，不下载模型或演示数据。中断后可以用同一命令续传。另一终端用 `tail -n 25 ~/cl_binding/.server/setup.log` 查看进度。

脚本使用项目内 uv 和 `.venv`，固定 CUDA 12.8 的 PyTorch 2.10.0、torchvision 0.25.0、TorchCodec 0.10.0，再安装当前固定 LeRobot 的 SmolVLA、训练、评估和 LIBERO 依赖。PyTorch/torchvision 对应关系见 [官方版本表](https://pytorch.org/get-started/previous-versions/)，TorchCodec 对应关系见 [官方说明](https://github.com/meta-pytorch/torchcodec)。

网络探测时，服务器直连 PyPI 主站失败，清华 PyPI 镜像和 PyTorch 官方 CUDA 源可达；脚本只对本次安装指定这些来源。若 shell 继承了不可用的代理，可在本终端先执行 `unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy`，不需要修改全局设置。

验收命令：

```bash
cd ~/cl_binding
.venv/bin/python scripts/verify_scutlab.py
```

验收程序禁止 Hub 在线获取模型，仅检查 Python 包、两张卡的 CUDA 运算、SmolVLA 导入、LIBERO 四任务注册及资源位置，并记录视频解码可用性。结果在 `.server/environment_check.json`，实际安装版本在 `.server/requirements.freeze.txt`。这些是机器本地记录，不加入 Git。

**环境检查不等于训练已跑通。** 完整策略加载、数据转换、普通回放和前后配对诊断仍须后续验证；本阶段不启动正式训练。

## 先运行资源清单，不下载

```bash
cd ~/cl_binding
.venv/bin/python scripts/download_stage1.py --plan
```

默认和 `--plan` 都只检查本地。模型检查会核对 safetensors 头与文件末尾偏移，拒绝 LFS 指针和明显截断文件，但不替代与上游哈希逐字节比对。清单在 `.server/resources.json`，会标记可复用的现有路径。

## 下载缺失主基座及小型处理配置

本次网络实测：Hugging Face 官方站直连拒绝连接，现有本地代理访问也超时；`hf-mirror.com` 公开模型 API 的 curl 探测返回 200，但 Python urllib 请求出现过 403，因此还不能视为下载链路已通。需要尝试镜像时，在当前终端先设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

这是第三方镜像，不是 Hugging Face 官方站。下载脚本对这些公开仓库显式使用 `token=False`，不向镜像发送保存的登录令牌；记录取得的 revision，正式实验前核对模型和数据版本。若之后官方站恢复可达，执行 `unset HF_ENDPOINT` 使用官方来源。此次只探测了小型 API，未声称大文件传输已经验证。

确认清单和下载来源后执行：

```bash
cd ~/cl_binding
.venv/bin/python scripts/download_stage1.py --models
```

这条命令：

1. 发现完整本地 `smolvla_base` 或默认 Hub 缓存就直接复用；否则下载到 `$HOME/hf_models/smolvla_base`。
2. 复用已存在的 `smolvla_libero`，不自动再下载一份。
3. 为 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` 只取配置、tokenizer 和 processor 小文件，不重复取得其完整 VLM 权重。
4. 记录实际 Hub revision，使用默认缓存/本地下载元数据续传，不启用 `force_download=True`。

加载**完整 SmolVLA 策略 checkpoint**时，使用 `load_vlm_weights=False` 避免构造阶段又下载同一 VLM 的独立权重；随后必须完整加载策略权重并检查 missing/unexpected keys。不能把这个设置用于“只有 VLM 初始化、尚无完整策略权重”的训练，然后误称已有预训练动作专家。

## 只下载本次缺失的四个原始演示

```bash
cd ~/cl_binding
.venv/bin/python scripts/download_stage1.py --data
```

脚本先检查项目数据目录和旧实验的原始演示目录。缺失时，读取官方 `yifengzhu-hf/LIBERO-datasets` 文件清单，按下列精确文件名只取四个 HDF5，不下载全部套件：

```text
STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy_demo.hdf5
STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy_demo.hdf5
STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy_demo.hdf5
STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo.hdf5
```

新数据保留在 `datasets/raw/` 对应的上游子目录下，实际路径写入资源清单。已存在且具有有效 HDF5 文件头的目标文件直接复用，不覆盖；若上游无法唯一定位某文件，脚本明确停止，不悄悄改成整包下载。

本次保留的上游 LIBERO 下载函数包含强制下载及删除已有目录的分支，因此这里使用单独的保守下载脚本。下载 HDF5 后还需核对演示和转换到当前 LeRobot 数据格式；本仓库尚未把这个转换与全部训练入口连成一键流程。

## 开始后续冒烟前的终端设置

```bash
cd ~/cl_binding
source .venv/bin/activate
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH="$PWD/.server/libero"
```

训练终端设置 `CUDA_VISIBLE_DEVICES=0`，评估终端设置 `CUDA_VISIBLE_DEVICES=1`。先完成单批前反向、保存重载和四条件环境检查，再建立 R0-small；不要直接开始长训练。当前 Git 忽略 `essay/`、权重、数据、`.venv`、工具和 `.server` 运行记录。

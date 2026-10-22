# 科研协作与代码交付

代码修改完成后，运行与改动相关的检查，提交Git commit，推送工作分支，并向`Gwenstadii/cl_binding`的`main`创建PR。已有开放PR时更新同一分支与PR。不要只保存在个人电脑或服务器而不交付。

有原仓库写权限时推送原仓库分支；没有写权限时使用个人fork，再向原仓库提PR。PR应说明行为变化、验证证据和未解决限制。维护者审阅、合并后，同学再pull main；合并前可以直接获取PR的源分支。

本轮源分支为`yangtaozhi/cl_binding`的`codex/stage1-r0-20261021`。在已有checkout且工作树干净时，可按下面方式获取并建立自己的接手分支：

```bash
git status --short --branch
git fetch https://github.com/yangtaozhi/cl_binding.git codex/stage1-r0-20261021
git switch -c review/r0-20261021 FETCH_HEAD
```

若本地已有同名接手分支，先检查该分支再选择新名称，不使用reset或强制覆盖。本来就在服务器实验分支的同学先核对提交，不要为了接手反复切换、pull或重装环境。

纯协议验证命令：

```bash
python -m unittest discover -s tests -p 'test_stage1_protocol.py' -v
```

完整模型训练和仿真只在项目独立环境中进行，并记录代码版本、配置、实际命令、日志、checkpoint和完成标记。测试通过不能替代旧任务能力资格。

模型、数据、环境和HF缓存保留在服务器，不加入Git。关键实验设计、控制变量、续训起点、删除和论文结论由研究者根据证据决定，不因PR合并而自动开始下一阶段训练。

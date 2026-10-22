"""Rebuild the local evidence summary without changing experiment outputs."""
import csv
import json
import statistics as stats
from collections import Counter
from pathlib import Path

root=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
training=[json.loads(line) for line in (root/'train_steps.jsonl').read_text().splitlines()]
finished=read(root/'training_finished_006000.json')
diagnostic=read(root/'control/offline_action_diagnostic.json')
rows=[]
for path in sorted((root/'control').glob('eval_*_screen/evaluation.json')):
    report=read(path)
    episodes=[json.loads(line) for line in (path.parent/'episodes.jsonl').read_text().splitlines()]
    step=int(Path(report['checkpoint']).name)
    categories=Counter(row['terminal_category'] for row in episodes)
    rows.append({'step':step,**{g:report['counts'][g]['correct'] for g in ('front','left','right')},
                 'total_correct':sum(r['correct'] for r in report['counts'].values()),
                 'ever_correct':sum(r['correct_ever'] for r in episodes),
                 'correct_then_destroyed':sum(r['correct_then_destroyed'] for r in episodes),
                 'unfinished':categories['unfinished'],'wrong_goal':categories['wrong_goal'],
                 'loss_last100':stats.mean(t['loss'] for t in training if step-100<t['step']<=step)})
assert len(rows)==12 and len(training)==6000 and training[-1]['step']==6000
with (root/'checkpoint_screening.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
offline=[]
for model in ('base','003000','006000'):
    for split in ('train','validation'):
        group=[r for r in diagnostic['rows'] if r['model']==model and r['split']==split]
        offline.append({'model':model,'split':split,
            'flow_matching_loss_mean':stats.mean(r['flow_matching_loss_mean'] for r in group),
            'xyz_first10_mae_mean':stats.mean(stats.mean(r['mae_first10'][:3]) for r in group)})
summary={'pipeline_status':'complete','r0_qualified':False,'qualification_started':False,
         'training':finished,'training_record_exposures':36000,'training_frames':16634,
         'frame_equivalent_passes':36000/16634,'screening':rows,'offline_diagnostic':offline,
         'conclusion':'Engineering pipeline complete; R0 has not passed its preregistered development/qualification gates.'}
(root/'analysis_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
progress=read(root/'progress.json')
progress.update(status='pipeline_complete_R0_not_qualified',training_step=6000,
                qualification_started=False,r0_qualified=False,
                diagnostic_status=diagnostic['status'],evidence_summary='analysis_summary.json')
(root/'progress.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
table='\n'.join(f"| {r['step']} | {r['front']}/3 | {r['left']}/3 | {r['right']}/3 | {r['total_correct']}/9 | {r['unfinished']} | {r['correct_then_destroyed']} |" for r in rows)
offline_table='\n'.join(f"| {r['model']} | {r['split']} | {r['flow_matching_loss_mean']:.4f} | {r['xyz_first10_mae_mean']:.4f} |" for r in offline)
report=f'''# R0 开发运行验收与诊断

运行编号：r0_20261021_174358。结论：**数据→训练→保存/恢复→多目标闭环评分的工程管线已运行完成；R0 尚未合格。**

本地归档：`{root}`。服务器项目：`/home/lzx/cl_binding`。所有 checkpoint 保留在服务器 `outputs/r0/r0_20261021_174358/checkpoints/`；未下载约19 GB模型/优化器状态到本机。

## 实际完成及可复核证据

- 原始四任务200条轨迹已审计；front/left/right各40条训练、10条验证，共120/30条、16634/4167帧。back未进入R0训练或归一化统计。见 `data_audit/audit.json`。
- seed=20261021，batch=6（各目标2），冻结VLM，BF16，lr=1e-4，warmup=100，chunk=50，n_action_steps=10，flow_steps=10，闭环时限400步。
- 6000步训练正常退出，用时{finished['seconds']/60:.2f}分钟，峰值模型分配显存{finished['peak_allocated_mib']/1024:.2f} GiB。36,000条记录采样，约相当于全训练帧数的2.16遍；随机抽样不保证每帧都被见过。
- 12个checkpoint保存完整；逐个重载固定噪声预测最大差异均为0，冻结参数哈希均未变化，优化器/RNG均保存。真实2→3步恢复与连续训练第3步的采样、loss、梯度一致。见 `control/final_verification.json`、`control/resume_equivalence_check.json`。
- GPU0训练、GPU1逐点只读筛查。最终退出码0，存在PIPELINE_COMPLETE；末次核查无本轮GPU计算进程。没有自动启动FT/ER。
- 训练提交 `460146ba426cc5117660164cbdcc8184ea9e215d`，评分提交 `9224eaddf0fcf7cf8e53d313daf3ee7b76d86c45`；两者差异为评分分类修正/记录版本。服务器工作树干净。分支 `codex/stage1-r0-20261021`，未推送GitHub。

## 闭环结果

每个checkpoint是相同3个开发初态×3条指令，共9次闭环；不同指令的完整初态一致。以下为**400步结束时仍满足正确目标**的次数；最后一列统计曾成功但结束时目标已被破坏的回合。

| step | front | left | right | 合计 | 未完成 | 先成功后破坏 |
|---:|---:|---:|---:|---:|---:|---:|
{table}

所有失败在当前四目标判据下均为未完成；没有记录到终态满足错误格子的回合。不能据此声称模型没有语言混淆，更不能声称已经出现持续学习遗忘。

没有checkpoint达到预定筛查门槛（每目标3/3），因此未启动完整开发10初态/目标或独立资格20初态/目标；独立资格仍保持未使用。3/3筛查较严格，未通过不等于已经证明完整开发成功率低于80%。3000步只是本次9回合筛查分数最高，**没有被确认为续训起点或正式R0**。

## 新增只读诊断

固定每目标24条训练记录及24条验证记录，对基础、3000步、6000步使用相同固定噪声作离线预测；验证轨迹未用于训练，处理器均使用训练统计。FM是训练目标；XYZ MAE是前10个有效动作的三个平移控制分量平均绝对误差，单位为控制器动作尺度，不能解释为米。

| 模型 | 划分 | FM loss均值 | XYZ前10动作MAE |
|---|---|---:|---:|
{offline_table}

模型的离线误差下降，说明并非完全没有学习；6000步训练/验证FM损失有间隔，但这组小样本单噪声诊断不足以认定过拟合。离线误差改善没有对应稳定的闭环改善，原因仍需定位，不能断言只需加训练步数。

HDF5原始图像与恢复场景的图像比较，在4个任务上原方向误差均远小于额外180度旋转误差；训练与部署使用同样的180度变换，没有发现明显方向不一致。示范观察存在上游生成时序差异，未擅自移位动作标签。见 `control/observation_alignment_check.json`。

离线动作按维度检查发现，越界主要在夹爪（第7维），部分平移也有少量越界；因此闭环的“发生过裁剪的步数”不能直接当作控制接口损坏证据。见 `control/offline_action_diagnostic.json`。

## 必须保留的限制

专家动作回放12条中10条至少一次达到正确目标，right/demo_0与right/demo_2未达到；12条原始终态恢复后判据均正确。恢复演示XML并重映射资源路径解决了部分重放不一致，但没有解决所有动作回放问题。谓词验证报告的pass只代表判据/接口检查通过，不代表12条专家动作全部成功。证据见 `validation/predicates_v3/predicate_checks.json`。

## 下一步建议，等待研究者确认后再改设计或启动新训练

优先做一个小范围的**单任务闭环过拟合诊断**：使用已能成功专家回放的front轨迹，固定该轨迹XML和初态，从基础模型开始，短预算训练后在相同状态闭环验证。目的仅是确认“记住示范动作后能否在仿真里执行”，不作为泛化、方法有效性或论文结果。需要先确定轨迹、训练预算及验收门槛；不在本轮自动执行。

如果该诊断通过，再讨论提高R0训练量；如果仍失败，优先排查控制执行/动作时序，而不是把低成功率归为持续学习遗忘。暂不根据这轮低分自行挑3000步或6000步续训，不动独立资格集，不进入FT/ER。

## 实际命令与重跑边界

训练：

```bash
cd /home/lzx/cl_binding
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl
export LIBERO_CONFIG_PATH="$PWD/.server/libero"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/stage1_r0_train.py --run-dir outputs/r0/r0_20261021_174358
```

该训练目录已经存在，**不要原样重跑训练命令**。完整实际训练/12次评分命令见 `control/commands.sh`；诊断命令见 `diagnostic_commands.sh`。离线诊断脚本、选中记录ID及所有指标已归档，可复核而无需重新训练。

这次你应该理解：冒烟通过证明接口能执行；R0合格需要模型稳定完成旧任务；loss下降不能代替闭环成功率；资格集不能反复选点。

下次你可以自己完成：用Xshell执行 `cat /home/lzx/cl_binding/.server/r0_20261021_174358/result_summary.json` 和 `cat /home/lzx/cl_binding/.server/r0_20261021_174358/eval_003000_screen/evaluation.json`，读取完成状态和每目标分母/成功次数；这两个命令只读，不会启动训练。
'''
(root/'R0运行验收与诊断.md').write_text(report,encoding='utf-8')
print(json.dumps({'screens':len(rows),'training_records':len(training),'report':str(root/'R0运行验收与诊断.md')},ensure_ascii=False))

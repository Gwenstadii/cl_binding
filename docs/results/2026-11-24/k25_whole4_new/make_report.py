import json
from pathlib import Path

root=Path.cwd();data=root/'docs/results/2026-11-24/k25_whole4_new'
s=json.loads((data/'summary.json').read_text(encoding='utf-8'))
old=json.loads((root/'docs/results/2026-11-24/k25_whole4/summary.json').read_text(encoding='utf-8'))
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |',*('| '+' | '.join(map(str,r))+' |' for r in rows)])
def pair(c):return f'{c["first"]}／{c["terminal"]}'
def gl(c):return f'{len(c["gained"])}／{len(c["lost"])}'
names={'left':'Left','front':'Front','right':'Right','back':'Back'}
training=json.loads((data/'history/training_COMPLETE.json').read_text())
qual='通过' if s['qualified'] else '未通过'
groups=[('父模型P',s['context']['parent']),('新续训原生F',s['context']['native'])]
groups += [(label+' '+name,s['comparisons'][arm][method]) for arm,label in [('whole150','1–150'),('whole400','1–400')] for method,name in [('parallel','parallel'),('whole','完整残差')]]
body=['# K25-whole4-new：Back重训后的完整残差与投影对照',
'2026-11-24 · 接续 [原whole4](K25-whole4.md) 与 [Back重训启动记录](K25-Back-retrain.md)',
f'Back重训已完成24000步，北京时间2026-11-24 18:36:45结束，耗时{training["wall_seconds"]/60:.1f}分钟。**开发能力验收{qual}**；本轮实际采用第{s["selected_step"]}步模型。新增{s["qualification_episodes"]}个Back开发验收回合、400个正式同条件评估回合，复用原父模型80个参照回合。所有新评估均从复位跑满400步，没有再训练或增加其他修复方法。',
'**主要结果：新训练改善了原生Back，但没有得到达标基线；完整残差也没有救回Back。** 在与原实验相同的20个正式初态上，原生Back曾放好16→19/20、最终1→4/20；独立开发的13档最高终态仍只有8/20。新模型下，完整残差两个窗口的Back曾成功和终态全部为0/20，三旧同时终态均为3/20；parallel分别为12/20、8/20，短窗口四句同时终态为2/20。全程完整残差仍使Right终态13→16，但同时使Left17→9、Front13→5。',
'**本轮不是旧任务首达重新恢复的例子。** 新F原生三旧同时首达已经20/20，而三旧同时终态为0/20；原Back-3000则三旧同时首达0/20。这次训练改变了遗忘的表现，下面重点是完成与保持，不能把终态收益写成找回已经消失的目标选择能力。',
'## 1. 先核对新模型是否真的学好Back',
'开发初态640–659使用独立复位随机种子，与正式800–819物理状态不重合。按事前锁定的“从晚到早”顺序检查保存档，首个曾放好及终态均≥16/20的档才标为通过；只看原生Back开发结果，不按旧任务或修复结果选权重。',
table(['步数','正确首达','曾放好','第150步放好','第400步最终放好'],[[r['step'],r['first'],r['ever'],r['placed150'],r['terminal']] for r in s['qualification']]),
('已达到预定Back开发门槛，以下比较使用该通过验收的模型。开发门槛通过仍不保证正式初态或四任务共同完成。' if s['qualified'] else '**所有检查档均未达到预定Back门槛，因此尚未获得“Back能力达标”的新基线。** 按本次用户要求继续复跑，使用事前指定的24000步终点作诊断；下面不作为“强Back模型上已验证四任务修复”的结果。不能把训练结束、loss变小或曾放好很多直接写成能力达标。'),
'## 2. 正式同初态逐目标结果',
'每格为**唯一正确首达／第400步最终放置，各 /20**。三旧成组要求同一初态Left、Front、Right全部正确；四句终态要求四句全部最终放好。1–150组在第151步交回原生F，1–400组全程干预。',
table(['执行方式','Left','Front','Right','Back','三旧首达／终态','四句终态'],[[name,*[pair(c['per_goal'][g]) for g in names],pair(c['old_joint']),c['joint']['terminal']] for name,c in groups]),
'![新模型：首达与最终放置](figures/2026-11-24/k25_whole4_new/matched_components.png)',
'## 3. Back的阶段成功与保持',
'曾放好是放置判据任一步为真；最终放好只看第400步。首达按抬升>3cm且连续5步进入唯一区域计分，与放置判据不同。“先Front后Back”使用区域进入顺序，故数量不必等于曾放好。',
table(['执行方式','正确首达','≤150曾放好','第150步放好','≤400曾放好','第400步放好','先Front后Back'],[[name,*[c['per_goal']['back'][k] for k in ['first','ever150','placed150','ever','terminal','Front_then_Back']]] for name,c in groups]),
'![每个初态的Back放置判据随时间变化](figures/2026-11-24/k25_whole4_new/back_placement_time.png)',
'新原生F有17/20在前150步曾放好、9/20在150步仍放好，但最终只剩4/20；这4例在最后50步的放置判据均一直为真。相比原3000步的16/20曾放好、1/20终态，已有改善，但达标所需的稳定完成仍不足。',
'parallel短窗口的10例Back曾成功，只有初态814在干预期间于第141步首次出现，另9例都在151步交回原生F后首次出现。最终成功是802、814、816。全程parallel只有814曾放好，最终也丢失。不能把短窗口的10例曾成功全部归为投影控制器自身执行出了Back，也不能全部归为交回之后；两段必须分开。',
'完整残差在两个窗口中均从未满足Back放置判据，且Back抓取均为20/20。因此本轮没有复现原whole4中“补回响应让早期Back曾成功0→3”的收益；它对新模型不构成稳定的Back恢复证据。其Back首达仍为0，经过Front与最终放置仍按不同判据报告。',
'各方法每个初态最后50步的放置有效比例保留在summary，防止用一帧成功代表持续稳定。ever≤150是原whole4追加的阶段审计，本轮保持同一定义；未用它调增益或挑模型。',
'## 4. 补回整块被移除响应的配对得失',
'下表以同窗口parallel为参照，完整残差新增／损失的初态数。每项按20个物理初态配对，去噪输入与控制块不算独立样本。',
table(['窗口／目标','首达新增／损失','终态新增／损失','曾放好新增／损失'],[[label+'/'+names[g],*[gl(s['comparisons'][arm]['paired']['per_goal'][g][m]) for m in ['first','terminal','ever']]] for arm,label in [('whole150','1–150'),('whole400','1–400')] for g in names]),
table(['窗口','三旧首达新增／损失','三旧终态新增／损失'],[[label,gl(s['comparisons'][arm]['paired_old']['joint']['first']),gl(s['comparisons'][arm]['paired_old']['joint']['terminal'])] for arm,label in [('whole150','1–150'),('whole400','1–400')]]),
'![相同初态上的新增成功与损失](figures/2026-11-24/k25_whole4_new/paired_changes.png)',
'补回响应后，Left终态在短窗口新增0、损失7，在全程新增0、损失8；Front两个窗口终态均新增0、损失7或8。Front首达失败都属于none，不能强行归为送进另一个格子。Front抓取为parallel20/20、完整残差19/20；其中初态803有抓取差异，因此也不能把Front全部差距仅归于目标方向。',
'Right短窗口终态两组虽然都是15/20，但有3例新增和3例损失，不能写成逐回合等价；全程则新增5、损失2，净增3，13→16。三旧成组终态短窗口新增1、损失10，12→3；全程新增1、损失6，8→3。被移除部分对不同目标的净作用仍不同，而非一组对所有任务都无用的噪声。',
'## 5. 与原Back-3000同条件结果对照',
'下表每格为最终放置数 /20，初态与评分完全相同。新旧训练配置同时改变预算、种子、采样和学习率日程，且为各自模型独立标定增益，因此跨模型表是描述性复验，不能单独归因于某项训练改动。单个新模型内部whole与parallel仍只有是否保留整块被移除响应这一处理差别。',
table(['模型／窗口','方法','Left终态','Front终态','Right终态','Back曾放好／终态','三旧终态'],[[version+'/'+label,name,*[c['per_goal'][g]['terminal'] for g in ['left','front','right']],f'{c["per_goal"]["back"]["ever"]}／{c["per_goal"]["back"]["terminal"]}',c['old_joint']['terminal']] for version,summary in [('原3000',old),('新'+str(s['selected_step']),s)] for arm,label in [('whole150','1–150'),('whole400','1–400')] for method,name in [('parallel','parallel'),('whole','完整残差')] for c in [summary['comparisons'][arm][method]]]),
table(['原生模型','Back首达','Back曾放好','第150步Back','Back终态'],[[name,*[summary['context']['native']['per_goal']['back'][k] for k in ['first','ever','placed150','terminal']]] for name,summary in [('原Back-3000',old),('本轮新模型',s)]]),
table(['原生模型','Left首达／终态','Front首达／终态','Right首达／终态','三旧首达／终态'],[[name,*[pair(summary['context']['native']['per_goal'][g]) for g in ['left','front','right']],pair(summary['context']['native']['old_joint'])] for name,summary in [('原Back-3000',old),('本轮新模型',s)]]),
'## 6. 方法、增益及可解释范围',
f'仍执行父μ＋αrF(q)与父μ＋αB32·pinv(B7)·rF(q)7。三旧参考固定Left／Front／Right；真实完整指令只进入F，P只查询固定旧参考。新模型两种控制器共用 **α={s["alpha"]:.12f}**；原模型α={old["alpha"]:.12f}。为新F重新按原规则标定：开发初态0–9、原生Back前160步、30至150步首个Euler输入共130项父／新三旧中心化响应范数比，取上中位数封顶10。不按正式成绩挑剂量，不在投影后补做等范数放大。',
'原parallel4的episode源码保持原样；完整残差仅替换field回调。两组均float64构造完整50×32响应后转原生dtype。投影在50×7上拟合系数，再提升到32维；完整残差保留补齐维度，因此不能将差别只归于执行7维的某个方向或称32维欧氏正交投影。公式及控制解释沿用原whole4报告。',
'原episode保留了历史profile字符串lfr_back3000；它不参与选权重。本轮实际模型由protocol.models、selected_step及每条记录的checkpoint_step绑定，文件SHA在起止身份记录中核验，不能将历史标签误读成实际仍执行3000步旧模型。',
table(['真实指令','完整残差强度','投影强度','被移除强度','偏轴范数比例'],[[names[g],*[f'{d[k]:.3f}' for k in ['scaled_full_over_parent','scaled_projection_over_parent','scaled_removed_over_parent','off_axis_fraction']]] for g,d in s['fixed_first_input'].items()]),
'上表为每句20个首次相同输入的中位数，强度以父三旧中心化响应范数归一化。偏轴比例是范数比例，不是语义信息百分比；数值只能辅助解释，是否有功能贡献仍以实际执行的配对结果为准。',
'这仍是依赖父模型的双模型控制诊断。单个模型内部的对照能识别整块被移除响应的净行为作用，不能逐维判定语义、有害性或必要性；闭环轨迹分岔后观测也随之变化。尤其在原生Back仍不稳时，不能将新任务失败唯一归咎于投影。',
'本轮增益小于1，是按同一标定原则**缩小**残差，不是套用历史2.959统一放大。不能把所有动作遗忘概括为d变小；同一规范句的首达保持、最终放置失败以及残差的方向／幅度匹配是不同问题。本轮没有µ-only或µF／µP切换对照，因此也没有证明共同响应偏移是唯一病因。',
'本轮对“有害残差还是投影删掉Back信息”的回答是：保留被移除响应对Left／Front和三旧完成整体更差，对全程Right仍有收益；**没有看到新模型Back的补回收益。** 这不支持“只要保留完整残差就解决Back”的解释，也不足以证明投影完全没有丢失有用Back分量，因为可能存在相互抵消且原生基线仍弱。强Back基线这一混杂因素尚未真正排除，四任务稳定共存仍未成立。',
'## 7. 核验与复现',
f'400个新正式回合和{s["qualification_episodes"]}个开发回合均由原轨迹重算评分，核验动作哈希、复位签名和41条状态签名记录数。新模型两种方法各80对窗口前150步前缀逐位相同（合计{s["bitwise_duration_prefix_pairs"]}对）；两方法在相同首次输入的投影诊断{s["identical_initial_diagnostics_pairs"]}对完全相同。80次完整32维公式／回调门禁和80次原投影门禁通过，训练参数前后指纹一致。模型、归一化、源码及28个共享依赖起止身份一致。',
'完整24000步训练记录逐步核验了采样索引、学习率、有限损失和梯度；全部保存档权重身份核验通过。3项既有完整残差公式检查与21项stage1协议检查通过；离线重算可复现summary。无新训练、措辞、反向或其他恢复算法扩展。',
'[全部计数与配对初态](results/2026-11-24/k25_whole4_new/summary.json) · [开发选档](results/2026-11-24/k25_whole4_new/selection.json) · [锁定协议](results/2026-11-24/k25_whole4_new/protocol.json) · [原始证据与离线入口](results/2026-11-24/k25_whole4_new/README.md)。',
]
(root/'docs/K25-whole4-new.md').write_text('\n\n'.join(body)+'\n',encoding='utf-8',newline='\n')
print('NEW_REPORT_WRITTEN')

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'Microsoft YaHei', 'svg.fonttype': 'path',
                     'pdf.fonttype': 42, 'axes.unicode_minus': False})
C = {'ink': '#192E40', 'muted': '#586A78', 'old': '#147D80', 'new': '#C56B37',
     'line': '#BBC9D0', 'pale': '#F3F7F8', 'oldbg': '#EAF5F3', 'newbg': '#FCF0E7'}
fig = plt.figure(figsize=(18, 11.2), facecolor='white')
ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0, 1800), ylim=(0, 1120)); ax.axis('off')

def txt(x, y, s, size=12, color=None, ha='left', weight='normal', va='center'):
    return ax.text(x, y, s, fontsize=size, color=color or C['ink'], ha=ha, va=va,
                   fontweight=weight, linespacing=1.55)

def box(x, y, w, h, face='white', edge=None):
    p = FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=10',
                      facecolor=face,edgecolor=edge or C['line'],linewidth=1.15)
    ax.add_patch(p)

def arrow(points, color=None, dashed=False):
    color = color or C['muted']
    for p,q in zip(points[:-2], points[1:-1]):
        ax.plot([p[0],q[0]],[p[1],q[1]],color=color,lw=1.3,
                linestyle='--' if dashed else '-')
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',
                                mutation_scale=13,lw=1.3,color=color,
                                linestyle='--' if dashed else '-'))

txt(60,1070,'共同响应 μ × 条件差异 d：交叉恢复与闭环效果',25,weight='bold')
txt(60,1025,'输出级诊断干预  |  R0：已学左格／前格   →   FT4：继续学右格',13,color=C['muted'])
ax.plot([60,1740],[995,995],color=C['line'],lw=1)
txt(60,966,'a   同输入分解、交叉组合、闭环执行',16,weight='bold')
txt(1110,966,'b   实测恢复效果：四种组合',16,weight='bold')
txt(60,934,'每次查询固定：观测 o、机器人状态 s、带噪动作 xτ、生成时刻 τ',11,color=C['muted'])

box(65,742,195,145,C['pale'])
txt(162,850,'当前输入',14,ha='center',weight='bold')
txt(162,798,'同一 o、s、xτ、τ\n分别询问 Left / Front',11,ha='center')
box(320,829,190,72,C['oldbg'],C['old'])
txt(415,866,'旧模型 R0',15,C['old'],ha='center',weight='bold')
box(320,704,190,72,C['newbg'],C['new'])
txt(415,741,'续训模型 FT4',15,C['new'],ha='center',weight='bold')
arrow([(260,815),(285,815),(285,865),(320,865)],C['old'])
arrow([(285,815),(285,740),(320,740)],C['new'])

for y, name, color, bg in [(815,'旧',C['old'],C['oldbg']), (690,'新',C['new'],C['newbg'])]:
    box(565,y,425,100,bg,color)
    txt(586,y+74,f'两条响应  v{name},L  与  v{name},F',12,color,weight='bold')
    txt(586,y+43,f'μ{name} = (v{name},L + v{name},F) / 2',13)
    txt(586,y+16,f'd{name} = (v{name},L − v{name},F) / 2',13)
arrow([(510,865),(565,865)],C['old'])
arrow([(510,740),(565,740)],C['new'])
txt(320,655,'μ：两条指令的共同响应     d：两条指令的半差',11,color=C['muted'])

box(320,492,670,132,'#F5F5FB','#9194BC')
txt(655,596,'交叉选择 μ 与 d 的来源',15,ha='center',weight='bold')
txt(655,556,r'$\hat{v}_{L}=\mu_i+d_j \qquad \hat{v}_{F}=\mu_i-d_j$',20,ha='center')
txt(655,517,'i, j ∈ {新, 旧}；四组各自独立执行',11,ha='center',color=C['muted'])
arrow([(990,865),(1020,865),(1020,591),(990,591)],C['old'])
arrow([(777,690),(777,624)],C['new'])
txt(1018,670,'',10)

box(320,330,670,102,C['pale'])
txt(655,403,'flow 积分 → 动作块 → 反归一化 → 环境执行',14,ha='center',weight='bold')
txt(655,363,'重新观察；每次重规划、每个生成步均重算当前响应',11,ha='center')
arrow([(655,492),(655,432)])
arrow([(320,380),(162,380),(162,742)],dashed=True)
txt(171,442,'闭环反馈',11,color=C['muted'])
txt(320,292,'v 是动作生成流场，不是机械臂世界坐标速度。',11,color=C['muted'])
txt(320,264,'这里不换网络参数，也不平均两条已执行轨迹。',11,color=C['muted'])

ax.plot([1065,1065],[255,940],color='#DAE2E7',lw=1)

def heatmap(y, title, values, subtitle):
    x0=1225; w=225; h=70
    txt(1110,y+198,title,14,weight='bold')
    txt(x0+w*.5,y+155,'d新',13,C['new'],ha='center')
    txt(x0+w*1.5,y+155,'d旧',13,C['old'],ha='center')
    for r, row in enumerate(values):
        yy=y+(1-r)*h
        txt(1205,yy+h/2,'μ新' if r==0 else 'μ旧',13,
            C['new'] if r==0 else C['old'],ha='right')
        for col,value in enumerate(row):
            color={0:'#F0F3F5',2:'#D3E8E5',5:'#72BAB1',8:'#147D80'}[value]
            ax.add_patch(Rectangle((x0+col*w,yy),w,h,facecolor=color,
                                  edgecolor='white',linewidth=2))
            txt(x0+(col+.5)*w,yy+h/2,f'{value}/8',22,
                'white' if value>=5 else C['ink'],ha='center',weight='bold')
    txt(1110,y-26,subtitle,10.5,color=C['muted'])

heatmap(710,'请求目标曾成功（原生成功谓词）',[[0,0],[5,8]],
        '只换 μ：左指令 4/4 曾成功，前指令仅 1/4。')
heatmap(425,'400 步终态仍正确',[[0,0],[2,8]],
        '曾完成后仍可能被破坏；两类指标不能混算。')

box(1110,255,565,100,C['pale'])
txt(1130,326,'只换旧 μ，仍有目标偏置',13,weight='bold')
txt(1130,291,'首次满足原生目标：左格 7 次、前格 1 次。',11)

for x,w,n,title,body in [
    (60,530,'01','单补条件差异不足','新 μ ＋ 旧 d：曾成功、终态均为 0/8。'),
    (620,530,'02','共同背景与差异需要配合','旧 μ ＋ 新 d：部分救援，但仍偏向旧左格。'),
    (1180,560,'03','完整旧响应是参照','旧 μ ＋ 旧 d = R0 当前响应；不等于学生自愈。')]:
    ax.plot([x,x+w],[218,218],color=C['old'],lw=2)
    txt(x,191,n,12,C['old'],weight='bold')
    txt(x+38,191,title,13,weight='bold')
    txt(x,153,body,10.5,color=C['muted'])

txt(60,100,'证据范围：4 个共享开发初态 × 2 条旧指令＝每组 8 回合；全程干预，400 个控制步。',11,color=C['muted'])
txt(60,70,'这是旧模型参与的推理诊断，不是训练后的独立恢复；未检验新右格保持，也不定位内部神经模块。',11,color=C['muted'])
txt(60,37,'来源：2026-10-28 共同／区别响应交换，§8.2。仅展示直接输出级 μ／d 实验，不纳入后续内部补丁。',9.5,color=C['muted'])

for extension in ('png','svg','pdf'):
    fig.savefig(OUT/f'mu_d_cross_recovery_v1.{extension}',dpi=220,facecolor='white')
plt.close(fig)
print(OUT/'mu_d_cross_recovery_v1.png')

"""Build scientific contact sheets from verified replay frames and trace statistics."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


LABELS={'front':'前格','left':'左格','right':'右格','back':'后格'}


def contact_sheet(folder, output, title, subtitle, rows, columns, font_path):
    title_font=ImageFont.truetype(str(font_path),36)
    body_font=ImageFont.truetype(str(font_path),23)
    small_font=ImageFont.truetype(str(font_path),20)
    frame_w,frame_h=384,192
    left,margin,gap=235,28,16
    row_h=frame_h+104
    width=left+columns*(frame_w+gap)+margin
    height=148+len(rows)*row_h+50
    canvas=Image.new('RGB',(width,height),'#f5f7fa')
    draw=ImageDraw.Draw(canvas)
    draw.text((margin,22),title,font=title_font,fill='#142638')
    draw.text((margin,77),subtitle,font=small_font,fill='#465768')
    for row_index,(case,label,steps) in enumerate(rows):
        top=135+row_index*row_h
        draw.multiline_text((margin,top+50),label,font=body_font,fill='#20394e',spacing=10)
        timeline=json.loads((folder/case/'timeline.json').read_text(encoding='utf-8'))
        for column,step in enumerate(steps):
            x=left+column*(frame_w+gap)
            image=Image.open(folder/case/f'frame_{step:03d}.png').convert('RGB')
            image=image.crop((0,40,256,168)).resize((frame_w,frame_h),Image.Resampling.NEAREST)
            canvas.paste(image,(x,top+35))
            active=[LABELS[g] for g,v in timeline[step-1]['predicates'].items() if v]
            text='条件满足：'+('、'.join(active) if active else '无')
            draw.text((x,top),f'第{step}步 · {step/20:.2f}秒',font=body_font,fill='#20394e')
            draw.text((x,top+frame_h+47),text,font=body_font,
                      fill='#147653' if active else '#6b7280')
    draw.text((margin,height-38),'双视角均来自原128×128相机；原动作重放逐步核验通过。格名以任务定义和谓词为准。',
              font=small_font,fill='#465768')
    canvas.save(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,default=Path('docs/results/2026-10-23/visual_cases'))
    parser.add_argument('--font',type=Path,default=Path('C:/Windows/Fonts/msyh.ttc'))
    args=parser.parse_args(); folder=args.folder
    manifest=json.loads((folder/'media_manifest.json').read_text(encoding='utf-8'))
    for entry in manifest['files']:
        assert hashlib.sha256((folder/entry['relative_path']).read_bytes()).hexdigest()==entry['sha256']
    cases=('before_right31','after_right31','after_front31','back30','back36')
    for case in cases:
        verification=json.loads((folder/case/'replay_verification.json').read_text(encoding='utf-8'))
        assert verification['status']=='pass' and verification['predicate_mismatches']==0
        assert max(verification['maximum_position_errors'].values())==0
    contact_sheet(folder,folder/'goal_substitution.png',
        '同一右格指令：原来放对右格，学前格后改放前格',
        '共同初态31；同一场景、初态、种子；对照指令为前格。下图不证明曾具有多目标语言选择能力。',
        [('before_right31','学前格前\n请求：右格\n右格9000步',[1,125,400]),
         ('after_right31','学前格后\n请求：右格\n前格12000步',[1,125,400]),
         ('after_front31','同一更新模型\n请求：前格\n前格12000步',[1,125,400])],3,args.font)
    contact_sheet(folder,folder/'success_lost.png',
        '曾成功 ≠ 400步后仍正确：两种不同的结果丢失',
        '后格12000步模型，均请求后格。初态30最终未完成；初态36后来进入左格，不能都叫同一种目标替代。',
        [('back30','初态30\n340—380步成功\n381步起失效',[320,340,360,400]),
         ('back36','初态36\n121—130步成功\n293步起到左格',[100,121,293,400])],4,args.font)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(str(args.font))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(args.font)).get_name(),
                         'axes.unicode_minus':False,'font.size':11})
    temporal=json.loads((folder/'back20_temporal_diagnostics.json').read_text(encoding='utf-8'))
    episodes=temporal['episodes']
    ever=sum(r['correct_ever'] for r in episodes);terminal=sum(r['terminal_correct'] for r in episodes)
    assert (ever,terminal)==(17,3)
    lost=[r for r in episodes if r['correct_ever'] and not r['terminal_correct']]
    assert len(lost)==14
    assert sum(not r['final_goals'] for r in lost)==12
    assert sum(bool(r['final_goals']) for r in lost)==2
    fig,ax=plt.subplots(figsize=(13,8.5))
    for i,row in enumerate(episodes):
        ax.hlines(i,0,400,color='#e4e9ef',lw=5,zorder=0)
        for start,end in row['success_intervals']:
            ax.broken_barh([(start-0.5,end-start+1)],(i-0.23,0.46),facecolors='#24966b')
        ending='、'.join(LABELS[g] for g in row['final_goals']) or '未完成'
        color='#147653' if row['terminal_correct'] else '#b44338'
        ax.text(410,i,ending,va='center',color=color,fontsize=10)
    ax.set_yticks(range(len(episodes)),[str(r['init_index']) for r in episodes])
    ax.set_ylim(len(episodes)-0.4,-1);ax.set_xlim(0,468)
    ax.set_xticks(range(0,401,50));ax.set_xlabel('控制步（20Hz；400步约20秒）')
    ax.set_ylabel('初态编号');ax.set_title('后格20回合：绿色为成功条件成立区间；右侧是400步终态',loc='left',pad=22)
    ax.text(0,-0.6,'曾成功17/20；最终正确3/20；成功后失效14＝12未完成＋2其他格',fontsize=11,color='#26384b')
    for side in ('top','right','left'):ax.spines[side].set_visible(False)
    ax.grid(axis='x',alpha=.18);fig.tight_layout()
    fig.savefig(folder/'back20_success_intervals.png',dpi=160);plt.close(fig)

    positive=sum(r['gripper_command_min_after_first_success']>0 for r in episodes if r['correct_ever'])
    summary={'source_media_hashes_verified':len(manifest['files']),'replays_verified':5,
             'all_replay_position_errors_zero':True,'new_policy_rollouts':0,'training_updates':0,
             'back20':{'ever':ever,'terminal':terminal,'lost_after_success':len(lost),
                 'lost_to_unfinished':12,'lost_to_other_goal':2,
                 'all_positive_gripper_commands_after_first_success':positive,
                 'positive_gripper_sign_is_a_control_clue_not_a_contact_or_release_measurement':True}}
    (folder/'visual_verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    main()

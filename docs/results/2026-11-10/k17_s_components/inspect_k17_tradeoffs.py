"""Descriptive K17 follow-up: height tradeoffs and pre-execution coordinate coupling."""
import argparse
import json
from pathlib import Path
from statistics import median

def stats(values):
    values=list(values)
    return {'n':len(values),'min':min(values),'median':median(values),'max':max(values)} if values else {'n':0}

def main(archive):
    summary=json.loads((archive/'summary.json').read_text(encoding='utf-8'))
    records={};traces={}
    for base,store in (('',records),('_traces',traces)):
        names=['reference_episodes.jsonl' if not base else 'reference_traces.jsonl',*[f'i{i}{base}.jsonl' for i in range(20,40)]]
        for name in names:
            for r in map(json.loads,(archive/name).read_text().splitlines()):store[r['init'],r['condition'],r['goal']]=r['rows'] if base else r
    assert len(records)==len(traces)==640
    def describe(init,condition,goal):
        r=records[init,condition,goal];rows=traces[init,condition,goal];target=r['requested'];boundary=r['boundary']
        successes=[v['step'] for v in rows if v['pred'][target]]
        exits=[b['step'] for a,b in zip(rows,rows[1:]) if a['pred'][target] and not b['pred'][target]]
        return {'condition':condition,'first':r['first'],'first_step':r['first_step'],'boundary':boundary,'intervention_end':boundary+60,'first_success':min(successes,default=None),'last_success':max(successes,default=None),'success_exit_steps':exits,'success_at_window_end':rows[boundary+59]['pred'][target],'terminal_flags':r['terminal'],'window_end_book_xyz':rows[boundary+59]['obj'],'final_book_xyz':rows[-1]['obj'],'window_end_both_contacts':bool(rows[boundary+59]['c1'] and rows[boundary+59]['c2']),'final_both_contacts':bool(rows[-1]['c1'] and rows[-1]['c2']),'window_all_positive_gripper':all(v['a'][6]>0 for v in rows[boundary:boundary+60])}
    def compact(edge):
        e=summary['factor_switch_edges'][edge]
        return {'after':e['after'],'before':e['before'],'metrics':{m:{k:e['metrics'][m][k] for k in ('point','ci95')} for m in ('L_first','R_first','L_ever','R_ever','L_term','R_term','term')},'transitions':e['transitions']}
    key_edges=('R:H_to_HR','Z:HR_to_HZR','G:HZR_to_HZRG','R:N_to_R')
    report={'status':'descriptive follow-up, not causal mediation or training-update identification','key_comparisons':{k:compact(k) for k in key_edges},'H_rotation_interaction':{m:{k:summary['conditional_pair_interactions']['H_x_R_given_N'][m][k] for k in ('point','ci95')} for m in ('L_first','R_first','L_term','R_term')},'height_left_losses':[],'height_right_terminal_changes':[],'first_chunk_coupling':{}}
    edge=summary['factor_switch_edges']['Z:HR_to_HZR']['transitions']
    for init in edge['A']['terminal']['lost']:
        report['height_left_losses'].append({'init':init,'before':describe(init,'HR','A'),'after':describe(init,'HZR','A')})
    for direction in ('gained','lost'):
        for init in edge['B']['terminal'][direction]:report['height_right_terminal_changes'].append({'init':init,'change':direction,'before':describe(init,'HR','B'),'after':describe(init,'HZR','B')})
    for label in ('R:H_to_HR','Z:HR_to_HZR','G:HZR_to_HZRG','R:N_to_R'):
        data=summary['diagnostics']['same_initial_input_first_chunk_coupling'][label]
        parts=data['per_episode']
        report['first_chunk_coupling'][label]={'outside_factor_changed_cases':data['outside_coordinates_changed_cases'],'n_cases':40,'x_command_rms_delta':stats(v['command_rms_difference_by_coordinate'][0] for v in parts),'y_command_rms_delta':stats(v['command_rms_difference_by_coordinate'][1] for v in parts),'gripper_first_chunk_sign_changed_cases':sum(v['gripper_sign_changes']>0 for v in parts),'gripper_window_sign_changed_cases':0}
        before,after=data['before'],data['after']
        for init in range(20,40):
            for goal in 'AB':
                start=records[init,before,goal]['boundary']
                a=traces[init,before,goal][start:start+60];b=traces[init,after,goal][start:start+60]
                sign=lambda x:1 if x>0 else -1 if x<0 else 0
                if any(sign(x['a'][6])!=sign(y['a'][6]) for x,y in zip(a,b)):report['first_chunk_coupling'][label]['gripper_window_sign_changed_cases']+=1
    report['left_height_loss_cases_all_first_left_and_ever_success']=all(v['before']['first']==v['after']['first']==['left'] and v['before']['first_success'] is not None and v['after']['first_success'] is not None for v in report['height_left_losses'])
    report['left_height_loss_cases_last_exit_after_window_end']=sum(v['after']['success_exit_steps'][-1]>v['after']['intervention_end'] for v in report['height_left_losses'])
    (archive/'mechanism_descriptors.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('H_rotation_interaction','first_chunk_coupling','left_height_loss_cases_all_first_left_and_ever_success','left_height_loss_cases_last_exit_after_window_end')},ensure_ascii=False))
    print('left_height_losses',[(v['init'],v['after']['first_success'],v['after']['success_exit_steps'],v['after']['terminal_flags']) for v in report['height_left_losses']])
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);main(p.parse_args().archive.resolve())

"""Supplemental historical-protocol learning curve and matched archived controls.

Only evaluates existing checkpoints. It never changes the training process.
All four checkpoint steps and all state indices are fixed before historical scores.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

RUN = Path(__file__).resolve().parent
ROOT = Path('/home/lzx/cl_binding_stage2a')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def execute(command, logname):
    with (RUN / logname).open('x') as log:
        subprocess.run(command, cwd=ROOT, env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4'),
                       stdout=log,stderr=subprocess.STDOUT,check=True)


def main():
    assert not (RUN / 'historical_matrix.json').exists()
    config = read(RUN / 'config.json')
    config['evaluation']['qualification_indices'] = list(range(20,40))
    config['evaluation']['qualification_required_correct'] = 16
    write(RUN / 'historical_confirmation_config.json', config)
    matrix = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),new={},previous_B0={},confirmations={},
                  note='Historical front-state protocol. No training changes or hidden replacement of random-reset scores.')
    qualified = None

    def update():
        write(RUN / 'historical_matrix.json', matrix)

    def confirm(step):
        nonlocal qualified
        if qualified is not None:
            return
        dev = matrix['new'][str(step)]
        if dev['counts']['back']['correct'] < 16:
            return
        output = RUN / f'historical_confirm_{step:06d}'
        execute([sys.executable,'-u',str(RUN/'historical_stage1_r0_eval.py'),'evaluate',
                 '--config',str(RUN/'historical_confirmation_config.json'),
                 '--checkpoint',str(RUN/'training/checkpoints'/f'{step:06d}'),
                 '--output',str(output),'--partition','qualification','--instruction-goals','back','--trace'],
                 f'historical_confirm_{step:06d}.log')
        result = read(output/'evaluation.json')
        matrix['confirmations'][str(step)] = result
        if result['counts']['back']['correct'] >= 16:
            qualified = dict(code=f'B0-Back-Hist-{step:06d}-HistoricalConfirmed',step=step,
                basis='historical native400: front init0-19 and20-39 each >=16 terminal Back',
                development=dev,confirmation=result,
                random_reset_gate_separate=True,training_unchanged=True)
            write(RUN/'QUALIFIED_HISTORICAL.json',qualified)
        update()

    for step in (6000,12000,18000):
        while not (RUN/f'eval_{step}_COMPLETE.json').exists():
            if (RUN/'COMPLETE.json').exists():
                break
            if (RUN/'FAILED_controller.json').exists():
                raise RuntimeError('Main controller failed')
            time.sleep(5)
        if not (RUN/f'eval_{step}_COMPLETE.json').exists():
            break
        primary = read(RUN/f'eval_{step}_COMPLETE.json')
        if primary['qualified']:
            break  # Main controller evaluates this selected checkpoint historically.
        output = RUN/f'historical_dev_{step:06d}'
        execute([sys.executable,'-u',str(RUN/'historical_stage1_r0_eval.py'),'evaluate',
                 '--config',str(RUN/'config.json'),'--checkpoint',str(RUN/'training/checkpoints'/f'{step:06d}'),
                 '--output',str(output),'--partition','development','--instruction-goals','back','--trace'],
                 f'historical_dev_{step:06d}.log')
        matrix['new'][str(step)] = read(output/'evaluation.json')
        update()
        confirm(step)
        # Both preselected archived controls; never pick the weaker one after seeing results.
        if step == 12000:
            for prior_step in (12000,24000):
                execute([sys.executable,'-u',str(RUN/'back_first_historical_baseline.py'),str(RUN),
                         '--step',str(prior_step)],f'previous_B0_{prior_step}_historical.log')
                matrix['previous_B0'][str(prior_step)] = read(RUN/f'previous_B0_{prior_step}_historical_dev20/evaluation.json')
                update()
    while not (RUN/'COMPLETE.json').exists():
        if (RUN/'FAILED_controller.json').exists():
            raise RuntimeError('Main controller failed')
        time.sleep(5)
    complete = read(RUN/'COMPLETE.json')
    step = complete['selected_step']
    matrix['new'][str(step)] = complete['historical_evaluation']
    update()
    confirm(step)
    # If primary qualification ended training early, the historical controls still stay matched.
    for prior_step in (12000,24000):
        if str(prior_step) not in matrix['previous_B0']:
            execute([sys.executable,'-u',str(RUN/'back_first_historical_baseline.py'),str(RUN),
                     '--step',str(prior_step)],f'previous_B0_{prior_step}_historical.log')
            matrix['previous_B0'][str(prior_step)] = read(RUN/f'previous_B0_{prior_step}_historical_dev20/evaluation.json')
            update()
    matrix['qualified_historical'] = qualified
    matrix['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    write(RUN/'HISTORICAL_MATRIX_COMPLETE.json',matrix)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        import traceback
        write(RUN/'FAILED_historical_matrix.json',dict(error=str(error),traceback=traceback.format_exc()))
        raise

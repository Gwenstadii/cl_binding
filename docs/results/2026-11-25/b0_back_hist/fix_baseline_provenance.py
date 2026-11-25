"""Correct descriptive processor provenance; never alter scores or episode data."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
run=Path('/home/lzx/cl_binding/.server/k25_b0_back_hist_20261125_01')
assert (run/'HISTORICAL_MATRIX_COMPLETE.json').exists() or (run/'STOPPED_BY_USER.json').exists()
source='/home/lzx/cl_binding_stage2a/datasets/stage1_dual_left_right_train_aligned_v1'
container='/home/lzx/cl_binding/.server/r1_left_right_20261104_01/train_r1/checkpoints/024000'
records=[];reports={}
for step in (12000,24000):
    p=run/f'previous_B0_{step}_historical_dev20/evaluation.json'
    report=json.loads(p.read_text());before=json.loads(p.read_text())
    assert report['reconstruction']['overwritten_task_tensors']==155
    assert report['reconstruction']['processors']=='R1 statistics, exactly as used by previous B0'
    episodes=p.parent/'episodes.jsonl';score_sha=hashlib.sha256(episodes.read_bytes()).hexdigest()
    report.update(normalization_source=source,runtime_processor_checkpoint=container,
                  evaluation_config_used_for_reset_and_partition_only=True)
    if report!=before:
        backup=p.parent/'evaluation.original_metadata.json';assert not backup.exists()
        backup.write_text(json.dumps(before,indent=2)+'\n')
        p.write_text(json.dumps(report,indent=2)+'\n')
        records.append(dict(step=step,old_descriptive_source=before['normalization_source'],
                            actual_processor_source=source,episode_sha256_unchanged=score_sha))
    assert report['counts']==before['counts'] and hashlib.sha256(episodes.read_bytes()).hexdigest()==score_sha
    reports[str(step)]=report
for name in ('historical_matrix.json','HISTORICAL_MATRIX_COMPLETE.json'):
    p=run/name
    if not p.exists():continue
    m=json.loads(p.read_text());m['previous_B0']=reports
    p.write_text(json.dumps(m,indent=2)+'\n')
(run/'BASELINE_PROVENANCE_CORRECTION.json').write_text(json.dumps(dict(at_utc=datetime.now(timezone.utc).isoformat(),
    reason='Original generic evaluator describes config training-data source. Archived B0 actually loads R1 pre/post-processors; correct that descriptive field, preserving all scores and episode bytes.',
    records=records),indent=2)+'\n')
print(json.dumps(records))

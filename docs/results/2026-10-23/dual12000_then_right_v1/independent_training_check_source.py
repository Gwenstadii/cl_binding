from datetime import datetime, timezone
import hashlib,json,math
from pathlib import Path

root=Path('/home/lzx/cl_binding')
run=root/'.server/dual12000_then_right_20261023_01'
report=json.loads((run/'run_summary.json').read_text())
training=Path(report['training_dir'])
trace=training/'train_steps.jsonl'
rows=[json.loads(line) for line in trace.read_text().splitlines()]
assert [row['step'] for row in rows]==list(range(1,12001))
assert all(math.isfinite(row['loss']) and math.isfinite(row['gradient_norm']) for row in rows)
assert all(row['samples_per_goal']==6 and len(row['batch_indices'])==6 for row in rows)
manifest=json.loads((training/'manifest.json').read_text())
assert manifest['config']['old_goals']==['right']
assert manifest['conversion_spec']['goals']==['right']
assert manifest['git_commit']==report['source_commit']
assert manifest['initialization_model_sha256']==report['initial_model_sha256']
assert manifest['optimizer_reset'] and manifest['scheduler_reset']
checkpoints={}
for step in (3000,6000,9000,12000):
    marker=json.loads((training/'checkpoints'/f'{step:06d}'/'CHECKPOINT_COMPLETE.json').read_text())
    assert marker['status']=='pass' and marker['step']==step and marker['frozen_hash_unchanged']
    checkpoints[str(step)]=marker
parent=Path(report['initial_checkpoint'])
final=Path(report['final_checkpoint'])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(parent/'model.safetensors')==report['initial_model_sha256']
processors={p.name:sha(p) for p in parent.glob('*processor*.safetensors')}
assert len(processors)==2
assert processors=={p.name:sha(p) for p in final.glob('*processor*.safetensors')}
result={'status':'pass','verified_at':datetime.now(timezone.utc).isoformat(),
        'continuous_finite_training_updates':12000,'all_batches_right_only_size6':True,
        'new_optimizer_and_scheduler':True,'four_checkpoint_acceptance_markers':checkpoints,
        'parent_model_sha256':sha(parent/'model.safetensors'),'final_model_sha256':sha(final/'model.safetensors'),
        'normalization_processor_hashes_unchanged':processors,'training_trace_sha256':sha(trace),
        'audit_script_sha256':sha(Path(__file__)),'source_commit':report['source_commit']}
(run/'independent_training_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

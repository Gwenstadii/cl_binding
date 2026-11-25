"""Create a named full inference checkpoint from existing tensors, CPU only.

No training, model forward pass or simulator. Every saved tensor is verified
against the original frozen base or the selected155 task tensors.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime,timezone


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main(registry_path):
    import torch
    from safetensors.torch import load_file,save_file
    torch.set_num_threads(2)
    registry=json.loads(registry_path.read_text())
    base=Path(registry['frozen_model_and_processor_base'])
    source=Path(registry['source_trainable'])
    target=Path(registry['checkpoint_dir'])
    assert target.name=='F4-B24' and not target.exists()
    assert sha(source)==registry['source_trainable_sha256']
    assert sha(base/'model.safetensors')==registry['base_model_sha256']
    assert sha(Path(registry['three_task_parent']))==registry['three_task_parent_sha256']
    original=load_file(str(base/'model.safetensors'),device='cpu')
    selected=load_file(str(source),device='cpu')
    assert len(selected)==155 and set(selected)<=set(original)
    combined=dict(original)
    for key,value in selected.items():
        assert value.shape==original[key].shape and value.dtype==original[key].dtype
        combined[key]=value.contiguous()
    target.mkdir(parents=True)
    ignored={'model.safetensors','training_state.pt','CHECKPOINT_COMPLETE.json','README.md'}
    for path in base.iterdir():
        if path.name in ignored:continue
        if path.is_dir():
            assert path.name=='tokenizer',path
            shutil.copytree(path,target/path.name)
        elif path.suffix in {'.json','.safetensors','.txt','.model'}:
            shutil.copyfile(path,target/path.name)
    save_file(combined,str(target/'model.safetensors'),metadata={'format':'pt'})
    restored=load_file(str(target/'model.safetensors'),device='cpu')
    assert set(restored)==set(original)
    assert all(torch.equal(tensor,selected[name] if name in selected else original[name]) for name,tensor in restored.items())
    registry.update(status='ready',full_checkpoint_model_sha256=sha(target/'model.safetensors'),
        materialized_at_utc=datetime.now(timezone.utc).isoformat(),
        tensor_verification=dict(selected_task_tensors=155,total_state_tensors=len(restored),
            all_tensors_bitwise_equal_to_selected_or_frozen_source=True,
            no_training_or_policy_forward_or_simulation=True,inference_checkpoint=True,
            optimizer_state_not_copied=True),
        processor_sha256={p.name:sha(p) for p in target.glob('*processor*.safetensors')})
    body=json.dumps(registry,ensure_ascii=False,indent=2)+'\n'
    (target/'MODEL_ALIAS.json').write_text(body)
    registry_path.write_text(body)
    print(json.dumps(registry,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('registry',type=Path);main(p.parse_args().registry)

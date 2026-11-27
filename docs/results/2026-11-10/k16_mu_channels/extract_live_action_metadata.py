"""Read-only metadata extraction on scutlab; does not load model weights or run rollouts."""
import contextlib
import hashlib
import importlib
import inspect
import io
import json
from pathlib import Path
import sys


ROOT = Path("/home/lzx/cl_binding_stage2a")
CHECKPOINTS = {
    "R": Path("/home/lzx/cl_binding/.server/r1_left_right_20261104_01/train_r1/checkpoints/024000"),
    "F": Path("/home/lzx/cl_binding/.server/r1_then_front_v1/train/checkpoints/012000"),
}


def file_info(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def source(obj):
    path = Path(inspect.getsourcefile(obj))
    lines, line = inspect.getsourcelines(obj)
    return {**file_info(path), "start_line": line, "source": "".join(lines)}


capture = io.StringIO()
with contextlib.redirect_stdout(capture):
    sys.path.insert(0, str(ROOT / "scripts"))
    import torch
    from safetensors.torch import load_file
    checkpoints = {}
    stats = {}
    for role, path in CHECKPOINTS.items():
        config_path = path / "config.json"
        config = json.loads(config_path.read_text())
        post_path = path / "policy_postprocessor.json"
        pre_path = path / "policy_preprocessor.json"
        post = json.loads(post_path.read_text())
        pre = json.loads(pre_path.read_text())
        stat_path = path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        tensors = load_file(str(stat_path), device="cpu")
        stats[role] = {k: v for k, v in tensors.items() if k.startswith("action.")}
        checkpoints[role] = {
            "path": str(path), "config": config, "postprocessor": post, "preprocessor": pre,
            "action_stats": {k: {"shape": list(v.shape), "dtype": str(v.dtype), "values": v.tolist()}
                             for k, v in stats[role].items()},
            "files": [file_info(p) for p in (config_path, post_path, pre_path, stat_path)],
        }
    assert stats["R"].keys() == stats["F"].keys()
    action_stats_identical = all(torch.equal(stats["R"][k], stats["F"][k]) for k in stats["R"])
    import robosuite
    osc_config = robosuite.load_controller_config(default_controller="OSC_POSE")
    osc = importlib.import_module("robosuite.controllers.osc")
    base = importlib.import_module("robosuite.controllers.base_controller")
    control_utils = importlib.import_module("robosuite.utils.control_utils")
    panda = importlib.import_module("robosuite.models.grippers.panda_gripper")
    wrapper = importlib.import_module("libero.libero.envs.env_wrapper")
    evaluation = importlib.import_module("stage1_r0_eval")
    normalize = importlib.import_module("lerobot.processor.normalize_processor")
    smolvla = importlib.import_module("lerobot.policies.smolvla.modeling_smolvla")
    sources = {
        "make_env": source(evaluation.make_env),
        "ControlEnv_init": source(wrapper.ControlEnv.__init__),
        "OSC_init": source(osc.OperationalSpaceController.__init__),
        "OSC_set_goal": source(osc.OperationalSpaceController.set_goal),
        "controller_update": source(base.Controller.update),
        "controller_scale_action": source(base.Controller.scale_action),
        "set_goal_position": source(control_utils.set_goal_position),
        "set_goal_orientation": source(control_utils.set_goal_orientation),
        "PandaGripper_format_action": source(panda.PandaGripper.format_action),
        "normalization_apply_transform": source(normalize._NormalizationMixin._apply_transform),
        "SmolVLA_prepare_action": source(smolvla.SmolVLAPolicy.prepare_action),
    }
    flow_class = next(cls for name, cls in vars(smolvla).items()
                      if inspect.isclass(cls) and cls.__module__ == smolvla.__name__
                      and hasattr(cls, "denoise_step"))
    sources["flow_embed_suffix"] = source(flow_class.embed_suffix)
    sources["flow_denoise_step"] = source(flow_class.denoise_step)
    model_lines = Path(smolvla.__file__).read_text().splitlines()
    projection_lines = [{"line": i + 1, "text": text}
                        for i, text in enumerate(model_lines)
                        if "self.action_in_proj =" in text or "self.action_out_proj =" in text]
    info = {
        "purpose": "K16 read-only metadata: actual checkpoints, processors, action channels and controller reference frame",
        "checkpoints": checkpoints, "action_stats_identical_R_F": action_stats_identical,
        "OSC_POSE_default_config": osc_config,
        "robosuite": {"path": robosuite.__file__, "version": getattr(robosuite, "__version__", None)},
        "projection_declarations": projection_lines, "sources": sources,
        "captured_import_stdout": capture.getvalue(),
        "operations": "Only read config/stat tensors and inspect source; no model weights loaded, environment created, action executed or remote file written.",
    }

print("K16_METADATA_JSON=" + json.dumps(info, ensure_ascii=False, allow_nan=False))

"""K1 runtime: same-draft velocity-field recombination of R0 / FT4 / ER outputs.

At every Euler flow step, every needed (model, instruction) velocity is evaluated on the SAME
action draft x_t and flow time t (one integration path), then recombined in float64:
    mu_M = (v_M(A) + v_M(B)) / 2,  d_M = (v_M(A) - v_M(B)) / 2
    v = mu_B + s * d_used,  s = +1 for instruction A, -1 for B
    d_used = d_D * (a_A / a_D),  a_M = ||d_M|| over 50 positions x 7 effective dims (padding excluded)
so padding dims follow the direction source and K1-01 / K1-08 reproduce FT4 / R0 exactly.
No hooks, no parameter change. Prefix (VLM) caches are rebuilt per replan and deep-copied per query.
"""
import copy
import hashlib
import json
from pathlib import Path

STRIDE = 1000003
EFF = 7
COND = {
    "K1-01": ("F", "F", "F"), "K1-02": ("F", "F", "R"), "K1-03": ("F", "R", "F"), "K1-04": ("F", "R", "R"),
    "K1-05": ("R", "F", "F"), "K1-06": ("R", "F", "R"), "K1-07": ("R", "R", "F"), "K1-08": ("R", "R", "R"),
    "K1-09": ("F", "zero"), "K1-10": ("R", "zero"), "K1-11": ("F", "neg_dR"), "K1-12": ("R", "neg_dR"),
    "K1-13": ("F", "q1"), "K1-14": ("R", "q1"), "K1-15": ("F", "q2"), "K1-16": ("R", "q2"),
    "K1-17": ("F", "dE"), "K1-18": ("E", "dF"),
}
NATIVE = {"NAT-R0": "R", "NAT-FT4": "F", "NAT-ER": "E"}
ROLE_NAME = {"R": "R0", "F": "FT4", "E": "ER"}


def replan_seed(seed, r):
    if not 0 <= r < 40:
        raise ValueError("Unexpected replanning index")
    return seed + STRIDE * r


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def needed_roles(cond):
    if cond in NATIVE:
        return {NATIVE[cond]}
    spec = COND[cond]
    if len(spec) == 3:
        return set(spec)
    mu, special = spec
    extra = {"zero": set(), "neg_dR": {"R"}, "q1": {"R"}, "q2": {"R"}, "dE": {"E"}, "dF": {"F"}}[special]
    return {mu} | extra


class Runtime:
    def __init__(self, pair, root, roles, evaluation, policies, constants, make_masks, integrate,
                 quat2axisangle, make_prefix, torch, np):
        self.torch, self.np, self.constants = torch, np, constants
        self.make_masks, self.integrate, self.q2a, self.make_prefix_fn = make_masks, integrate, quat2axisangle, make_prefix
        self.models, self.pre, self.post, self.sha, self.paths = {}, {}, {}, {}, {}
        for r in roles:
            path = Path(root) / pair[ROLE_NAME[r]]
            policy = evaluation.load_policy(path)
            policy.eval()
            policy.config.n_action_steps = 10
            policy.config.num_steps = 10
            if (policy.config.adapt_to_pi_aloha or policy.model._rtc_enabled() or policy.config.n_obs_steps != 1
                    or not policy.model.config.use_cache or policy.config.chunk_size != 50
                    or policy.config.action_feature.shape[0] != 7):
                raise ValueError("Unexpected action/observation/cache settings")
            self.models[r] = policy
            self.pre[r], self.post[r] = policies.make_pre_post_processors(policy.config, pretrained_path=path)
            self.sha[r] = sha_file(path / "model.safetensors")
            self.paths[r] = str(path)
        self.processor_check = self.check_processors()
        self.max_dim = self.models[sorted(roles)[0]].config.max_action_dim
        rng = np.random.default_rng
        self.Q = {}
        for k in (1, 2):
            a = rng(20261101 + k).standard_normal((6, 6))
            q, r_ = np.linalg.qr(a)
            self.Q[k] = q * np.sign(np.diag(r_))
        self.counts = {"denoise": 0, "prefix": 0, "degenerate_flows": 0, "flows": 0}

    def check_processors(self):
        """All roles must share identical normalizer/unnormalizer tensors (shared action coordinates)."""
        from safetensors.torch import load_file
        out = {}
        for name in ("policy_preprocessor_step_5_normalizer_processor.safetensors",
                     "policy_postprocessor_step_0_unnormalizer_processor.safetensors"):
            tensors = {r: load_file(str(Path(p) / name)) for r, p in self.paths.items()}
            ref_role = sorted(tensors)[0]
            ref = tensors[ref_role]
            used = sorted(k for k in ref if k.startswith(("action.", "observation.")))  # only these enter (un)normalization
            for r, t in tensors.items():
                if any(k not in t or not self.torch.equal(t[k].double().flatten(), ref[k].double().flatten()) for k in used):
                    raise ValueError(f"Normalization tensors differ between roles: {name} {r}")
            ignored = sorted(k for k in ref if k not in used and any(
                k not in t or t[k].shape != ref[k].shape or not self.torch.equal(t[k].double(), ref[k].double()) for t in tensors.values()))
            out[name] = {"compared_keys": len(used), "identical_across": sorted(tensors), "bookkeeping_keys_differing_ignored": ignored}
        return out

    def batch(self, role, obs, instruction):
        torch, np = self.torch, self.np
        raw = {"task": instruction, "observation.state": torch.from_numpy(np.concatenate((
            obs["robot0_eef_pos"], self.q2a(obs["robot0_eef_quat"].copy()), obs["robot0_gripper_qpos"])).astype(np.float32))}
        for key, name in (("observation.images.image", "agentview_image"), ("observation.images.image2", "robot0_eye_in_hand_image")):
            raw[key] = torch.from_numpy(np.ascontiguousarray(obs[name][::-1, ::-1])).permute(2, 0, 1).float() / 255
        return self.pre[role](copy.deepcopy(raw))

    def prefix(self, role, obs, instruction):
        policy = self.models[role]
        b = self.batch(role, obs, instruction)
        images, masks = policy.prepare_images(b)
        inputs = (images, masks, b[self.constants.OBS_LANGUAGE_TOKENS], b[self.constants.OBS_LANGUAGE_ATTENTION_MASK],
                  policy.prepare_state(b))
        self.counts["prefix"] += 1
        return self.make_prefix_fn(policy, inputs, self.make_masks)

    def noise(self, seed, r):
        g = self.torch.Generator(device="cpu").manual_seed(replan_seed(seed, r))
        return self.torch.randn((1, 50, self.max_dim), generator=g, dtype=self.torch.float32, device="cpu").cuda()

    def query(self, role, prefix, x_t, t):
        pad, original, _ = prefix
        cache = copy.deepcopy(original)
        if any(a.keys.data_ptr() == b.keys.data_ptr() for a, b in zip(cache.layers, original.layers)):
            raise AssertionError("KV cache alias")
        v = self.models[role].model.denoise_step(pad, cache, x_t.clone(), t.clone())
        if v.dtype not in (self.torch.float32, self.torch.bfloat16) or not self.torch.isfinite(v).all():
            raise ValueError("Unexpected velocity dtype/values")
        self.vdtype = v.dtype
        if any(layer.keys.shape[-2] != pad.shape[1] for layer in cache.layers):
            raise AssertionError("Denoise did not restore cropped prefix")
        self.counts["denoise"] += 1
        return v

    def physical(self, chunk):
        role = sorted(self.post)[0]  # identical across roles (checked)
        out = self.post[role](chunk.clone()).float().cpu().numpy()
        if out.shape != (1, 50, 7) or not self.np.isfinite(out).all():
            raise ValueError("Unexpected postprocessed action")
        return out[0]

    # ---------------- recombination ----------------
    def _eff_norm(self, z):
        return float(self.np.sqrt((z[..., :EFF] ** 2).sum()))

    def _rotate(self, d, k):
        q = d.copy()
        q[0, :, :6] = d[0, :, :6] @ self.Q[k].T
        return q

    def combine(self, cond, s, v, stats):
        np = self.np
        mu = {m: (v[m, "A"] + v[m, "B"]) / 2 for m in {k[0] for k in v}}
        d = {m: (v[m, "A"] - v[m, "B"]) / 2 for m in mu}
        a = {m: self._eff_norm(d[m]) for m in d}
        for m in a:
            stats.setdefault("a_" + m, []).append(a[m])
        if "R" in d and "F" in d:
            uR, uF = d["R"][..., :EFF].ravel(), d["F"][..., :EFF].ravel()
            stats.setdefault("cos_uR_uF", []).append(float(uR @ uF / max(a["R"] * a["F"], 1e-30)))
            stats.setdefault("mu_rel_F_R", []).append(self._eff_norm(mu["F"] - mu["R"]) / max(self._eff_norm(mu["R"]), 1e-30))
        if "R" in d and "E" in d:
            uR, uE = d["R"][..., :EFF].ravel(), d["E"][..., :EFF].ravel()
            stats.setdefault("cos_uR_uE", []).append(float(uR @ uE / max(a["R"] * a["E"], 1e-30)))
        spec = COND[cond]
        B = spec[0]
        degenerate, nerr = False, 0.0
        if len(spec) == 3:
            _, D, A = spec
            if a[D] < 1e-4 * self._eff_norm(mu[B]):
                degenerate, used = True, np.zeros_like(d[D])
            else:
                ratio = a[A] / a[D]
                used = d[D] * ratio if A != D else d[D]
                nerr = abs(self._eff_norm(used) - a[A]) / max(a[A], 1e-30)
        else:
            special = spec[1]
            if special == "zero":
                used = np.zeros_like(mu[B])
            elif special == "neg_dR":
                used = -d["R"]
            elif special in ("q1", "q2"):
                used = self._rotate(d["R"], int(special[1]))
                nerr = abs(self._eff_norm(used) - a["R"]) / max(a["R"], 1e-30)
            elif special == "dE":
                used = d["E"]
            elif special == "dF":
                used = d["F"]
        stats.setdefault("degenerate", []).append(int(degenerate))
        stats.setdefault("norm_err", []).append(nerr)
        return mu[B] + s * used

    def sample(self, cond, instr, obs, texts, seed, r):
        """One 50-step normalized action chunk for condition cond under eval instruction instr ('A'/'B')."""
        torch = self.torch
        stats = {}
        x0 = self.noise(seed, r)
        flow = [0]
        if cond in NATIVE:
            role = NATIVE[cond]
            pre = self.prefix(role, obs, texts[instr])

            def velocity(x_t, t):
                flow[0] += 1
                return self.query(role, pre, x_t, t)
        else:
            roles = needed_roles(cond)
            pres = {(m, g): self.prefix(m, obs, texts[g]) for m in roles for g in ("A", "B")}
            s = 1.0 if instr == "A" else -1.0

            def velocity(x_t, t):
                flow[0] += 1
                v = {k: self.query(k[0], p, x_t, t).double().cpu().numpy() for k, p in pres.items()}
                out = self.combine(cond, s, v, stats)
                self.counts["flows"] += 1
                self.counts["degenerate_flows"] += stats["degenerate"][-1]
                # same dtype as a native velocity (bf16 under autocast), so K1-01/K1-08 equal natives bit for bit
                return torch.from_numpy(out).to(device=x_t.device, dtype=self.vdtype)
        actions = self.integrate(velocity, x0, 10)[..., :EFF]
        if flow[0] != 10:
            raise AssertionError("Expected ten Euler flow steps")
        meta = {"replan": r, "noise_sha": hashlib.sha256(x0.cpu().numpy().tobytes()).hexdigest()[:16]}
        for k, val in stats.items():
            meta[k] = [round(float(x), 6) for x in val]
        return actions, meta

    def fields(self, cond_roles, obs, texts, seed, r, order=("A", "B")):
        """Engineering: velocities of the first flow step for given roles and instruction order."""
        x0 = self.noise(seed, r)
        t = self.torch.ones((1,), device=x0.device, dtype=self.torch.float32)
        out = {}
        for m in cond_roles:
            pa, pb = self.prefix(m, obs, texts[order[0]]), self.prefix(m, obs, texts[order[1]])
            va = self.query(m, pa, x0, t).double().cpu().numpy()
            vb = self.query(m, pb, x0, t).double().cpu().numpy()
            out[m] = ((va + vb) / 2, (va - vb) / 2)
        return out

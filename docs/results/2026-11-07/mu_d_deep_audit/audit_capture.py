"""One fixed-input audit of activation aliases, true module outputs and final normalization."""
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/lzx/cl_binding/.server')
sys.path.insert(0, str(ROOT / 'k1_screen_20261101_01/code'))
import k1_main as km


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    original = ROOT / 'k13_r1_front_20_20261107_01'
    legacy = load('audit_legacy', original / 'k11_reference.py')
    fixed = load('audit_fixed', HERE / 'k11_reference_capture_fixed.py')
    ctx = km.Ctx('scene2')
    torch, np = ctx.torch, ctx.np
    torch.set_num_threads(2)
    rt = km.make_runtime(ctx, {'R0':str(ROOT/'r1_left_right_20261104_01/train_r1/checkpoints/024000'),
                              'FT4':str(ROOT/'r1_then_front_v1/train/checkpoints/012000')}, ['R','F'])
    assert rt.sha == {'R':'225322ba28d0ad152fcced19950c43e3e40a435f1835cf64a664226da7830003',
                      'F':'8ff1bf8f372d728a3218101a3cc93fb6fb9d484e8fefb39c48d81a0c8ab2c515'}
    rows = [json.loads(l) for l in (original/'i21.jsonl').read_text().splitlines()]
    row = next(r for r in rows if r['condition']=='ft4' and r['goal']=='B')
    traces = [json.loads(l) for l in (original/'i21_traces.jsonl').read_text().splitlines()]
    trace = next(r['rows'] for r in traces if r['condition']=='ft4' and r['goal']=='B')
    obs, _ = ctx.reset(21)
    for r in trace[:row['boundary']]:
        obs, _ = ctx.step(np.asarray(r['a'], dtype=np.float32))
        if r['step'] % 10 == 0:
            obs = ctx.fresh(ctx.env)
    assert ctx.signature(obs) == row['boundary_signature']
    instr = {g:f'pick up the book and place it in the {label} compartment of the caddy' for g,label in (('A','left'),('B','right'))}
    x = rt.noise(row['seed'], row['boundary']//10)
    t = torch.ones(1, dtype=torch.float32, device=x.device)
    observations, summaries, aliases_proved = {}, [], []
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
        prefixes = {(r,g):rt.prefix(r,obs,instr[g]) for r in 'RF' for g in 'AB'}
        for role in 'RF':
            model = rt.models[role].model.vlm_with_expert.lm_expert
            sites = legacy.modules(rt.models[role])
            for goal in 'AB':
                pure, alias, residual, norm = {}, {}, {}, {}
                def capture(name):
                    def hook(m,inputs,out):
                        pure[name] = out.detach().clone()
                        alias[name] = out.detach()
                    return hook
                def pre(name, dictionary):
                    def hook(m,inputs): dictionary[name] = inputs[0].detach().clone()
                    return hook
                def after_norm(m,inputs,out): norm['after'] = out.detach().clone()
                native = rt.query(role,prefixes[role,goal],x,t)
                handles = [m.register_forward_hook(capture(name)) for name,m in sites.items()]
                handles += [model.layers[7].post_attention_layernorm.register_forward_pre_hook(pre('CA-7',residual)),
                            model.layers[10].input_layernorm.register_forward_pre_hook(pre('MLP-9',residual)),
                            model.norm.register_forward_pre_hook(pre('before',norm)),
                            model.norm.register_forward_hook(after_norm)]
                try:
                    observed = rt.query(role,prefixes[role,goal],x,t)
                finally:
                    for h in handles: h.remove()
                assert torch.equal(native,observed)
                observations[role,goal] = {'pure':pure,'alias':alias,'norm':norm}
                for name in sites:
                    same_residual = torch.equal(alias[name],residual[name])
                    same_pure = torch.equal(alias[name],pure[name])
                    assert same_residual and not same_pure
                    aliases_proved.append({'model':role,'goal':goal,'site':name,'alias_equals_post_residual':same_residual,
                        'alias_equals_pure_module_output':same_pure,'shape':list(pure[name].shape),
                        'relative_change_after_hook':float((alias[name].float()-pure[name].float()).norm()/pure[name].float().norm())})
        corrections = {}
        for name in legacy.SITES:
            differences = {capture:{r:(observations[r,'A'][capture][name].float()-observations[r,'B'][capture][name].float())/2 for r in 'RF'} for capture in ('pure','alias')}
            a = differences['pure']['R'] - differences['pure']['F']
            b = differences['alias']['R'] - differences['alias']['F']
            corrections[name] = {'legacy_norm_over_fixed_norm':float(b.norm()/a.norm()),
                'legacy_fixed_cosine':float((a.flatten()@b.flatten())/(a.norm()*b.norm())),
                'relative_difference':float((a-b).norm()/a.norm())}
        final_norm = {}
        for role in 'RF':
            final_norm[role] = {}
            for at in ('before','after'):
                a,b = (observations[role,g]['norm'][at].float()[:,-50:,:][:,:10,:].flatten() for g in 'AB')
                mu,d = (a+b)/2,(a-b)/2
                final_norm[role][at] = {'mean_norm':float(mu.norm()),'difference_norm':float(d.norm()),
                    'difference_mean_cosine':float((d@mu)/(d.norm()*mu.norm()))}
        for label, helper in (('legacy',legacy),('capture_fixed',fixed)):
            diagnostics = defaultdict(list)
            helper.velocity_pair(rt,{r:helper.modules(rt.models[r]) for r in 'RF'},prefixes,x,t,'joint',
                                 row['seed']*10000+(row['boundary']//10)*100+2,diagnostics)
            summaries.append({'implementation':label,**{k:v[0] for k,v in diagnostics.items()}})
    result = {'scope':'one fixed input, R1/Front-12000, init21 adapted Right prefix at60, first flow time; no new rollout outcome',
        'models':rt.sha,'boundary_signature_match':True,'four_native_outputs_unchanged_by_audit_hooks':True,
        'alias_checks':aliases_proved,'correction_comparison':corrections,'joint_projection':summaries,
        'final_norm_single_input_observations':final_norm,
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (original/'k11_reference.py',HERE/'k11_reference_capture_fixed.py',Path(__file__))}}
    (HERE/'audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    ctx.env.close()


if __name__=='__main__': main()

"""Compare 1000-step no-rotation models with preset full checkpoints at similar demo error."""
import argparse
from collections import Counter
import json

from summarize import counts, demo_summary, jsonl
from evaluate import Lab5, episode, load_branch, await_file
from train_controls import RUN, SEEDS, SAVE, read, write

import torch


def select():
    rows=[]
    for seed in SEEDS:
        def error(arm,step):
            file=RUN/f"probe_{seed}_{arm}_{step:06d}.json"
            await_file(file)
            return demo_summary(read(file)["front_demo"])["all"]["all"]["median"]
        target=error("no_rot",1000)
        full={step:error("full",step) for step in SAVE}
        nearest=min(SAVE,key=lambda step:abs(full[step]-target))
        relative=abs(full[nearest]/target-1)
        rows.append({"seed":seed,"masked_step":1000,"masked_mse":target,"full_step":nearest,
                     "full_mse":full[nearest],"relative_difference":relative,
                     "eligible":relative<=.1,"all_full_mse":full})
    write(RUN/"fit_selection.json",{"comparisons":rows,"rule":"closest full at predefined 30/100/300/1000 to no_rot 1000 within 10% relative median demo MSE"})


def run(lane,lanes):
    selected=[r for r in read(RUN/"fit_selection.json")["comparisons"] if r["eligible"] and r["full_step"]!=1000]
    if selected:
        lab=Lab5()
        records=read(RUN/"holdout.json")["selected"][lane::lanes]
        try:
            with (RUN/f"fit_w{lane}.jsonl").open("x") as handle:
                for item in selected:
                    seed,step=item["seed"],item["full_step"]
                    load_branch(lab,seed,"full",step)
                    for rec in records:
                        for goal in "BN":
                            row={"seed":seed,"step":step,**episode(lab,rec,goal)}
                            handle.write(json.dumps(row,allow_nan=False)+"\n")
                            handle.flush()
                            print("FIT_NATIVE",lane,seed,step,rec["init"],goal,row["first"],flush=True)
        finally:
            lab.ctx.env.close()
    write(RUN/f"fit_w{lane}_COMPLETE.json",{"lane":lane})


def aggregate():
    selected={r["init"] for r in read(RUN/"holdout.json")["selected"]}
    data=read(RUN/"fit_selection.json")
    primary=[r for f in sorted(RUN.glob("native_w[0-5].jsonl")) for r in jsonl(f)]
    extra=[r for f in sorted(RUN.glob("fit_w[0-5].jsonl")) for r in jsonl(f)]
    expected=sum(r["eligible"] and r["full_step"]!=1000 for r in data["comparisons"])*40
    assert len(extra)==expected
    for row in data["comparisons"]:
        if not row["eligible"]:
            continue
        seed,step=row["seed"],row["full_step"]
        candidates=([r for r in primary if r["seed"]==seed and r["arm"]=="full"] if step==1000
                    else [r for r in extra if r["seed"]==seed and r["step"]==step])
        masked=[r for r in primary if r["seed"]==seed and r["arm"]=="no_rot"]
        for label,group in (("full",candidates),("no_rot",masked)):
            row[label]={}
            for g,name in (("B","right"),("N","front")):
                sub=[r for r in group if r["goal"]==g]
                assert len(sub)==20 and {r["init"] for r in sub}==selected
                row[label][name]=counts(sub)
    data["extra_episodes"]=len(extra)
    write(RUN/"fit_summary.json",data)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("select","native","aggregate"))
    p.add_argument("--lane",type=int,default=0)
    p.add_argument("--lanes",type=int,default=6)
    a=p.parse_args()
    torch.set_num_threads(2)
    if a.mode=="select":
        select()
    elif a.mode=="native":
        run(a.lane,a.lanes)
    else:
        aggregate()

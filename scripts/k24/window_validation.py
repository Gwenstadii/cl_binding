"""Measure Front generalization on the independent K19 demonstration set."""
from window_eval import RUN, TRAIN, Lab5, assign, load_weights, write
from eval_k20 import validation_bank, validation_fields


def run():
    lab = Lab5()
    try:
        bank, phases = validation_bank(lab)
        for arm in (*TRAIN, "anchor", "remove_cross", "remove_mlp"):
            assign(lab,"F",load_weights(arm))
            write(RUN/f"demo_{arm}.json",validation_fields(lab,bank,phases))
            print("VALID",arm,flush=True)
    finally:
        lab.ctx.env.close()


if __name__=="__main__":
    run()

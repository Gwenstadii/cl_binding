"""Response composition, independent of the simulator and tensor library."""


def compose(actual,late_mean,parent_mean,delta,background,beta,coefficient):
    if beta == 1.:
        out = actual
        if background == "parent":
            out = out+(parent_mean-late_mean)
    else:
        mean = parent_mean if background == "parent" else late_mean
        out = mean+beta*(actual-late_mean)
    if coefficient:
        out = out+coefficient*delta
    return out

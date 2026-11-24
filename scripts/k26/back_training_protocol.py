"""Deterministic training-only sampling and schedule for Back qualification."""
import math
import random


def phase_ranges(episodes, fraction=.30):
    ranges=[]
    offset=0
    for episode in episodes:
        count=episode['frames']
        assert episode['goal']=='back' and count>1
        ranges.append((offset, offset+count, offset+max(0, count-math.ceil(count*fraction))))
        offset+=count
    assert ranges and 0<fraction<=1
    return ranges,offset


def sample_indices(ranges, seed, step, batch=6):
    """Half uniform whole-demonstration frames, half real final-phase frames."""
    assert batch>0 and batch%2==0 and step>=0
    rng=random.Random((seed+1)*1_000_003+step)
    result=[]
    for late in (False,True):
        for _ in range(batch//2):
            start,end,tail=rng.choice(ranges)
            result.append(rng.randrange(tail if late else start,end))
    return result


def learning_rate(step, total=24000, warmup=300, peak=5e-5, floor=5e-6):
    assert 1<=step<=total and 0<warmup<total and 0<floor<=peak
    if step<=warmup:
        return peak*step/warmup
    progress=(step-warmup)/(total-warmup)
    return floor+.5*(peak-floor)*(1+math.cos(math.pi*progress))

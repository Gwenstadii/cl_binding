"""LIBERO-style requested-goal success events, with per-action termination.

The environment must evaluate the requested task's native BDDL goal. Geometry
region hits, dwell time and gripper contacts are separate diagnostics.
"""
from dataclasses import dataclass

GOALS = ('left', 'front', 'right', 'back')
PROTOCOL = 'K25-native-event-v1'


@dataclass
class NativeEventEpisode:
    goal: str
    horizon: int = 400
    steps: int = 0
    success: bool = False
    done: bool = False
    first_success_step: int | None = None

    def __post_init__(self):
        if self.goal not in GOALS or self.horizon < 1:
            raise ValueError('Known requested goal and positive action horizon required')

    def observe(self, native_predicates):
        """Call once AFTER each executed action, not once per 10-action chunk."""
        if self.done:
            raise RuntimeError('Episode already ended; do not execute further actions')
        if self.goal not in native_predicates:
            raise KeyError('Requested native predicate missing; generic env.done may name another task')
        self.steps += 1
        if bool(native_predicates[self.goal]):
            self.success = self.done = True
            self.first_success_step = self.steps
        elif self.steps == self.horizon:
            self.done = True
        return self.done

    def result(self):
        if not self.done:
            raise ValueError('Incomplete episode cannot be counted as a completed trial')
        return dict(protocol=PROTOCOL, goal=self.goal, success=self.success,
                    first_success_step=self.first_success_step, stop_step=self.steps,
                    termination='native_goal_success' if self.success else 'horizon')


def execute_action_chunk(actions, episode, step_fn, native_predicate_fn):
    """Execute only the prefix before success/horizon, including mid-chunk hits.

    step_fn(action) executes one control; native_predicate_fn(outcome) returns
    flags for the requested task. No model, parent policy or fallback is used.
    """
    iterator = iter(actions)
    last = None
    executed = 0
    while not episode.done:
        try:
            action = next(iterator)
        except StopIteration:
            break
        last = step_fn(action)
        executed += 1
        episode.observe(native_predicate_fn(last))
    return last, executed


def rescore_native_flags(goal, predicates, horizon=400):
    """Truncate an existing trace at its first requested native success."""
    episode = NativeEventEpisode(goal, horizon)
    for flags in predicates:
        if episode.observe(flags):
            return episode.result()
    return episode.result()  # Reject truncated unsuccessful traces.

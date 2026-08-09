"""Strategy A -- uniform random seed selection.

The control arm. Every seed in the corpus gets an equal share of the fuzzing
budget, regardless of size, execution cost, depth, rarity or how productive it has
been.

This is *not* the same as "run AFL++ with no custom scheduling" -- that is strategy
B, and stock AFL++ is heavily prioritised (favored entries, ``n_fuzz`` rarity,
exec-time and bitmap-size weighting). Strategy A deliberately flattens all of it,
which is what makes it a meaningful floor: the gap between A and B is the value of
AFL++'s own scheduling, and the gap between B and C/D is our contribution on top.

Flattening happens in the controller, not here. Returning a constant priority means
"uniform target distribution", and the acceptance controller corrects for AFL++'s
non-uniform proposal distribution using measured offer counts.
"""

from __future__ import annotations

from harness.corpus_model import CorpusView, SeedRecord
from strategies.base import SchedulingStrategy


class RandomStrategy(SchedulingStrategy):
    name = "random"
    description = (
        "Uniform random seed selection; AFL++'s own prioritisation is cancelled "
        "out by the acceptance controller."
    )
    # AFL++ has no uniform power schedule, so we pick its mildest ("explore",
    # which applies no extra factor in calculate_score) and let the controller
    # flatten the residual weighting.
    afl_extra_args = ("-p", "explore")

    def priority(self, seed: SeedRecord, corpus: CorpusView) -> float:
        return 1.0

    def explain(self) -> dict:
        return {"kind": "uniform", "note": "constant priority; no per-seed features used"}

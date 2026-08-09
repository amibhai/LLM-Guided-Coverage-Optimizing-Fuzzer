"""The common interface that all four scheduling strategies implement.

Design constraint
-----------------
We do not patch AFL++. The only seed-scheduling lever AFL++ exposes to an external
module is the custom-mutator callback ``queue_get(filename) -> bool``, which is
called once at the top of ``fuzz_one()`` and, when it returns false, abandons the
entry AFL++ had picked (src/afl-fuzz-one.c:345).

Note that ``fuzz_count()`` -- which looks like an energy-assignment hook -- is
nested inside ``if (el->afl_custom_fuzz)`` (src/afl-fuzz-one.c:1942) and only sizes
the *custom mutator stage*. Using it would mean replacing AFL++'s mutators with our
own, which is explicitly out of scope. So ``queue_get`` is the whole lever.

That makes every strategy an **acceptance filter** over AFL++'s native selection
rather than a replacement for it. Strategies therefore do not choose seeds and do
not assign energy directly. They answer one question:

    "Relative to the other seeds in the corpus, how much of the fuzzing budget
     should this seed get?"

...by returning a non-negative :meth:`SchedulingStrategy.priority`. The harness's
``AcceptanceController`` closes the loop: it observes the rate at which AFL++
*offers* each seed, compares that to the strategy's desired share, and adjusts
per-seed acceptance probabilities until the realised distribution matches the
target. Because energy is realised as "how often this seed gets fuzzed", biasing
acceptance *is* biasing energy, in expectation.

Consequences worth keeping in mind when reading results:

* Effective distribution = AFL++'s base distribution x our acceptance mask,
  renormalised. The controller measures and corrects for the base, but it can
  only bias seeds AFL++ offers at least occasionally.
* AFL++ applies its own ``pending_favored`` skip *after* our hook
  (src/afl-fuzz-one.c:361). That residual is invisible from Python. It applies
  identically to strategies A, C and D, and strategy B is native anyway, so it
  does not bias the comparison -- but it does mean realised shares are measured,
  not assumed. Both offers and accepts are logged for exactly this reason.

Cost accounting
---------------
Every callback below is wrapped by the harness in a ``perf_counter_ns`` /
``process_time_ns`` pair, and the totals are attributed to the strategy. This is
what makes the "coverage per compute-second" comparison honest: strategy D's LLM
latency and tokens are charged to strategy D, not silently absorbed.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, ClassVar

from harness.corpus_model import CorpusView, SeedRecord


@dataclass(frozen=True)
class RunContext:
    """Immutable facts about the run, handed to a strategy at startup."""

    run_id: str
    target: str
    strategy_name: str
    trial: int
    seed: int
    duration_s: float
    out_dir: str
    #: Directory holding artefacts the sidecar analyser publishes (edge maps etc).
    state_dir: str
    #: edge id -> "function_name" from AFL_LLVM_DOCUMENT_IDS, when the target was
    #: built with afl-clang-lto. Empty dict otherwise; strategies must cope.
    edge_to_function: dict[int, str] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    """One accept/reject decision, logged for post-hoc analysis."""

    seed_id: int
    accepted: bool
    priority: float
    accept_prob: float
    t_s: float


class SchedulingStrategy(abc.ABC):
    """Base class for A/B/C/D.

    Subclasses override :meth:`priority` and, if they need periodic work,
    :meth:`on_tick`. Everything else has a working default.
    """

    #: Stable identifier used in filenames, plots and result tables.
    name: ClassVar[str] = "unnamed"

    #: Human-readable description, emitted into the run manifest.
    description: ClassVar[str] = ""

    #: When False the harness does not register ``queue_get`` with AFL++ at all,
    #: leaving AFL++'s native scheduling completely untouched. Strategy B sets
    #: this so that our "standard baseline" really is stock AFL++ and not stock
    #: AFL++ plus an accept-everything callback.
    uses_queue_filter: ClassVar[bool] = True

    #: Extra command-line arguments this strategy needs from afl-fuzz, e.g.
    #: ``["-p", "fast"]``. Merged by the runner.
    afl_extra_args: ClassVar[tuple[str, ...]] = ()

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.ctx: RunContext | None = None

    # -------------------------------------------------------------- lifecycle

    def on_start(self, ctx: RunContext) -> None:
        """Called once, after AFL++ has loaded us but before fuzzing begins."""
        self.ctx = ctx

    def on_stop(self) -> None:
        """Called at teardown. Flush anything buffered here."""

    # ----------------------------------------------------------- corpus events

    def on_new_seed(self, seed: SeedRecord, corpus: CorpusView) -> None:
        """A new queue entry appeared (via AFL++'s ``queue_new_entry`` hook).

        ``seed.edges`` is usually still None at this point -- the showmap replay
        runs asynchronously in the sidecar. Strategies that need edge data should
        read it lazily in :meth:`priority` instead of caching it here.
        """

    def on_analysis(self, seed: SeedRecord, corpus: CorpusView) -> None:
        """Edge data for ``seed`` just arrived from the sidecar analyser."""

    def on_decision(self, decision: Decision, corpus: CorpusView) -> None:
        """A seed was offered by AFL++ and accepted or rejected by us."""

    # ------------------------------------------------------------ the one job

    @abc.abstractmethod
    def priority(self, seed: SeedRecord, corpus: CorpusView) -> float:
        """Desired share of fuzzing budget for ``seed``, as a non-negative weight.

        Only *ratios* matter -- the controller normalises. Returning 0.0 means
        "never fuzz this seed"; the controller still applies a small floor so a
        seed can recover if the strategy changes its mind later.

        This runs on AFL++'s critical path, so it must stay cheap. It is called
        once per ``fuzz_one()`` (a few times per second at typical throughput,
        *not* once per execution), so a handful of dict lookups and float
        operations is well within budget; file I/O and network calls are not.
        """

    def priorities(self, corpus: CorpusView) -> dict[int, float]:
        """Priorities for the whole corpus at once.

        The harness calls *this*, not :meth:`priority`, when refreshing the
        controller (about once a second, off the execution path). The default
        implementation just loops, which is what A and B want.

        Strategies whose scoring is inherently corpus-relative override it:
        rank-normalising a term, or normalising by a corpus-wide median, cannot be
        done correctly one seed at a time. Overriding here keeps that logic in one
        place instead of smearing hidden cross-seed state through ``priority()``.
        """
        return {rec.seed_id: self.priority(rec, corpus) for rec in corpus}

    def on_tick(self, now_s: float, corpus: CorpusView) -> None:
        """Coarse periodic callback, roughly 1 Hz, off the execution path.

        This is where periodic work belongs -- rescoring the corpus, reading a
        plan the LLM planner published, decaying statistics. Strategy D uses it to
        pick up planner output; it never blocks on the LLM here.
        """

    # ------------------------------------------------------------- observability

    def explain(self) -> dict[str, Any]:
        """Current internal state, snapshotted into the run log periodically.

        Used for the qualitative half of the study: for strategy C this is the
        weight vector and term distributions; for strategy D it also carries the
        active plan and the reasoning the model gave for it.
        """
        return {}

    def manifest(self) -> dict[str, Any]:
        """Static description of this strategy, written once into the run manifest."""
        return {
            "name": self.name,
            "description": self.description,
            "uses_queue_filter": self.uses_queue_filter,
            "afl_extra_args": list(self.afl_extra_args),
            "config": self.config,
        }

"""Strategy B -- stock AFL++ ``-p fast``. The standard baseline.

This arm installs *no* scheduling hook at all. ``uses_queue_filter = False`` tells
the bridge not to expose ``queue_get`` to AFL++, so AFL++'s ``fuzz_one()`` never
makes a Python call and its native alias-table scheduling runs completely
untouched.

That distinction matters for the paper's validity. An "accept everything" callback
would still be a callback: it would enter the ``custom_mutators_count`` branch at
src/afl-fuzz-one.c:339 and pay a Python round trip per ``fuzz_one``. Small, but it
would mean the baseline we compare against is not the AFL++ anyone else would run.
By making the hook genuinely absent, strategy B's numbers are directly comparable
to published AFL++ results.

The cost is a deliberate asymmetry: A, C and D pay a per-``fuzz_one`` Python call
that B does not. That overhead is measured rather than assumed -- the harness
records scheduler wall and CPU time for every arm, and the analysis reports
"coverage per compute-second" so the overhead is visible in the comparison instead
of hidden by it.
"""

from __future__ import annotations

from harness.corpus_model import CorpusView, SeedRecord
from strategies.base import SchedulingStrategy


class AFLNativeStrategy(SchedulingStrategy):
    name = "afl_native"
    description = "Stock AFL++ power schedule (-p fast); no external scheduling hook."

    #: The whole point of this arm -- see module docstring.
    uses_queue_filter = False

    afl_extra_args = ("-p", "fast")

    def priority(self, seed: SeedRecord, corpus: CorpusView) -> float:
        # Never consulted: with uses_queue_filter False the harness does not
        # register queue_get and the controller is not driven. Defined only to
        # satisfy the interface.
        return 1.0

    def explain(self) -> dict:
        return {"kind": "native", "schedule": "fast", "hook_installed": False}

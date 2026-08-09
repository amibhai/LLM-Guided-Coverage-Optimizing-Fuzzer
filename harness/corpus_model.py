"""Corpus state shared by every scheduling strategy.

This module owns the *only* place where per-seed statistics live, so that all four
strategies see identical data and differ solely in how they weigh it. Every field
traces back to something AFL++ or ``afl-showmap`` actually produced:

===========================  =========================================================
field                        source
===========================  =========================================================
``seed_id``/``parents``      AFL++ queue filename (``id:``/``src:``)
``discovered_ms``            AFL++ queue filename (``time:``)
``new_cov``                  AFL++ queue filename (``,+cov``, i.e. new_bits == 2)
``op``                       AFL++ queue filename (``op:``, the producing stage)
``size``                     ``os.stat`` on the queue file
``edges``                    ``afl-showmap`` replay of the seed (sidecar, off hot path)
``exec_us``                  wall time of that same ``afl-showmap`` replay
``offers``/``accepts``       counted by our own ``queue_get`` hook
``children``/``child_cov``   derived from ``src:`` back-references of later entries
===========================  =========================================================

Nothing is estimated or back-filled with a placeholder. A seed whose
``afl-showmap`` replay has not completed yet has ``edges is None``, and strategies
are expected to fall back to a documented prior rather than invent a value.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from dataclasses import dataclass, field

from .queue_names import QueueName, parse_queue_name


@dataclass
class SeedRecord:
    """Everything the harness knows about a single corpus entry."""

    seed_id: int
    path: str
    name: QueueName

    size: int = 0
    #: Edge ids covered by this seed, from afl-showmap. None => not yet analysed.
    edges: frozenset[int] | None = None
    #: Single-run execution time from the showmap replay, microseconds.
    exec_us: float | None = None

    #: Distance from the initial corpus, computed by walking ``src:`` links.
    depth: int = 0

    # --- selection accounting, maintained by our queue_get hook ---
    offers: int = 0
    accepts: int = 0

    # --- productivity accounting, derived from later entries' src: fields ---
    children: int = 0
    child_new_cov: int = 0

    #: Monotonic seconds since run start when we first saw the file.
    first_seen_s: float = 0.0

    @property
    def parent_id(self) -> int | None:
        return self.name.parent_id

    @property
    def is_analysed(self) -> bool:
        return self.edges is not None


class CorpusModel:
    """Mutable corpus state. Owned by the harness; strategies get a read-only view.

    Kept deliberately cheap: the hot path (``queue_get``) only does dict lookups
    and float arithmetic. All I/O -- stat-ing files, running ``afl-showmap`` --
    happens in the sidecar analyser and lands here via :meth:`attach_analysis`.
    """

    def __init__(self) -> None:
        self.seeds: dict[int, SeedRecord] = {}
        self.by_path: dict[str, SeedRecord] = {}
        #: edge id -> number of corpus seeds covering it (the rarity denominator)
        self.edge_seed_count: Counter[int] = Counter()
        self.total_offers: int = 0
        self.total_accepts: int = 0

    # ------------------------------------------------------------------ ingest

    def add_seed(self, path: str, first_seen_s: float) -> SeedRecord | None:
        """Register a queue file. Returns None for non-AFL++ filenames."""
        name = parse_queue_name(path)
        if name is None:
            return None
        if name.seed_id in self.seeds:
            return self.seeds[name.seed_id]

        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0

        rec = SeedRecord(
            seed_id=name.seed_id,
            path=path,
            name=name,
            size=size,
            first_seen_s=first_seen_s,
        )
        # Depth follows the src: chain; initial-corpus entries have no parent.
        parent = self.seeds.get(name.parent_id) if name.parent_id is not None else None
        if parent is not None:
            rec.depth = parent.depth + 1
            parent.children += 1
            if name.new_cov:
                parent.child_new_cov += 1

        self.seeds[rec.seed_id] = rec
        self.by_path[path] = rec
        return rec

    def attach_analysis(
        self, seed_id: int, edges: frozenset[int], exec_us: float
    ) -> None:
        """Attach afl-showmap results, updating global edge rarity counts."""
        rec = self.seeds.get(seed_id)
        if rec is None:
            return
        if rec.edges is not None:
            # Re-analysis: retract the old contribution before adding the new one.
            self.edge_seed_count.subtract(rec.edges)
        rec.edges = edges
        rec.exec_us = exec_us
        self.edge_seed_count.update(edges)

    def record_offer(self, seed_id: int, accepted: bool) -> None:
        rec = self.seeds.get(seed_id)
        self.total_offers += 1
        if accepted:
            self.total_accepts += 1
        if rec is not None:
            rec.offers += 1
            if accepted:
                rec.accepts += 1

    # ------------------------------------------------------------------ derived

    def rarity(self, rec: SeedRecord) -> float:
        """How unusual this seed's edges are within the current corpus, in [0, 1].

        Uses the *rarest* edge the seed covers, which is what actually makes a seed
        worth keeping: a seed carrying one edge that only it reaches is valuable
        even if its other 500 edges are ubiquitous. Averaging would wash that out.

        ``1 / count`` mirrors the shape of AFL++'s own ``n_fuzz``-based rarity
        weighting (``weight /= log10(hits) + 1`` in create_alias_table), but is
        computed over *corpus occurrences* rather than execution hit counts, since
        that is what we can observe from outside the fuzzer.
        """
        if not rec.edges:
            return 0.0
        rarest = min(self.edge_seed_count.get(e, 1) for e in rec.edges)
        return 1.0 / float(max(rarest, 1))

    def yield_rate(self, rec: SeedRecord, alpha: float = 1.0, beta: float = 8.0) -> float:
        """Laplace-smoothed rate at which mutating this seed produced new coverage.

        ``(new-coverage children + alpha) / (times selected + beta)``. The prior
        (alpha/beta) makes an unfuzzed seed start at a modest optimism of 1/8
        rather than at 0 (which would starve it forever) or at 1 (which would make
        every new arrival look like a jackpot). Both are config-exposed.
        """
        return (rec.child_new_cov + alpha) / (rec.accepts + beta)

    def cost(self, rec: SeedRecord, median_exec_us: float | None = None) -> float:
        """Relative execution cost, >= 0, where 1.0 is the corpus median.

        Returns 1.0 when the seed has not been analysed yet, so an unanalysed seed
        is treated as average-cost rather than free.
        """
        if rec.exec_us is None:
            return 1.0
        med = median_exec_us if median_exec_us else self.median_exec_us()
        if not med:
            return 1.0
        return rec.exec_us / med

    def median_exec_us(self) -> float:
        vals = sorted(r.exec_us for r in self.seeds.values() if r.exec_us is not None)
        if not vals:
            return 0.0
        n = len(vals)
        return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])

    def total_edges(self) -> int:
        return len(self.edge_seed_count)

    def rank_normalise(self, scores: dict[int, float]) -> dict[int, float]:
        """Map raw scores onto [0, 1] by rank.

        Rank-normalising rather than min-max scaling keeps a single outlier seed
        (one that is 1000x rarer than everything else) from collapsing every other
        seed's normalised score to ~0. It also makes strategy C's weights directly
        comparable to each other, since every term arrives on the same scale.
        """
        if not scores:
            return {}
        if len(scores) == 1:
            return {k: 1.0 for k in scores}
        order = sorted(scores.items(), key=lambda kv: kv[1])
        denom = float(len(order) - 1)
        out: dict[int, float] = {}
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and order[j + 1][1] == order[i][1]:
                j += 1
            # Ties share the midpoint rank so equal inputs get equal outputs.
            shared = ((i + j) / 2.0) / denom
            for k in range(i, j + 1):
                out[order[k][0]] = shared
            i = j + 1
        return out


class CorpusView:
    """Read-only façade handed to strategies.

    Strategies must not mutate corpus state -- if a strategy could, the four arms
    of the experiment would no longer be observing identical data, and the
    comparison would be meaningless. This wrapper makes that a type-level fact
    rather than a code-review convention.
    """

    __slots__ = ("_m",)

    def __init__(self, model: CorpusModel) -> None:
        self._m = model

    def __len__(self) -> int:
        return len(self._m.seeds)

    def __iter__(self):
        return iter(self._m.seeds.values())

    def get(self, seed_id: int) -> SeedRecord | None:
        return self._m.seeds.get(seed_id)

    def analysed_seeds(self):
        return (r for r in self._m.seeds.values() if r.is_analysed)

    def rarity(self, rec: SeedRecord) -> float:
        return self._m.rarity(rec)

    def yield_rate(self, rec: SeedRecord, alpha: float = 1.0, beta: float = 8.0) -> float:
        return self._m.yield_rate(rec, alpha, beta)

    def cost(self, rec: SeedRecord, median_exec_us: float | None = None) -> float:
        return self._m.cost(rec, median_exec_us)

    def median_exec_us(self) -> float:
        return self._m.median_exec_us()

    def edge_seed_count(self, edge: int) -> int:
        return self._m.edge_seed_count.get(edge, 0)

    def total_edges(self) -> int:
        return self._m.total_edges()

    def rank_normalise(self, scores: dict[int, float]) -> dict[int, float]:
        return self._m.rank_normalise(scores)

    @property
    def total_offers(self) -> int:
        return self._m.total_offers

    def depth_histogram(self) -> dict[int, int]:
        h: Counter[int] = Counter()
        for r in self._m.seeds.values():
            h[r.depth] += 1
        return dict(h)

    def summary(self) -> dict:
        """Compact corpus statistics -- also the payload the LLM planner sees."""
        seeds = list(self._m.seeds.values())
        analysed = [s for s in seeds if s.is_analysed]
        counts = self._m.edge_seed_count
        singleton_edges = sum(1 for c in counts.values() if c == 1)
        return {
            "corpus_size": len(seeds),
            "analysed": len(analysed),
            "total_edges": len(counts),
            "singleton_edges": singleton_edges,
            "median_exec_us": self.median_exec_us(),
            "median_size": _median([s.size for s in seeds]),
            "max_depth": max((s.depth for s in seeds), default=0),
            "new_cov_seeds": sum(1 for s in seeds if s.name.new_cov),
            "total_offers": self._m.total_offers,
            "total_accepts": self._m.total_accepts,
        }


def _median(vals: list[int]) -> float:
    if not vals:
        return 0.0
    v = sorted(vals)
    n = len(v)
    return float(v[n // 2]) if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def entropy(counts) -> float:
    """Shannon entropy in bits -- used to report corpus edge-distribution spread."""
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h

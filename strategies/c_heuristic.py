"""Strategy C -- local, interpretable, non-LLM seed scorer.

The score is a weighted sum of three terms, each rank-normalised onto [0, 1] so the
weights are directly comparable and individually tunable::

    priority(s) = w_rarity * Rhat(s)
                + w_yield  * Yhat(s)
                + w_cheap  * (1 - Chat(s))

with default weights (0.40, 0.40, 0.20) summing to 1, so priority itself lands in
[0, 1] and reads as "expected value per unit of budget".

Provenance of every term
------------------------
No term is a placeholder or a guess; each is computed from data the fuzzer or a
replay actually produced.

``Rhat`` -- **edge rarity.** For each seed we take the corpus-wide occurrence count
    of its *rarest* edge and score ``1 / count`` (CorpusModel.rarity), then
    rank-normalise. Edge sets come from replaying each new seed under
    ``afl-showmap`` in the sidecar analyser. We use the rarest edge rather than the
    mean because a seed carrying one otherwise-unreachable edge is valuable even if
    its other 500 edges are ubiquitous -- averaging would bury exactly the signal
    we want. This mirrors the shape of AFL++'s own ``weight /= log10(hits) + 1``
    rarity weighting (src/afl-fuzz-queue.c:169), but is computed over corpus
    occurrences, which is what is observable from outside the fuzzer.

``Yhat`` -- **parent -> child new-coverage yield.** AFL++ stamps every new queue
    entry with ``src:<parent id>`` and appends ``,+cov`` when the entry hit a
    genuinely new edge (``new_bits == 2``, src/afl-fuzz-bitmap.c:426). Walking
    those two fields gives exact, fuzzer-reported attribution of new coverage back
    to the seed that produced it -- no inference. We score the Laplace-smoothed
    rate ``(new-cov children + alpha) / (times selected + beta)``. The prior starts
    an unfuzzed seed at a modest 1/8 rather than at 0 (permanent starvation) or 1
    (every new arrival looks like a jackpot).

``Chat`` -- **execution cost.** Wall time of the seed's ``afl-showmap`` replay,
    relative to the corpus median, rank-normalised; it enters as ``1 - Chat`` so
    cheap seeds score high. Rationale: at equal expected yield, a seed that runs in
    half the time buys twice the executions. Seed size is deliberately *not* a
    separate term -- it correlates strongly with exec time on parsing targets, and
    including both would double-count. It is logged for analysis regardless.

Seeds that the sidecar has not analysed yet (``edges is None``) have no rarity or
cost measurement. Rather than invent one, they receive ``unanalysed_priority``
(default 0.5, mid-scale) until real data arrives, and the count of such seeds is
reported in :meth:`explain` so a run where analysis lags badly is visible rather
than silently degraded.
"""

from __future__ import annotations

from typing import Any

from harness.corpus_model import CorpusView, SeedRecord
from strategies.base import SchedulingStrategy

DEFAULT_WEIGHTS: dict[str, float] = {
    "rarity": 0.40,
    "yield": 0.40,
    "cheapness": 0.20,
}


class HeuristicStrategy(SchedulingStrategy):
    name = "heuristic"
    description = (
        "Interpretable weighted scorer over edge rarity, parent->child new-coverage "
        "yield, and execution cost."
    )
    afl_extra_args = ("-p", "explore")

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        cfg = self.config
        self.weights: dict[str, float] = {
            **DEFAULT_WEIGHTS,
            **(cfg.get("weights") or {}),
        }
        self.yield_alpha: float = float(cfg.get("yield_alpha", 1.0))
        self.yield_beta: float = float(cfg.get("yield_beta", 8.0))
        self.min_priority: float = float(cfg.get("min_priority", 0.01))
        self.unanalysed_priority: float = float(cfg.get("unanalysed_priority", 0.5))

        # Populated on each priorities() pass; surfaced by explain() and consumed
        # by strategy D, which layers region multipliers on top of these scores.
        self._last_terms: dict[int, dict[str, float]] = {}
        self._last_priorities: dict[int, float] = {}
        self._unanalysed: int = 0

    # ------------------------------------------------------------------ scoring

    def priorities(self, corpus: CorpusView) -> dict[int, float]:
        """Score the whole corpus.

        Rank normalisation is corpus-relative, so this must be a batch operation;
        see SchedulingStrategy.priorities. Runs about once a second, off AFL++'s
        execution path.
        """
        records = list(corpus)
        if not records:
            return {}

        analysed = [r for r in records if r.is_analysed]
        self._unanalysed = len(records) - len(analysed)

        median_exec = corpus.median_exec_us()
        raw_rarity: dict[int, float] = {}
        raw_yield: dict[int, float] = {}
        raw_cost: dict[int, float] = {}

        for rec in analysed:
            raw_rarity[rec.seed_id] = corpus.rarity(rec)
            raw_yield[rec.seed_id] = corpus.yield_rate(
                rec, self.yield_alpha, self.yield_beta
            )
            raw_cost[rec.seed_id] = corpus.cost(rec, median_exec)

        # Rank-normalise each term independently so a single extreme seed cannot
        # dominate, and so the weights stay interpretable against each other.
        rhat = corpus.rank_normalise(raw_rarity)
        yhat = corpus.rank_normalise(raw_yield)
        chat = corpus.rank_normalise(raw_cost)

        w = self.weights
        out: dict[int, float] = {}
        terms: dict[int, dict[str, float]] = {}

        for rec in records:
            sid = rec.seed_id
            if sid not in rhat:
                # Not yet analysed: use the documented neutral prior rather than
                # fabricating rarity/cost numbers we have not measured.
                out[sid] = self.unanalysed_priority
                continue
            r, y, c = rhat[sid], yhat[sid], chat[sid]
            score = w["rarity"] * r + w["yield"] * y + w["cheapness"] * (1.0 - c)
            out[sid] = max(self.min_priority, score)
            terms[sid] = {"rarity": r, "yield": y, "cost": c, "score": score}

        self._last_terms = terms
        self._last_priorities = out
        return out

    def priority(self, seed: SeedRecord, corpus: CorpusView) -> float:
        """Single-seed lookup, served from the last batch pass.

        Falls back to the neutral prior for a seed that appeared since the last
        refresh, rather than doing corpus-wide work on the critical path.
        """
        return self._last_priorities.get(seed.seed_id, self.unanalysed_priority)

    # ------------------------------------------------------------ observability

    def explain(self) -> dict[str, Any]:
        terms = self._last_terms
        n = len(terms)
        if not n:
            return {
                "weights": dict(self.weights),
                "scored_seeds": 0,
                "unanalysed_seeds": self._unanalysed,
            }

        def mean(key: str) -> float:
            return sum(t[key] for t in terms.values()) / n

        top = sorted(
            self._last_priorities.items(), key=lambda kv: kv[1], reverse=True
        )[:10]
        return {
            "weights": dict(self.weights),
            "yield_prior": {"alpha": self.yield_alpha, "beta": self.yield_beta},
            "scored_seeds": n,
            "unanalysed_seeds": self._unanalysed,
            "mean_terms": {
                "rarity": mean("rarity"),
                "yield": mean("yield"),
                "cost": mean("cost"),
            },
            "top_seeds": [{"seed_id": s, "priority": p} for s, p in top],
        }

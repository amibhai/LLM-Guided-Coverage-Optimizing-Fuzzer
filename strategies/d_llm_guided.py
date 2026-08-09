"""Strategy D -- strategy C, periodically re-weighted by an LLM planner.

D *extends* C rather than replacing it. C's three measured terms still do the
per-seed scoring at full frequency; the LLM only adjusts how that scoring is
applied, and only every N seconds or K new-coverage events::

    priority_D(s) = priority_C(s)  *  region_multiplier(s)

with C's weight vector optionally nudged by the same plan. If the planner never
produces a plan -- API down, budget exhausted, malformed response -- every
multiplier stays 1.0 and D degrades to exactly C. That fallback is the point: a
failed LLM call must not be able to look like a scheduling result.

Where the LLM sits
------------------
Not in this process. ``planner/planner_daemon.py`` runs as a **separate OS
process**, so its CPU time and wall-clock latency are attributable and, more
importantly, so a slow API call cannot stall AFL++'s ``fuzz_one()``. The two sides
communicate through an atomically-replaced JSON file:

    planner daemon  --write-->  <state_dir>/plan.json  --read-->  this strategy

:meth:`on_tick` polls that file about once a second and does a cheap mtime check
before parsing. The fuzzing loop never blocks on the model, and there is no code
path from ``queue_get`` to the network.

How a plan becomes a multiplier
-------------------------------
The planner returns priorities over **code regions** (function names), not over
individual seeds -- asking a model to rank thousands of opaque seed ids would be
both expensive and meaningless. Function names come from ``AFL_LLVM_DOCUMENT_IDS``,
which AFL++ writes at compile time under ``afl-clang-lto`` and which maps edge id
to the function that edge lives in.

A seed's multiplier is the maximum multiplier over the functions containing its
*rare* edges (occurrence count <= ``rare_edge_threshold``). Maximum, not mean,
because a seed is worth boosting if it reaches a targeted region *at all* -- most
of its edges will be in common startup/parsing paths that dilute a mean toward 1.0.
Restricting to rare edges keeps the boost pointed at the frontier of that region
rather than at whatever seed happens to touch it incidentally.

If the target was not built with ``afl-clang-lto``, ``edge_to_function`` is empty.
D then has no grounded way to map a plan onto seeds, so it logs the condition once
and runs as pure C. Ungrounded fallbacks (fuzzy-matching function names against
file paths, say) are deliberately absent -- they would manufacture a result.
"""

from __future__ import annotations

import json
import os
from typing import Any

from harness.corpus_model import CorpusView, SeedRecord
from strategies.c_heuristic import HeuristicStrategy


class LLMGuidedStrategy(HeuristicStrategy):
    name = "llm_guided"
    description = (
        "Heuristic scorer with periodic LLM re-weighting of code regions and term "
        "weights; falls back to pure heuristic when no plan is available."
    )
    afl_extra_args = ("-p", "explore")

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        cfg = self.config
        self.plan_path: str | None = None
        self.rare_edge_threshold: int = int(cfg.get("rare_edge_threshold", 3))
        self.max_multiplier: float = float(cfg.get("max_multiplier", 4.0))
        self.min_multiplier: float = float(cfg.get("min_multiplier", 0.25))
        self.poll_interval_s: float = float(cfg.get("plan_poll_interval_s", 1.0))

        self._base_weights = dict(self.weights)
        self._plan: dict[str, Any] | None = None
        self._plan_mtime: float = 0.0
        self._plan_seq: int = 0
        self._last_poll_s: float = -1e9
        self._region_mult: dict[str, float] = {}
        self._edge_to_function: dict[int, str] = {}
        self._grounding_warned = False
        self._applied_plans: int = 0
        #: seed_id -> multiplier from the most recent scoring pass
        self._last_multipliers: dict[int, float] = {}

    # ------------------------------------------------------------------ startup

    def on_start(self, ctx) -> None:  # noqa: ANN001 - RunContext, avoids import cycle
        super().on_start(ctx)
        self.plan_path = os.path.join(ctx.state_dir, "plan.json")
        self._edge_to_function = dict(ctx.edge_to_function or {})
        if not self._edge_to_function and not self._grounding_warned:
            self._grounding_warned = True

    @property
    def is_grounded(self) -> bool:
        """False when we have no edge->function map, i.e. D can only behave as C."""
        return bool(self._edge_to_function)

    # ------------------------------------------------------------- plan polling

    def on_tick(self, now_s: float, corpus: CorpusView) -> None:
        if now_s - self._last_poll_s < self.poll_interval_s:
            return
        self._last_poll_s = now_s
        self._maybe_load_plan()

    def _maybe_load_plan(self) -> None:
        """Load a new plan if the daemon published one. Never raises.

        The daemon writes to a temp file and ``os.replace``s it, so a partial file
        is never observed. We still guard the parse: a malformed plan must leave
        the previous plan in force rather than take the fuzzing run down with it.
        """
        path = self.plan_path
        if not path:
            return
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return
        if mtime <= self._plan_mtime:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                plan = json.load(fh)
        except (OSError, ValueError):
            return

        self._plan_mtime = mtime
        self._apply_plan(plan)

    def _apply_plan(self, plan: dict[str, Any]) -> None:
        """Adopt a validated plan. Unknown or out-of-range values are clamped."""
        if not isinstance(plan, dict):
            return

        regions = plan.get("region_priorities") or {}
        mult: dict[str, float] = {}
        if isinstance(regions, dict):
            for fn, m in regions.items():
                try:
                    val = float(m)
                except (TypeError, ValueError):
                    continue
                # Clamp so one bad generation cannot hand a single region a 10^6x
                # boost and effectively freeze the schedule.
                mult[str(fn)] = max(self.min_multiplier, min(self.max_multiplier, val))
        self._region_mult = mult

        # Optional nudge to C's term weights. Renormalised to sum to 1 so the
        # priority scale stays comparable with strategy C's.
        new_w = plan.get("term_weights")
        if isinstance(new_w, dict) and new_w:
            merged = dict(self._base_weights)
            for k, v in new_w.items():
                if k in merged:
                    try:
                        merged[k] = max(0.0, float(v))
                    except (TypeError, ValueError):
                        pass
            total = sum(merged.values())
            if total > 0:
                self.weights = {k: v / total for k, v in merged.items()}

        self._plan = plan
        self._plan_seq = int(plan.get("seq", self._plan_seq + 1))
        self._applied_plans += 1

    # ------------------------------------------------------------------ scoring

    def priorities(self, corpus: CorpusView) -> dict[int, float]:
        base = super().priorities(corpus)
        if not self._region_mult or not self.is_grounded:
            self._last_multipliers = {}
            return base

        e2f = self._edge_to_function
        rmult = self._region_mult
        thresh = self.rare_edge_threshold
        out: dict[int, float] = {}
        mults: dict[int, float] = {}

        for rec in corpus:
            sid = rec.seed_id
            p = base.get(sid, self.unanalysed_priority)
            m = self._seed_multiplier(rec, corpus, e2f, rmult, thresh)
            if m != 1.0:
                mults[sid] = m
            out[sid] = max(self.min_priority, p * m)

        self._last_multipliers = mults
        return out

    def _seed_multiplier(
        self,
        rec: SeedRecord,
        corpus: CorpusView,
        e2f: dict[int, str],
        rmult: dict[str, float],
        thresh: int,
    ) -> float:
        """Region multiplier for this seed, from the functions holding its rare edges.

        Boosts win over suppressions: if a seed reaches *any* region the planner
        asked for, it is worth fuzzing even if it also touches a deprioritised one.
        Only when every matched region is suppressed do we take the strongest
        suppression.
        """
        if not rec.edges:
            return 1.0
        matched: list[float] = []
        for edge in rec.edges:
            fn = e2f.get(edge)
            if fn is None:
                continue
            m = rmult.get(fn)
            if m is None or m == 1.0:
                continue
            if corpus.edge_seed_count(edge) <= thresh:
                matched.append(m)
        if not matched:
            return 1.0
        boosts = [m for m in matched if m > 1.0]
        return max(boosts) if boosts else min(matched)

    # ------------------------------------------------------------ observability

    def explain(self) -> dict[str, Any]:
        base = super().explain()
        plan = self._plan or {}
        base.update(
            {
                "grounded": self.is_grounded,
                "plans_applied": self._applied_plans,
                "plan_seq": self._plan_seq,
                "active_regions": len(self._region_mult),
                "boosted_seeds": len(self._last_multipliers),
                # Kept in the run log so plan quality can be reviewed qualitatively
                # against what the schedule actually did next.
                "plan_reasoning": plan.get("reasoning"),
                "plan_model": plan.get("model"),
                "plan_cost_usd": plan.get("cost_usd"),
            }
        )
        return base

    def manifest(self) -> dict[str, Any]:
        m = super().manifest()
        m["extends"] = HeuristicStrategy.name
        m["rare_edge_threshold"] = self.rare_edge_threshold
        m["multiplier_range"] = [self.min_multiplier, self.max_multiplier]
        return m

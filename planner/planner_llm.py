"""Swappable planner backends. This is the only file that talks to a model.

The fuzzing loop never imports this module -- ``planner_daemon`` does, in its own
process, and publishes results as JSON. So the prompt, the model, the SDK, even
the provider can change here without touching anything on AFL++'s critical path.

Three backends, all behind :class:`Planner`:

``ClaudePlanner``  the real thing -- Claude API, structured output, cost tracked
``MockPlanner``    deterministic, offline, free; used by CI and pipeline tests
``NullPlanner``    always returns the inert plan; makes "D degrades to C" testable

The mock is not a stub for the sake of a stub. Any measured difference between D
and C has to survive the question "is this the LLM, or just the extra machinery
around it?" Running the D arm against ``MockPlanner`` answers that: same daemon,
same IPC, same plan-application path, same overhead accounting, but the plan
content is a fixed rule instead of a model. It is the ablation the experiment
needs, so it is a first-class backend rather than test scaffolding.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Protocol

from planner.schema import (
    MAX_REGIONS,
    MULT_MAX,
    MULT_MIN,
    PLAN_JSON_SCHEMA,
    Plan,
    PlanObservation,
    empty_plan,
    validate_plan,
)

#: Default model. Claude Opus 5 is the current flagship; the planner runs at most
#: a few times a minute, so per-call cost is dominated by the fuzzing compute it
#: is steering, not the other way around.
DEFAULT_MODEL = "claude-opus-5"

#: USD per million tokens, for the cost column in the results table. Update
#: alongside DEFAULT_MODEL. Recorded per call so a run's LLM spend is a measured
#: number in the data rather than an estimate made at analysis time.
PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    # model: (input, output)
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

SYSTEM_PROMPT = """\
You are directing the seed-scheduling policy of an AFL++ fuzzing campaign.

A local scorer already ranks every seed in the corpus at high frequency using
three measured terms: edge rarity, historical parent->child new-coverage yield,
and execution cheapness. You do not replace it and you do not see individual
seeds. You adjust it, periodically, in two ways:

1. region_priorities -- boost or suppress seeds whose *rare* edges lie inside
   named functions. This is how you steer the campaign toward code you believe
   is reachable but under-explored.
2. term_weights -- reweight the three scorer terms when the campaign's shape
   calls for it (e.g. favour rarity when coverage has plateaued, favour yield
   when a productive lineage is paying off).

Everything in the observation is measured from this run: coverage comes from
replaying the corpus under afl-showmap, function names from the target's
instrumentation map, trends from timestamped queue entries. There are no
estimates in it, and you should not invent any.

How to be useful here:

- Prefer functions that are *partially* covered. A function with some edges hit
  and many unhit is a reachable frontier. A function with zero coverage may be
  unreachable from this harness entirely, and boosting it wastes the budget.
- Say what hypothesis each boost tests, in terms of the evidence you were given.
- Boost a handful of regions, not thirty. A plan that touches everything is
  indistinguishable from no plan at all.
- If the campaign is progressing and nothing stands out, return an empty
  region_priorities array and unchanged weights. "No change" is a real answer,
  and a worthless plan is worse than none -- it costs budget and adds noise.
"""


class Planner(Protocol):
    """The interface strategy D's daemon depends on."""

    name: str

    def plan(self, obs: PlanObservation, seq: int) -> Plan:
        """Produce a plan for the current run state. Must not raise."""
        ...


# --------------------------------------------------------------------------- mock


class MockPlanner:
    """Deterministic, offline planner. No API, no cost, no network.

    Applies one fixed rule: boost the partially-covered functions with the most
    uncovered edges. That is a defensible heuristic on its own, which is exactly
    what makes it a useful control -- if strategy D with a real model does not
    beat strategy D with this, the model is not contributing.
    """

    name = "mock"

    def __init__(self, boost: float = 2.0, top_k: int = 5) -> None:
        self.boost = boost
        self.top_k = top_k

    def plan(self, obs: PlanObservation, seq: int) -> Plan:
        t0 = time.perf_counter()
        ranked = sorted(
            obs.partial_functions,
            key=lambda f: f.get("uncovered", 0),
            reverse=True,
        )[: self.top_k]
        regions = {
            str(f["function"]): self.boost for f in ranked if f.get("function")
        }
        return Plan(
            seq=seq,
            region_priorities=regions,
            reasoning=(
                f"[mock] boosted the {len(regions)} partially-covered functions with "
                "the most uncovered edges"
            ),
            model="mock",
            latency_s=time.perf_counter() - t0,
        )


class NullPlanner:
    """Never plans. Strategy D running on this must behave exactly like C."""

    name = "null"

    def plan(self, obs: PlanObservation, seq: int) -> Plan:
        return empty_plan(seq)


# -------------------------------------------------------------------------- claude


class ClaudePlanner:
    """Claude API planner using structured outputs.

    Constructed lazily so importing this module never requires the SDK or an API
    key -- CI runs the mock backend and must not need either.
    """

    name = "claude"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 4096,
        effort: str = "medium",
        timeout_s: float = 90.0,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.timeout_s = timeout_s
        self.system_prompt = system_prompt
        self._client = None

    def _client_or_none(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError:
            return None
        try:
            # Resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an `ant auth
            # login` profile -- an unset API key does not mean no credentials.
            self._client = anthropic.Anthropic(timeout=self.timeout_s)
        except Exception:
            return None
        return self._client

    # ------------------------------------------------------------------ prompting

    def build_prompt(self, obs: PlanObservation) -> str:
        """Render the observation. Separate method so prompt iteration is testable
        without spending an API call."""
        o = obs.as_dict()
        parts = [
            f"Target: {obs.target}   elapsed: {obs.elapsed_s:.0f}s"
            f"   new edges since last plan: {obs.edges_since_last_plan}",
            "",
            "## Corpus",
            json.dumps(o["corpus"], indent=2, sort_keys=True),
            "",
            "## Coverage trend  [t_seconds, total_edges]",
            json.dumps(obs.coverage_trend[-40:]),
            "",
            "## Partially covered functions (the frontier)",
            json.dumps(obs.partial_functions[:60], indent=2),
            "",
            f"## Never-covered functions ({len(obs.uncovered_functions)} total, "
            "first 100 shown)",
            json.dumps(obs.uncovered_functions[:100]),
        ]
        if obs.crashes:
            parts += ["", "## Unique crashes so far", json.dumps(obs.crashes, indent=2)]
        if obs.previous_plan:
            parts += [
                "",
                "## Your previous plan (judge it against the trend above)",
                json.dumps(obs.previous_plan, indent=2),
            ]
        parts += [
            "",
            f"Return at most {MAX_REGIONS} region_priorities, each multiplier in "
            f"[{MULT_MIN}, {MULT_MAX}].",
        ]
        return "\n".join(parts)

    # ---------------------------------------------------------------------- call

    def plan(self, obs: PlanObservation, seq: int) -> Plan:
        client = self._client_or_none()
        if client is None:
            return empty_plan(seq, error="anthropic SDK unavailable or unconfigured")

        prompt = self.build_prompt(obs)
        t0 = time.perf_counter()
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                # Adaptive thinking: this is a judgement call over noisy evidence,
                # and it runs a few times a minute, so the latency is affordable.
                thinking={"type": "adaptive"},
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": PLAN_JSON_SCHEMA},
                },
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # network, auth, rate limit, validation
            return empty_plan(seq, error=f"{type(exc).__name__}: {exc}")

        latency = time.perf_counter() - t0

        # Check stop_reason before touching content: on a refusal, content is
        # empty (pre-output) or partial (mid-stream), and indexing it would raise
        # inside the daemon.
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            plan = empty_plan(seq, error=f"refusal (category={category})")
            plan.latency_s = latency
            plan.model = self.model
            return plan

        text = next(
            (b.text for b in response.content if getattr(b, "type", None) == "text"),
            None,
        )
        if not text:
            plan = empty_plan(seq, error="no text block in response")
            plan.latency_s = latency
            return plan

        try:
            raw = json.loads(text)
        except ValueError as exc:
            plan = empty_plan(seq, error=f"unparseable JSON: {exc}")
            plan.latency_s = latency
            return plan

        plan = validate_plan(raw, seq)
        plan.latency_s = latency
        plan.model = self.model

        usage = getattr(response, "usage", None)
        if usage is not None:
            plan.input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            plan.output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            plan.cost_usd = estimate_cost_usd(
                self.model, plan.input_tokens, plan.output_tokens
            )
        return plan


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Dollar cost of one planner call, for the compute-cost accounting.

    Returns 0.0 for an unpriced model rather than guessing -- an unknown model
    should show up in the results as a visible zero, not as a fabricated figure.
    """
    rates = PRICING_USD_PER_MTOK.get(model)
    if rates is None:
        return 0.0
    in_rate, out_rate = rates
    return (input_tokens / 1e6) * in_rate + (output_tokens / 1e6) * out_rate


# ------------------------------------------------------------------------ factory

_BACKENDS = {
    "claude": ClaudePlanner,
    "mock": MockPlanner,
    "null": NullPlanner,
}


def build_planner(config: dict[str, Any] | None = None) -> Planner:
    """Construct a planner from run config.

    Backend precedence: explicit config, then ``FUZZHARNESS_PLANNER``, then
    ``claude``. Env override exists so a whole experiment can be re-run against
    the mock -- the ablation -- without editing any config file.
    """
    cfg = dict(config or {})
    backend = cfg.pop("backend", None) or os.environ.get("FUZZHARNESS_PLANNER") or "claude"
    cls = _BACKENDS.get(backend)
    if cls is None:
        known = ", ".join(sorted(_BACKENDS))
        raise KeyError(f"unknown planner backend {backend!r}; known: {known}")
    cfg.pop("interval_s", None)          # daemon-level knobs, not planner-level
    cfg.pop("min_new_edges", None)
    cfg.pop("max_calls", None)
    return cls(**cfg)

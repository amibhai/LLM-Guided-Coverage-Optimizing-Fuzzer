"""Wire format between the planner and strategy D.

Deliberately plain dicts + hand-written validation rather than pydantic: this is
the trust boundary between a language model's output and a scheduling loop that
must not crash six hours into a run. Validation here is total -- every field is
range-checked and clamped, and anything unrecognised is dropped rather than
propagated.

The observation (what the planner sees) is equally explicit. Everything in it is
measured: coverage comes from replayed corpus edge sets, function names from
AFL_LLVM_DOCUMENT_IDS, trends from timestamped queue entries. The planner is never
handed a derived "score" that we could not defend the provenance of.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

#: Bound on how far a single plan may move a region's weight. Also enforced in
#: strategy D; duplicated on purpose, because either side may be swapped out.
MULT_MIN = 0.25
MULT_MAX = 4.0

#: Cap on regions per plan. Keeps prompts and responses bounded, and stops a plan
#: from being so diffuse that it is indistinguishable from no plan at all.
MAX_REGIONS = 40


@dataclass
class PlanObservation:
    """The measured state of the run, as handed to the planner."""

    run_id: str
    target: str
    elapsed_s: float
    #: Corpus statistics from CorpusView.summary()
    corpus: dict[str, Any] = field(default_factory=dict)
    #: Functions with instrumented edges where none have been covered.
    uncovered_functions: list[str] = field(default_factory=list)
    #: function -> covered edge count, for functions with any coverage.
    covered_functions: dict[str, int] = field(default_factory=dict)
    #: Functions where coverage exists but most edges remain unreached -- the
    #: frontier, and usually the most actionable thing in the whole observation.
    partial_functions: list[dict[str, Any]] = field(default_factory=list)
    #: (t_seconds, total_edges) samples, for trend context.
    coverage_trend: list[list[float]] = field(default_factory=list)
    #: Recent unique crash signatures with first-seen timestamps.
    crashes: list[dict[str, Any]] = field(default_factory=list)
    #: What the previous plan did, so the model can course-correct.
    previous_plan: dict[str, Any] | None = None
    #: Edge-coverage delta since the previous planning round.
    edges_since_last_plan: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Plan:
    """A validated planner response."""

    seq: int = 0
    region_priorities: dict[str, float] = field(default_factory=dict)
    term_weights: dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    model: str = ""
    #: Accounting, charged to strategy D in the analysis.
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    #: Set when the planner failed and this is the inert fallback.
    is_fallback: bool = False
    error: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def empty_plan(seq: int, error: str | None = None) -> Plan:
    """The inert plan. All multipliers absent => strategy D behaves as strategy C.

    Used whenever the planner cannot produce something trustworthy. Failing to a
    no-op rather than to a guess is what keeps a broken API call from masquerading
    as a scheduling result in the data.
    """
    return Plan(seq=seq, is_fallback=True, error=error)


def validate_plan(raw: Any, seq: int) -> Plan:
    """Coerce a model response into a :class:`Plan`. Never raises.

    Unknown keys are ignored, out-of-range multipliers are clamped rather than
    rejected (a model that says 100x probably means "as much as possible"), and a
    response that yields nothing usable becomes an explicit fallback so the run log
    distinguishes "planned nothing" from "planner broke".
    """
    if not isinstance(raw, dict):
        return empty_plan(seq, error=f"expected object, got {type(raw).__name__}")

    regions: dict[str, float] = {}
    raw_regions = raw.get("region_priorities")

    # Wire format is a list of {function, multiplier} objects (see
    # PLAN_JSON_SCHEMA for why it is not a map). A dict is also accepted so the
    # mock planner and hand-written fixtures can use the more natural form.
    items: list[tuple[Any, Any]] = []
    if isinstance(raw_regions, list):
        for entry in raw_regions:
            if isinstance(entry, dict):
                items.append((entry.get("function"), entry.get("multiplier")))
    elif isinstance(raw_regions, dict):
        items = list(raw_regions.items())

    for name, val in items[:MAX_REGIONS]:
        if not isinstance(name, str) or not name.strip():
            continue
        try:
            m = float(val)
        except (TypeError, ValueError):
            continue
        if m != m:  # NaN
            continue
        regions[name.strip()] = max(MULT_MIN, min(MULT_MAX, m))

    weights: dict[str, float] = {}
    raw_weights = raw.get("term_weights")
    if isinstance(raw_weights, dict):
        for key in ("rarity", "yield", "cheapness"):
            if key in raw_weights:
                try:
                    w = float(raw_weights[key])
                except (TypeError, ValueError):
                    continue
                if w == w and w >= 0:
                    weights[key] = w

    reasoning = raw.get("reasoning")
    if not isinstance(reasoning, str):
        reasoning = ""

    if not regions and not weights:
        p = empty_plan(seq, error="no usable directives in response")
        p.reasoning = reasoning[:4000]
        return p

    return Plan(
        seq=seq,
        region_priorities=regions,
        term_weights=weights,
        reasoning=reasoning[:4000],
    )


#: JSON Schema handed to the Claude API as ``output_config.format``.
#:
#: Constrained to the subset structured outputs actually supports, which rules
#: out two shapes that would otherwise be the natural choice:
#:
#: * ``region_priorities`` is an **array of objects**, not a
#:   ``{function: multiplier}`` map. A map would need a typed
#:   ``additionalProperties``, and structured outputs only accepts
#:   ``additionalProperties: false``.
#: * Ranges are stated in the ``description`` rather than as ``minimum`` /
#:   ``maximum``, which structured outputs ignores. Bounds are enforced for real
#:   in :func:`validate_plan` and again in strategy D -- prose in a schema is a
#:   hint to the model, never a guarantee, so the clamp has to live in code.
#:
#: Every property is listed in ``required``: structured outputs expects a total
#: schema, and "omit the field" is not a signal we want to have to interpret.
#: The model expresses "no change" as an empty array / neutral weights instead.
PLAN_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": (
                "Why this re-weighting, in 2-4 sentences. Reference specific "
                "function names and the coverage evidence behind each choice. "
                "This is recorded for qualitative analysis, so state the "
                "hypothesis you are testing, not a summary of the input."
            ),
        },
        "region_priorities": {
            "type": "array",
            "description": (
                f"At most {MAX_REGIONS} entries. Only name functions that appear "
                "in the observation -- a name that is not in the target's symbol "
                "map is silently ignored. Return an empty array to leave the "
                "current schedule alone."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "function": {
                        "type": "string",
                        "description": "Function name exactly as given in the observation.",
                    },
                    "multiplier": {
                        "type": "number",
                        "description": (
                            f"Clamped to [{MULT_MIN}, {MULT_MAX}]. Above 1.0 "
                            "prioritises seeds whose rare edges lie in this "
                            "function; below 1.0 deprioritises them."
                        ),
                    },
                },
                "required": ["function", "multiplier"],
                "additionalProperties": False,
            },
        },
        "term_weights": {
            "type": "object",
            "description": (
                "Weights for the local scorer's three terms. Renormalised to sum "
                "to 1 before use, so only ratios matter. Return the current "
                "weights unchanged if no adjustment is warranted."
            ),
            "properties": {
                "rarity": {"type": "number", "description": "Weight on edge rarity."},
                "yield": {
                    "type": "number",
                    "description": "Weight on historical new-coverage yield.",
                },
                "cheapness": {
                    "type": "number",
                    "description": "Weight on execution cheapness.",
                },
            },
            "required": ["rarity", "yield", "cheapness"],
            "additionalProperties": False,
        },
    },
    "required": ["reasoning", "region_priorities", "term_weights"],
    "additionalProperties": False,
}

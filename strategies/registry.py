"""Strategy lookup by name.

Kept as an explicit table rather than an import scan so that the set of arms in an
experiment is a reviewable list, and so a typo in a config fails loudly at startup
instead of silently running the wrong arm for six hours.
"""

from __future__ import annotations

from typing import Any

from strategies.a_random import RandomStrategy
from strategies.b_afl_native import AFLNativeStrategy
from strategies.base import SchedulingStrategy
from strategies.c_heuristic import HeuristicStrategy
from strategies.d_llm_guided import LLMGuidedStrategy

#: Canonical arm order for plots and tables: A, B, C, D.
STRATEGIES: dict[str, type[SchedulingStrategy]] = {
    RandomStrategy.name: RandomStrategy,
    AFLNativeStrategy.name: AFLNativeStrategy,
    HeuristicStrategy.name: HeuristicStrategy,
    LLMGuidedStrategy.name: LLMGuidedStrategy,
}

#: Short labels used in the paper: A/B/C/D.
ARM_LABELS: dict[str, str] = {
    RandomStrategy.name: "A",
    AFLNativeStrategy.name: "B",
    HeuristicStrategy.name: "C",
    LLMGuidedStrategy.name: "D",
}


def build_strategy(name: str, config: dict[str, Any] | None = None) -> SchedulingStrategy:
    try:
        cls = STRATEGIES[name]
    except KeyError:
        known = ", ".join(sorted(STRATEGIES))
        raise KeyError(f"unknown strategy {name!r}; known strategies: {known}") from None
    return cls(config)

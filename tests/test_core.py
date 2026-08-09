"""Tests for the parts that can be verified without AFL++ or a container.

These cover the logic most likely to be silently wrong in a way that would
produce plausible-but-meaningless experimental results: the queue-name parser
(everything about attribution flows through it), the rarity/yield/cost maths,
the acceptance controller's convergence, and the planner's response validation.

Run: python -m pytest tests/ -q   (or: python tests/test_core.py)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.controller import AcceptanceController, ControllerConfig
from harness.corpus_model import CorpusModel, CorpusView
from harness.queue_names import parse_queue_name
from planner.schema import MULT_MAX, MULT_MIN, validate_plan
from strategies.registry import build_strategy


# ------------------------------------------------------------------ queue names


def test_parse_basic():
    q = parse_queue_name("id:000123,src:000045,time:98765,execs:1234567,op:havoc,rep:8,+cov")
    assert q is not None
    assert q.seed_id == 123
    assert q.parents == (45,)
    assert q.parent_id == 45
    assert q.time_ms == 98765
    assert q.execs == 1234567
    assert q.op == "havoc"
    assert q.rep == 8
    assert q.new_cov is True


def test_parse_splice_two_parents():
    q = parse_queue_name("id:000007,src:000001+000003,time:4210,execs:9912,op:splice")
    assert q.parents == (1, 3)
    assert q.parent_id == 1
    assert q.new_cov is False


def test_parse_sync_entry():
    q = parse_queue_name("id:000002,sync:strategy_c,src:000019,time:5000")
    assert q.is_imported and q.sync_from == "strategy_c"


def test_parse_full_path_and_rejects_non_queue():
    assert parse_queue_name("/out/default/queue/id:000001,time:0,execs:0").seed_id == 1
    assert parse_queue_name("README.md") is None
    assert parse_queue_name(".state") is None


def test_parse_unknown_field_is_kept_not_fatal():
    # A future AFL++ adding a field must degrade to "ignored", never to a
    # parse failure that would silently drop the entry from the corpus model.
    q = parse_queue_name("id:000005,src:000001,time:10,execs:20,newfield:xyz")
    assert q.seed_id == 5 and q.extra.get("newfield") == "xyz"


# ---------------------------------------------------------------- corpus model


def _corpus():
    m = CorpusModel()
    m.add_seed("id:000000,time:0,execs:0", 0.0)
    m.add_seed("id:000001,src:000000,time:100,execs:10,op:havoc,+cov", 1.0)
    m.add_seed("id:000002,src:000000,time:200,execs:20,op:havoc", 2.0)
    m.add_seed("id:000003,src:000001,time:300,execs:30,op:havoc,+cov", 3.0)
    return m


def test_depth_and_child_attribution():
    m = _corpus()
    assert m.seeds[0].depth == 0
    assert m.seeds[1].depth == 1
    assert m.seeds[3].depth == 2

    # seed 0 produced two children, one of which hit new coverage
    assert m.seeds[0].children == 2
    assert m.seeds[0].child_new_cov == 1
    # seed 1 produced one child, which hit new coverage
    assert m.seeds[1].child_new_cov == 1


def test_rarity_uses_rarest_edge():
    m = _corpus()
    m.attach_analysis(0, frozenset({1, 2, 3}), 100.0)
    m.attach_analysis(1, frozenset({1, 2, 3}), 100.0)
    m.attach_analysis(2, frozenset({1, 2, 3, 99}), 100.0)

    # Edge 99 is covered by exactly one seed, so seed 2 is maximally rare (1.0)
    # even though its other three edges are the most common in the corpus. A
    # mean would have buried that; this is the behaviour the scorer relies on.
    assert m.rarity(m.seeds[2]) == 1.0
    assert m.rarity(m.seeds[0]) < 1.0


def test_reanalysis_does_not_double_count_edges():
    m = _corpus()
    m.attach_analysis(0, frozenset({1, 2}), 10.0)
    assert m.edge_seed_count[1] == 1
    m.attach_analysis(0, frozenset({2, 3}), 10.0)   # re-analysed with new edges
    assert m.edge_seed_count.get(1, 0) == 0          # old contribution retracted
    assert m.edge_seed_count[2] == 1                 # not doubled
    assert m.edge_seed_count[3] == 1


def test_yield_prior_is_optimistic_not_zero_or_one():
    m = _corpus()
    fresh = m.seeds[2]                    # never selected, no new-cov children
    assert 0.0 < m.yield_rate(fresh) < 0.2   # ~1/8 with the default prior


def test_unanalysed_seed_has_average_cost_not_free():
    m = _corpus()
    assert m.cost(m.seeds[0]) == 1.0


def test_rank_normalise_handles_ties_and_bounds():
    m = CorpusModel()
    out = m.rank_normalise({1: 5.0, 2: 5.0, 3: 9.0, 4: 1.0})
    assert out[4] == 0.0 and out[3] == 1.0
    assert out[1] == out[2]                       # ties share a rank
    assert all(0.0 <= v <= 1.0 for v in out.values())
    assert m.rank_normalise({}) == {}
    assert m.rank_normalise({7: 3.0}) == {7: 1.0}


# ------------------------------------------------------------------ strategies


def test_registry_builds_all_four_arms():
    for name in ("random", "afl_native", "heuristic", "llm_guided"):
        assert build_strategy(name) is not None


def test_only_baseline_skips_the_hook():
    # If this ever flips, strategy B stops being stock AFL++ and the baseline
    # silently becomes something no one else can reproduce.
    assert build_strategy("afl_native").uses_queue_filter is False
    for name in ("random", "heuristic", "llm_guided"):
        assert build_strategy(name).uses_queue_filter is True


def test_random_is_uniform_over_corpus():
    m = _corpus()
    view = CorpusView(m)
    prios = build_strategy("random").priorities(view)
    assert len(set(prios.values())) == 1


def test_heuristic_prefers_rare_cheap_productive_seeds():
    m = CorpusModel()
    m.add_seed("id:000000,time:0,execs:0", 0.0)
    m.add_seed("id:000001,time:0,execs:0", 0.0)
    # seed 0: unique rare edge, fast. seed 1: only common edges, slow.
    m.attach_analysis(0, frozenset({1, 2, 500}), 50.0)
    m.attach_analysis(1, frozenset({1, 2}), 5000.0)

    prios = build_strategy("heuristic").priorities(CorpusView(m))
    assert prios[0] > prios[1]


def test_heuristic_uses_neutral_prior_for_unanalysed():
    m = CorpusModel()
    m.add_seed("id:000000,time:0,execs:0", 0.0)   # never analysed
    s = build_strategy("heuristic")
    prios = s.priorities(CorpusView(m))
    assert prios[0] == s.unanalysed_priority
    assert s.explain()["unanalysed_seeds"] == 1


def test_llm_guided_without_plan_equals_heuristic():
    # The fallback that makes a broken planner impossible to mistake for a result.
    m = _corpus()
    m.attach_analysis(0, frozenset({1, 2, 500}), 50.0)
    m.attach_analysis(1, frozenset({1, 2}), 5000.0)
    m.attach_analysis(2, frozenset({3, 4}), 100.0)
    m.attach_analysis(3, frozenset({5}), 100.0)
    view = CorpusView(m)

    assert build_strategy("llm_guided").priorities(view) == \
           build_strategy("heuristic").priorities(view)


def test_llm_guided_boosts_seeds_reaching_planned_region():
    m = CorpusModel()
    m.add_seed("id:000000,time:0,execs:0", 0.0)
    m.add_seed("id:000001,time:0,execs:0", 0.0)
    m.attach_analysis(0, frozenset({10}), 100.0)   # edge 10 -> png_handle_iCCP
    m.attach_analysis(1, frozenset({20}), 100.0)   # edge 20 -> png_read_row
    view = CorpusView(m)

    d = build_strategy("llm_guided")

    class _Ctx:
        state_dir = "."
        edge_to_function = {10: "png_handle_iCCP", 20: "png_read_row"}

    d.on_start(_Ctx())
    base = d.priorities(view)
    d._apply_plan({"region_priorities": {"png_handle_iCCP": 3.0}, "seq": 1})
    boosted = d.priorities(view)

    assert boosted[0] > base[0]      # reaches the targeted region
    assert boosted[1] == base[1]     # does not


def test_llm_guided_is_inert_without_symbol_map():
    d = build_strategy("llm_guided")

    class _Ctx:
        state_dir = "."
        edge_to_function = {}

    d.on_start(_Ctx())
    assert d.is_grounded is False

    m = CorpusModel()
    m.add_seed("id:000000,time:0,execs:0", 0.0)
    m.attach_analysis(0, frozenset({10}), 100.0)
    view = CorpusView(m)

    d._apply_plan({"region_priorities": {"whatever": 4.0}, "seq": 1})
    assert d.priorities(view) == build_strategy("heuristic").priorities(view)


# ------------------------------------------------------------------ controller


def test_controller_converges_to_target_distribution():
    """The core claim of the acceptance-filter design.

    AFL++ offers seed 1 four times as often as seed 2, but the strategy wants
    them fuzzed equally. The controller must cancel the base distribution out.
    """
    import random

    ctrl = AcceptanceController(ControllerConfig(accept_floor=0.01), rng_seed=7)
    rng = random.Random(11)
    target = {1: 1.0, 2: 1.0}
    accepted = {1: 0, 2: 0}

    for step in range(20000):
        if step % 200 == 0:
            ctrl.refresh(target, now_s=step * 0.01)
        offered = 1 if rng.random() < 0.8 else 2      # 4:1 skew in AFL++'s favour
        ok, _ = ctrl.decide(offered)
        if ok:
            accepted[offered] += 1

    total = accepted[1] + accepted[2]
    share1 = accepted[1] / total
    # Without correction this sits at 0.8. Converging near 0.5 is the whole
    # mechanism working.
    assert 0.40 < share1 < 0.60, f"controller failed to flatten: share1={share1:.3f}"


def test_controller_biases_toward_higher_priority():
    import random

    ctrl = AcceptanceController(ControllerConfig(accept_floor=0.01), rng_seed=3)
    rng = random.Random(5)
    accepted = {1: 0, 2: 0}

    for step in range(20000):
        if step % 200 == 0:
            ctrl.refresh({1: 4.0, 2: 1.0}, now_s=step * 0.01)
        offered = 1 if rng.random() < 0.5 else 2       # unbiased base
        ok, _ = ctrl.decide(offered)
        if ok:
            accepted[offered] += 1

    ratio = accepted[1] / max(accepted[2], 1)
    assert 2.5 < ratio < 6.0, f"expected ~4:1, got {ratio:.2f}"


def test_controller_never_starves_a_seed_completely():
    ctrl = AcceptanceController(ControllerConfig(accept_floor=0.05), rng_seed=1)
    ctrl.refresh({1: 1000.0, 2: 0.0}, now_s=0.0)
    assert ctrl.stats.accept_probs[2] >= 0.05


def test_controller_falls_back_to_uniform_when_nothing_is_wanted():
    ctrl = AcceptanceController(rng_seed=1)
    ctrl.refresh({1: 0.0, 2: 0.0}, now_s=0.0)
    assert all(p == 1.0 for p in ctrl.stats.accept_probs.values())


# --------------------------------------------------------------- plan validation


def test_validate_plan_accepts_wire_array_form():
    plan = validate_plan(
        {
            "reasoning": "png_handle_iCCP is the frontier",
            "region_priorities": [{"function": "png_handle_iCCP", "multiplier": 2.5}],
            "term_weights": {"rarity": 0.5, "yield": 0.3, "cheapness": 0.2},
        },
        seq=1,
    )
    assert plan.region_priorities == {"png_handle_iCCP": 2.5}
    assert plan.is_fallback is False


def test_validate_plan_clamps_out_of_range_multipliers():
    plan = validate_plan(
        {"region_priorities": [{"function": "f", "multiplier": 1e9},
                               {"function": "g", "multiplier": -5}]},
        seq=1,
    )
    assert plan.region_priorities == {"f": MULT_MAX, "g": MULT_MIN}


def test_validate_plan_rejects_garbage_without_raising():
    for bad in (None, "nonsense", 42, [], {"region_priorities": "not a list"}):
        plan = validate_plan(bad, seq=1)
        assert plan.is_fallback and plan.region_priorities == {}


def test_validate_plan_survives_nan_and_bad_types():
    plan = validate_plan(
        {"region_priorities": [
            {"function": "a", "multiplier": float("nan")},
            {"function": "", "multiplier": 2.0},
            {"function": "ok", "multiplier": "3.0"},
        ]},
        seq=1,
    )
    assert plan.region_priorities == {"ok": 3.0}


def test_empty_plan_is_distinguishable_from_a_broken_one():
    good = validate_plan({"region_priorities": [], "reasoning": "steady"}, seq=1)
    assert good.is_fallback and good.error == "no usable directives in response"
    assert good.reasoning == "steady"   # reasoning survives for the qualitative log


if __name__ == "__main__":
    import traceback

    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)

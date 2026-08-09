# LLM-Guided Coverage-Optimizing Fuzzer

A comparative study of fuzzing **seed-scheduling** strategies.

> **Research question.** Does an LLM-guided, periodic corpus-reweighting strategy
> improve fuzzing efficiency (time-to-coverage, time-to-crash) compared to
> conventional scheduling strategies, **when compute cost is accounted for**?

Four scheduling arms run against the same AFL++ build, the same target binary,
the same seed corpus and the same wall-clock budget. We do not write a fuzzer:
AFL++ does the mutation, execution and coverage tracking. The contribution is the
scheduling layer and the evaluation framework around it.

**Status:** harness + strategies + planner implemented and unit-tested
(`28/28`); target build scripts written; **no fuzzing campaign has been run yet**
— the pilot is the next step and needs your go-ahead (see *Running the pilot*).

---

## The central mechanism (read this first)

Everything about this design follows from one constraint, so it is worth stating
before the directory listing.

**AFL++ exposes exactly one seed-scheduling lever to an external module:** the
custom-mutator callback `queue_get(filename) -> bool`, called once at the top of
`fuzz_one()`. Returning false abandons the entry AFL++ had picked
(`src/afl-fuzz-one.c:345`).

`fuzz_count()` *looks* like an energy-assignment hook, and is widely assumed to
be one. It is not: it sits inside `if (el->afl_custom_fuzz)`
(`src/afl-fuzz-one.c:1942`) and only sizes the **custom mutator stage**. Using it
would mean replacing AFL++'s mutators with our own — out of scope, and it would
put CPython in the per-execution path.

So each strategy is an **acceptance filter over AFL++'s native selection**, not a
replacement for it. Strategies return a non-negative `priority` per seed;
`harness/controller.py` turns priorities into accept probabilities by online
rejection sampling, estimating AFL++'s base distribution from the offers it
actually makes:

```
accept_prob(s)  ∝  desired_share(s) / observed_offer_share(s)
```

Because energy is realised as *how often a seed gets fuzzed*, biasing acceptance
is biasing energy in expectation. The controller converges to the target
distribution as long as AFL++ offers every seed occasionally, which its alias
table guarantees. `tests/test_core.py` verifies this directly: given a 4:1 skew
in AFL++'s offers and a uniform target, the realised share converges to ~0.5.

Consequences we report rather than hide:

- The effective distribution is AFL++'s base × our mask, renormalised. The
  controller measures and corrects the base, but can only bias seeds AFL++
  offers at least sometimes.
- AFL++ applies its own `pending_favored` skip *after* our hook
  (`src/afl-fuzz-one.c:361`). That residual is invisible from Python. It applies
  identically to arms A, C and D, and B is native anyway — so it does not bias
  the comparison, but realised shares are **measured, not assumed**. Offers and
  accepts are both logged for exactly this reason.

**Cost of the hook.** `queue_get` fires once per `fuzz_one()`, not once per
execution — single-digit calls per second at typical throughput. We never define
`fuzz()`, so AFL++ skips its custom-mutator stage entirely and no Python touches
the execution path.

---

## Repository structure

```
├── docker/
│   ├── Dockerfile               AFL++ (pinned, Python-enabled) + LLVM toolchain
│   └── requirements.txt         container-side deps only
│
├── targets/                     one directory per benchmark target
│   └── libpng/
│       ├── target.yaml          metadata the runner reads; nothing hardcoded
│       ├── build.sh             fetch @ pinned commit, build 3 variants
│       └── driver.c             AFL++ persistent-mode driver over the OSS-Fuzz harness
│
├── strategies/                  the four arms, behind one interface
│   ├── base.py                  ← SchedulingStrategy: the common interface
│   ├── a_random.py              A. uniform random
│   ├── b_afl_native.py          B. stock AFL++ -p fast  (installs no hook)
│   ├── c_heuristic.py           C. interpretable local scorer
│   ├── d_llm_guided.py          D. C + periodic LLM re-weighting
│   └── registry.py              explicit name → class table
│
├── harness/                     the AFL++ wrapper
│   ├── afl_bridge.py            ← AFL_PYTHON_MODULE entry point (runs in-process)
│   ├── afl_runner.py            launches afl-fuzz + sidecars for one run
│   ├── controller.py            priorities → accept probabilities
│   ├── corpus_model.py          per-seed state; the only source of scoring data
│   ├── queue_names.py           AFL++ queue filename parser (src:/+cov attribution)
│   ├── analyzer.py              sidecar: afl-showmap replay → edge sets
│   ├── symbols.py               edge id → function, from AFL_LLVM_DOCUMENT_IDS
│   └── metrics.py               scheduler wall/CPU accounting + JSONL logs
│
├── planner/                     strategy D's LLM half — swappable, out-of-process
│   ├── planner_llm.py           ← Claude / mock / null backends
│   ├── planner_daemon.py        separate process; publishes plan.json atomically
│   └── schema.py                wire format + total validation of model output
│
├── experiments/                 matrix runner and configs        [next]
├── analysis/                    plots + summary tables           [next]
├── results/                     per-run raw data (git-ignored)
└── tests/test_core.py           28 tests over the schedulable logic
```

### Process layout of one run

```
afl_runner
  ├── afl-fuzz ............ embeds CPython, imports harness.afl_bridge
  │                          → queue_get / queue_new_entry decisions
  ├── analyzer ............ afl-showmap replays → analysis.jsonl
  └── planner_daemon ...... strategy D only; LLM calls → plan.json
```

The analyser and planner are **separate processes on purpose**. `afl-showmap`
forks and executes the target; an LLM call takes seconds. Either one inside
`queue_get` would wreck the throughput numbers the study exists to measure. Out
of process, a slow replay only delays *our knowledge* of a seed, and a slow model
call only delays the *next* plan — AFL++ keeps fuzzing at full rate throughout.

---

## The common strategy interface

All four arms implement `strategies/base.py`. They do not choose seeds and do not
assign energy directly; they answer one question:

> *Relative to the other seeds in the corpus, how much of the fuzzing budget
> should this seed get?*

```python
class SchedulingStrategy(abc.ABC):
    name: ClassVar[str]
    uses_queue_filter: ClassVar[bool] = True     # B sets False → no hook at all
    afl_extra_args: ClassVar[tuple[str, ...]] = ()

    def on_start(self, ctx: RunContext) -> None: ...
    def on_new_seed(self, seed: SeedRecord, corpus: CorpusView) -> None: ...
    def on_analysis(self, seed: SeedRecord, corpus: CorpusView) -> None: ...
    def on_decision(self, decision: Decision, corpus: CorpusView) -> None: ...

    @abc.abstractmethod
    def priority(self, seed: SeedRecord, corpus: CorpusView) -> float: ...
    def priorities(self, corpus: CorpusView) -> dict[int, float]: ...   # batch
    def on_tick(self, now_s: float, corpus: CorpusView) -> None: ...    # ~1 Hz

    def explain(self) -> dict: ...      # snapshotted into the run log
    def manifest(self) -> dict: ...     # written once at startup
```

Three details that carry weight:

- **`priorities()` is the batch form and the one the harness actually calls.**
  Rank-normalisation and median-relative costs are corpus-wide; they cannot be
  computed correctly one seed at a time. Overriding here keeps that logic in one
  place instead of smuggling hidden cross-seed state through `priority()`.
- **`CorpusView` is read-only.** If a strategy could mutate corpus state, the
  four arms would stop observing identical data and the comparison would be
  meaningless. That is a type-level fact, not a code-review convention.
- **`uses_queue_filter = False`** means the hook is genuinely absent, not
  accept-everything. See arm B below.

Every callback is wrapped in a `perf_counter_ns` / `process_time_ns` pair and
charged to the strategy — that is what makes "coverage per compute-second"
honest.

---

## The four arms

| | Arm | What it is | Hook? |
|---|---|---|---|
| **A** | `random` | Uniform target distribution; AFL++'s own prioritisation is cancelled by the controller | yes |
| **B** | `afl_native` | Stock AFL++ `-p fast` | **no** |
| **C** | `heuristic` | Interpretable weighted scorer | yes |
| **D** | `llm_guided` | C, periodically re-weighted by an LLM | yes |

**A is not "no scheduling"** — that is B, and stock AFL++ is heavily prioritised
(favored entries, `n_fuzz` rarity, exec-time and bitmap-size weighting). A
deliberately flattens all of it. The A↔B gap is the value of AFL++'s own
scheduling; the B↔C/D gap is our contribution on top.

**B installs no hook at all.** An accept-everything callback would still enter
the `custom_mutators_count` branch and pay a Python round trip per `fuzz_one`.
Small — but it would mean the baseline is not the AFL++ anyone else would run.
The cost is a deliberate asymmetry (A/C/D pay a per-`fuzz_one` call that B does
not), which we *measure* rather than assume, and report in the cost-normalised
column.

### C — every term maps to measured data

```
priority(s) = w_rarity · R̂(s) + w_yield · Ŷ(s) + w_cheap · (1 − Ĉ(s))
```

with defaults `(0.40, 0.40, 0.20)`, each term rank-normalised to [0, 1] so the
weights are comparable and individually tunable.

| Term | Where it comes from |
|---|---|
| **R̂** edge rarity | Corpus-wide occurrence count of the seed's *rarest* edge, `1/count`, from replaying each seed under `afl-showmap`. The rarest edge, not the mean: a seed carrying one otherwise-unreachable edge is valuable even if its other 500 edges are ubiquitous — averaging buries exactly the signal we want. Mirrors AFL++'s own `weight /= log10(hits)+1` (`afl-fuzz-queue.c:169`), computed over corpus occurrences since that is what is observable from outside. |
| **Ŷ** yield | AFL++ stamps every new queue entry with `src:<parent>` and appends `,+cov` when `new_bits == 2` (`afl-fuzz-bitmap.c:426`) — a genuinely new edge, not a new hitcount bucket. Walking those two fields gives **exact, fuzzer-reported attribution**, no inference. Scored as Laplace-smoothed `(new-cov children + α)/(times selected + β)`; the prior starts an unfuzzed seed at ~1/8 rather than 0 (permanent starvation) or 1 (every arrival a jackpot). |
| **Ĉ** cost | Wall time of the seed's `afl-showmap` replay relative to the corpus median. Enters as `1 − Ĉ`: at equal expected yield, a seed that runs in half the time buys twice the executions. Seed size is deliberately *not* a separate term — it correlates strongly with exec time on parsing targets and would double-count. It is logged anyway. |

Seeds the sidecar has not analysed yet get a documented neutral prior (0.5), and
the count of them is reported in `explain()` — so a run where analysis lags badly
is *visible* rather than silently degraded. Nothing is back-filled with a
placeholder.

### D — C, re-weighted periodically

```
priority_D(s) = priority_C(s) × region_multiplier(s)
```

C still scores every seed at full frequency. The LLM only adjusts how that
scoring is applied, every N seconds **and** only after ≥ K new edges — a
plateaued campaign should not be re-planned at full token cost every interval.

The planner returns priorities over **code regions** (function names), not
individual seeds — ranking thousands of opaque seed ids would be expensive and
meaningless. Function names come from `AFL_LLVM_DOCUMENT_IDS`, which AFL++ emits
at compile time under `afl-clang-lto`, mapping edge id → function. A seed's
multiplier is the max over functions containing its *rare* edges (occurrence
≤ threshold), so the boost points at the frontier of a region rather than at
whatever seed touches it incidentally.

**If the target was not built with LTO, `edge_to_function` is empty and D runs as
pure C.** There is deliberately no ungrounded fallback (no fuzzy-matching names
against paths) — that would manufacture a result. Likewise, every planner failure
path returns the *inert* plan, so a broken API call can never masquerade as a
scheduling result. `planner/planner_llm.py` also ships `mock` and `null`
backends; running D against `mock` is the ablation that separates "the LLM
helped" from "the extra machinery helped".

Every plan is logged to JSONL with its reasoning, model, latency, tokens and
dollar cost, for the qualitative half of the study.

---

## Setup

Requires Docker (AFL++ needs Linux; on Windows use the WSL2 backend).

```bash
docker build -t llmfuzz:latest docker/            # ~10 min; pins AFL++ v4.21c
```

The image build **asserts** that AFL++ compiled with Python support. Without it
`AFL_PYTHON_MODULE` is silently ignored, every arm degrades to stock AFL++, and
the experiment produces plausible-looking but meaningless results — so it fails
the build instead.

Build a target (three variants: `fuzz` for the campaign, `cov` for ground-truth
coverage replay, `asan` for crash triage):

```bash
docker run --rm -v "$PWD:/work" llmfuzz:latest bash targets/libpng/build.sh
```

For strategy D, provide credentials — `ANTHROPIC_API_KEY`, or an `ant auth login`
profile the SDK picks up automatically.

---

## Running the pilot

**Nothing compute-heavy has been run yet.** The intended first step is a short
single-target smoke run to validate the pipeline end to end before spending real
compute:

```bash
# ~5 minutes per arm, 1 trial, mock planner — validates plumbing, not results
docker run --rm -v "$PWD:/work" -e FUZZHARNESS_PLANNER=mock llmfuzz:latest \
  python3 -m harness.afl_runner \
    --run-id smoke --target libpng --strategy heuristic --trial 0 \
    --duration 300 \
    --seeds targets/libpng/build/seeds \
    --out results/smoke \
    --edge-ids targets/libpng/build/fuzz/edge_ids.txt \
    -- targets/libpng/build/fuzz/libpng_fuzzer @@
```

Then the pilot matrix: 4 arms × 1 target × 3 trials × 1 hour ≈ **12 CPU-hours**,
plus roughly 60–200 planner calls for arm D.

Single runs are individually reproducible: the RNG seed is derived by blake2b
from `(target, strategy, trial)` — not Python's `hash()`, which is salted per
process and would silently destroy reproducibility.

---

## What is still to build

The harness, strategies, planner, target build and tests are done. Remaining
before the pilot can produce a paper figure:

1. **`experiments/runner.py`** — matrix over (target × strategy × trial), CPU
   pinning, one container per run, resumable.
2. **`harness/crash_triage.py`** — replay crashes under the ASAN build, parse the
   report, hash the top-N frames after filtering interceptors, dedup.
3. **`analysis/`** — `llvm-cov` replay of timestamped corpus snapshots against
   the `cov` build (FuzzBench-style ground truth, independent of AFL++'s
   hash-bucketed edge counts), then median + IQR coverage-over-time plots and the
   summary table including **coverage per compute-second**.

Measured but not yet charted: exec/s over time, edge coverage over time, unique
crashes with timestamps, time-to-first-crash, time-to-new-coverage, scheduler
wall/CPU time per arm, and planner latency/tokens/cost.

---

## Scaling up

libpng is a coverage-only pilot target — v1.6.37 has no seeded ground-truth bugs,
so crash counts there are opportunistic and **not** the metric. Bug-finding claims
need Magma targets (libtiff, libxml2), which carry documented seeded bugs and
report ground-truth reached/triggered signals. Adding one is a directory, a
`target.yaml` and a `build.sh` — the harness reads the metadata and hardcodes
nothing.

---

## Testing

```bash
python tests/test_core.py        # or: python -m pytest tests/ -q
```

28 tests over the logic that would otherwise fail silently: queue-name parsing
(all attribution flows through it), the rarity/yield/cost maths, edge-count
retraction on re-analysis, each arm's distinguishing property, controller
convergence under a skewed base distribution, and total validation of model
output including NaN, wrong types and garbage.

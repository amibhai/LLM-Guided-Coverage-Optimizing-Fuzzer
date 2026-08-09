"""The module AFL++ loads via ``AFL_PYTHON_MODULE``. This runs *inside* afl-fuzz.

AFL++ imports this module by name and binds module-level functions by
``PyObject_GetOptionalAttrString`` (src/afl-fuzz-python.c:255-337). We deliberately
define only three of them:

``init(seed)``            once at startup
``queue_get(filename)``   once per ``fuzz_one()`` -- our scheduling decision
``queue_new_entry(new, orig)``  when a queue entry is added
``deinit()``              at teardown

**We do not define ``fuzz()``.** That is load-bearing, not an omission. AFL++ guards
its entire custom mutator stage with ``if (el->afl_custom_fuzz)``
(src/afl-fuzz-one.c:1942), so leaving it undefined means the stage is skipped
outright and AFL++'s native mutators run untouched -- no Python call anywhere near
the per-execution path. Defining it would put CPython in the hot loop and collapse
throughput, and would also be a violation of the project's "wrap AFL++, don't
reimplement it" constraint.

Cost of what remains: ``queue_get`` fires once per ``fuzz_one()``, not once per
execution. A ``fuzz_one()`` typically runs hundreds to thousands of executions, so
at a few thousand execs/sec this is single-digit calls per second. Everything it
does is a dict lookup and a float compare; scoring happens on a ~1 Hz timer in
:meth:`_tick`, and all file and network I/O lives in other processes entirely.

Configuration arrives via environment variables because that is the only channel
AFL++ gives us -- it constructs the interpreter itself and passes nothing:

``FUZZHARNESS_CONFIG``   path to the JSON run config written by the runner
``FUZZHARNESS_STATE``    directory for plan.json, analysis output, logs
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from harness.controller import AcceptanceController, ControllerConfig  # noqa: E402
from harness.corpus_model import CorpusModel, CorpusView  # noqa: E402
from harness.metrics import SchedulerCostMeter, RunLogger  # noqa: E402
from harness.symbols import load_edge_function_map  # noqa: E402
from strategies.base import Decision, RunContext  # noqa: E402
from strategies.registry import build_strategy  # noqa: E402

#: Module-level singleton. AFL++ owns the interpreter, so there is nowhere else
#: to hang state.
_B: "Bridge | None" = None


class Bridge:
    """Glue between AFL++'s callbacks and our strategy/controller/corpus objects."""

    def __init__(self) -> None:
        cfg_path = os.environ.get("FUZZHARNESS_CONFIG")
        if not cfg_path or not os.path.exists(cfg_path):
            raise RuntimeError(
                "FUZZHARNESS_CONFIG must point at the run config JSON written by "
                "harness/afl_runner.py"
            )
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)

        self.cfg = cfg
        self.state_dir = os.environ.get("FUZZHARNESS_STATE") or cfg["state_dir"]
        os.makedirs(self.state_dir, exist_ok=True)

        self.t0 = time.monotonic()
        self.corpus = CorpusModel()
        self.view = CorpusView(self.corpus)
        self.log = RunLogger(self.state_dir, run_id=cfg["run_id"])
        self.meter = SchedulerCostMeter()

        strategy_cfg = cfg.get("strategy_config") or {}
        self.strategy = build_strategy(cfg["strategy"], strategy_cfg)

        edge_map = load_edge_function_map(cfg.get("edge_ids_file"))
        self.ctx = RunContext(
            run_id=cfg["run_id"],
            target=cfg["target"],
            strategy_name=cfg["strategy"],
            trial=int(cfg.get("trial", 0)),
            seed=int(cfg.get("seed", 0)),
            duration_s=float(cfg.get("duration_s", 0.0)),
            out_dir=cfg["out_dir"],
            state_dir=self.state_dir,
            edge_to_function=edge_map,
            config=cfg,
        )

        ctrl_cfg = ControllerConfig(**(cfg.get("controller") or {}))
        self.controller = AcceptanceController(ctrl_cfg, rng_seed=self.ctx.seed)

        self.queue_dir = os.path.join(cfg["out_dir"], cfg.get("fuzzer_name", "default"), "queue")
        self.analysis_path = os.path.join(self.state_dir, "analysis.jsonl")
        self._analysis_offset = 0
        self._last_tick = 0.0
        self._tick_interval = float(cfg.get("tick_interval_s", 1.0))
        self._explain_interval = float(cfg.get("explain_interval_s", 30.0))
        self._last_explain = 0.0
        self._decisions = 0

        self.strategy.on_start(self.ctx)
        self.log.manifest(
            {
                "run_id": self.ctx.run_id,
                "target": self.ctx.target,
                "strategy": self.strategy.manifest(),
                "controller": ctrl_cfg.__dict__,
                "edge_map_entries": len(edge_map),
                "grounded": bool(edge_map),
                "pid": os.getpid(),
            }
        )

    # ------------------------------------------------------------------ helpers

    def now(self) -> float:
        return time.monotonic() - self.t0

    def _ingest_analysis(self) -> None:
        """Pick up edge sets the sidecar analyser has appended since last read.

        Incremental tail read of an append-only JSONL file: cheap, and it cannot
        block on the analyser. If the analyser is behind, we simply have fewer
        analysed seeds this tick and strategy C uses its documented neutral prior.
        """
        try:
            size = os.path.getsize(self.analysis_path)
        except OSError:
            return
        if size <= self._analysis_offset:
            return
        try:
            with open(self.analysis_path, "r", encoding="utf-8") as fh:
                fh.seek(self._analysis_offset)
                lines = fh.readlines()
                self._analysis_offset = fh.tell()
        except OSError:
            return

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue  # torn final line; it will be re-read next tick
            sid = rec.get("seed_id")
            edges = rec.get("edges")
            if sid is None or edges is None:
                continue
            self.corpus.attach_analysis(
                int(sid), frozenset(int(e) for e in edges), float(rec.get("exec_us", 0.0))
            )
            seed = self.corpus.seeds.get(int(sid))
            if seed is not None:
                self.strategy.on_analysis(seed, self.view)

    def _tick(self, now_s: float) -> None:
        """Periodic, off-execution-path work: ingest, rescore, refresh, log."""
        self._ingest_analysis()
        self.strategy.on_tick(now_s, self.view)

        if self.controller.needs_refresh(now_s):
            priorities = self.strategy.priorities(self.view)
            self.controller.refresh(priorities, now_s)

        if now_s - self._last_explain >= self._explain_interval:
            self._last_explain = now_s
            self.log.explain(
                now_s,
                self.strategy.explain(),
                controller={
                    "offers": self.controller.stats.offers,
                    "accepts": self.controller.stats.accepts,
                    "accept_rate": self.controller.stats.accept_rate,
                    "refreshes": self.controller.stats.refreshes,
                },
                cost=self.meter.snapshot(),
                corpus=self.view.summary(),
            )

    # ---------------------------------------------------------- AFL++ callbacks

    def queue_get(self, filename: str) -> bool:
        with self.meter.measure("queue_get"):
            now_s = self.now()
            if now_s - self._last_tick >= self._tick_interval:
                self._last_tick = now_s
                self._tick(now_s)

            rec = self.corpus.by_path.get(filename)
            if rec is None:
                # First sighting (initial corpus, or a sync import we have not been
                # told about). Register it so it is schedulable immediately.
                rec = self.corpus.add_seed(filename, now_s)
                if rec is None:
                    return True  # not an AFL++ queue name; never block it
                self.controller.note_new_seed(rec.seed_id)
                self.strategy.on_new_seed(rec, self.view)

            accepted, prob = self.controller.decide(rec.seed_id)
            self.corpus.record_offer(rec.seed_id, accepted)
            self._decisions += 1

            decision = Decision(
                seed_id=rec.seed_id,
                accepted=accepted,
                priority=self.strategy.priority(rec, self.view),
                accept_prob=prob,
                t_s=now_s,
            )
            self.strategy.on_decision(decision, self.view)
            self.log.decision(decision)
            return accepted

    def queue_new_entry(self, filename_new: str, filename_orig: str | None) -> bool:
        with self.meter.measure("queue_new_entry"):
            now_s = self.now()
            rec = self.corpus.add_seed(filename_new, now_s)
            if rec is not None:
                self.controller.note_new_seed(rec.seed_id)
                self.strategy.on_new_seed(rec, self.view)
                self.log.new_seed(now_s, rec, filename_orig)
        # Return False: we did not modify the file's contents. Returning True
        # tells AFL++ the on-disk data changed and it should re-read it.
        return False

    def deinit(self) -> None:
        try:
            self.strategy.on_stop()
            self.log.explain(
                self.now(),
                self.strategy.explain(),
                controller={
                    "offers": self.controller.stats.offers,
                    "accepts": self.controller.stats.accepts,
                    "accept_rate": self.controller.stats.accept_rate,
                    "refreshes": self.controller.stats.refreshes,
                },
                cost=self.meter.snapshot(),
                corpus=self.view.summary(),
                final=True,
            )
        finally:
            self.log.close()


# --------------------------------------------------------------------------
# AFL++ entry points. AFL++ binds these by name; signatures are fixed by
# src/afl-fuzz-python.c and must not be changed.
#
# Each wrapper is defensive: an exception escaping into AFL++'s C code would
# abort a fuzzing run that may already be hours old. On error we log and fail
# open (accept the seed), which degrades to unbiased scheduling rather than
# losing the run.
# --------------------------------------------------------------------------


def init(seed: int) -> None:
    global _B
    try:
        _B = Bridge()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.write("[afl_bridge] init failed; scheduling disabled\n")
        sys.stderr.flush()
        _B = None


def queue_get(filename) -> bool:
    if _B is None:
        return True
    try:
        return _B.queue_get(_as_str(filename))
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return True


def queue_new_entry(filename_new_queue, filename_orig_queue) -> bool:
    if _B is None:
        return False
    try:
        return _B.queue_new_entry(
            _as_str(filename_new_queue), _as_str(filename_orig_queue)
        )
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return False


def deinit() -> None:
    if _B is None:
        return
    try:
        _B.deinit()
    except Exception:
        traceback.print_exc(file=sys.stderr)


def _as_str(value) -> str:
    """AFL++ hands us bytes for path arguments; normalise to str."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)

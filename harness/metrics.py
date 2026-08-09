"""Scheduler cost accounting and structured run logging.

Cost accounting is not incidental bookkeeping here -- it is the measurement the
research question turns on. "Does the LLM help *when compute cost is accounted
for*" is unanswerable unless the time spent deciding is measured as carefully as
the time spent fuzzing. So :class:`SchedulerCostMeter` records both:

``wall_ns``   elapsed real time -- what actually delayed AFL++
``cpu_ns``    process CPU time -- what actually consumed a core

The two differ in exactly the case that matters. A blocking LLM call burns wall
time while using almost no CPU; a heavy rescore burns both. Reporting only CPU
would flatter strategy D, reporting only wall would flatter it on an idle machine.
The analysis charges strategy D for LLM wall time *and* tokens, and the planner
daemon's own CPU is measured separately in its own process.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from harness.corpus_model import SeedRecord
from strategies.base import Decision


@dataclass
class CostBucket:
    calls: int = 0
    wall_ns: int = 0
    cpu_ns: int = 0

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "wall_ms": self.wall_ns / 1e6,
            "cpu_ms": self.cpu_ns / 1e6,
            "mean_wall_us": (self.wall_ns / self.calls / 1e3) if self.calls else 0.0,
        }


class SchedulerCostMeter:
    """Accumulates time spent inside our scheduling logic, by call site."""

    def __init__(self) -> None:
        self.buckets: dict[str, CostBucket] = {}

    @contextmanager
    def measure(self, label: str):
        b = self.buckets.get(label)
        if b is None:
            b = self.buckets[label] = CostBucket()
        w0 = time.perf_counter_ns()
        c0 = time.process_time_ns()
        try:
            yield
        finally:
            b.calls += 1
            b.wall_ns += time.perf_counter_ns() - w0
            b.cpu_ns += time.process_time_ns() - c0

    def snapshot(self) -> dict:
        total_wall = sum(b.wall_ns for b in self.buckets.values())
        total_cpu = sum(b.cpu_ns for b in self.buckets.values())
        return {
            "total_wall_ms": total_wall / 1e6,
            "total_cpu_ms": total_cpu / 1e6,
            "by_call": {k: v.as_dict() for k, v in self.buckets.items()},
        }


class RunLogger:
    """Append-only JSONL logs, one file per event kind.

    Separate files rather than one interleaved stream: the decision log is by far
    the highest-volume and is the one an analysis pass most often wants to skip, so
    keeping it apart means loading plans or coverage events stays cheap.

    Writes are line-buffered and flushed on the low-volume streams so a run killed
    by timeout still leaves a readable log.
    """

    def __init__(self, state_dir: str, run_id: str, log_decisions: bool = True) -> None:
        self.state_dir = state_dir
        self.run_id = run_id
        os.makedirs(state_dir, exist_ok=True)
        self._files: dict[str, object] = {}
        self.log_decisions = log_decisions

    def _fh(self, kind: str):
        fh = self._files.get(kind)
        if fh is None:
            path = os.path.join(self.state_dir, f"{kind}.jsonl")
            fh = self._files[kind] = open(path, "a", encoding="utf-8", buffering=1)
        return fh

    def _write(self, kind: str, obj: dict) -> None:
        try:
            self._fh(kind).write(json.dumps(obj, default=str) + "\n")
        except (OSError, TypeError, ValueError):
            # Logging must never take down a fuzzing run.
            pass

    def manifest(self, obj: dict) -> None:
        path = os.path.join(self.state_dir, "manifest.json")
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(obj, fh, indent=2, default=str)
        except OSError:
            pass

    def decision(self, d: Decision) -> None:
        if not self.log_decisions:
            return
        self._write(
            "decisions",
            {
                "t": round(d.t_s, 3),
                "seed": d.seed_id,
                "acc": int(d.accepted),
                "p": round(d.priority, 5),
                "ap": round(d.accept_prob, 5),
            },
        )

    def new_seed(self, t_s: float, rec: SeedRecord, parent_path: str | None) -> None:
        self._write(
            "seeds",
            {
                "t": round(t_s, 3),
                "seed": rec.seed_id,
                "parent": rec.parent_id,
                "depth": rec.depth,
                "size": rec.size,
                "new_cov": rec.name.new_cov,
                "op": rec.name.op,
                "afl_time_ms": rec.name.time_ms,
                "afl_execs": rec.name.execs,
                "imported": rec.name.is_imported,
            },
        )

    def explain(self, t_s: float, strategy: dict, **extra) -> None:
        self._write("explain", {"t": round(t_s, 3), "strategy": strategy, **extra})

    def close(self) -> None:
        for fh in self._files.values():
            try:
                fh.close()  # type: ignore[attr-defined]
            except OSError:
                pass
        self._files.clear()

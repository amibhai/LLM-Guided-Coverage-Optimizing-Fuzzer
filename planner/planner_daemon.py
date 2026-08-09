"""Planner sidecar: builds observations, calls the planner, publishes plan.json.

Runs as a **separate process** from afl-fuzz. That is the whole point:

* A slow model call delays only this process. AFL++ keeps fuzzing at full rate,
  and strategy D keeps scheduling with the plan it already has.
* The planner's CPU and wall time are attributable, because they belong to a
  process we can measure independently rather than being smeared into the
  fuzzer's own accounting.
* A crash here cannot take down a fuzzing run that may be hours old.

Triggering is deliberately *both* time- and event-based: every ``interval_s``
seconds, but only if at least ``min_new_edges`` new edges have appeared since the
last plan. A campaign that has plateaued does not need to be re-planned every
30 seconds at full token cost, and one that is discovering rapidly benefits from
being re-planned sooner. ``max_calls`` caps total spend per run so a long
experiment cannot silently run up a bill.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from harness.symbols import (  # noqa: E402
    function_coverage,
    load_edge_function_map,
    uncovered_functions,
)
from planner.planner_llm import build_planner  # noqa: E402
from planner.schema import Plan, PlanObservation  # noqa: E402


class PlannerDaemon:
    def __init__(self, config: dict) -> None:
        self.cfg = config
        self.state_dir = config["state_dir"]
        self.run_id = config["run_id"]
        self.target = config["target"]

        pcfg = dict(config.get("planner") or {})
        self.interval_s = float(pcfg.get("interval_s", 60.0))
        self.min_new_edges = int(pcfg.get("min_new_edges", 5))
        self.max_calls = int(pcfg.get("max_calls", 200))

        self.planner = build_planner(pcfg)
        self.edge_map = load_edge_function_map(config.get("edge_ids_file"))

        self.analysis_path = os.path.join(self.state_dir, "analysis.jsonl")
        self.plan_path = os.path.join(self.state_dir, "plan.json")
        self.log_path = os.path.join(self.state_dir, "planner_calls.jsonl")

        self._analysis_offset = 0
        self.covered_edges: set[int] = set()
        #: edge -> number of corpus seeds covering it, for the frontier ranking
        self.edge_counts: dict[int, int] = {}
        self.coverage_trend: list[list[float]] = []
        self.seeds_seen = 0

        self.seq = 0
        self.calls = 0
        self.total_cost_usd = 0.0
        self.edges_at_last_plan = 0
        self.last_plan: Plan | None = None
        self.t0 = time.monotonic()

    # ------------------------------------------------------------------ ingest

    def ingest(self) -> None:
        """Tail the analyser's output. Same incremental read as the bridge."""
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
                continue
            edges = rec.get("edges") or []
            self.seeds_seen += 1
            for e in edges:
                e = int(e)
                self.covered_edges.add(e)
                self.edge_counts[e] = self.edge_counts.get(e, 0) + 1

        self.coverage_trend.append(
            [round(time.monotonic() - self.t0, 1), len(self.covered_edges)]
        )

    # -------------------------------------------------------------- observation

    def build_observation(self) -> PlanObservation:
        covered_per_fn = function_coverage(self.covered_edges, self.edge_map)

        # Total instrumented edges per function, from the static map. The
        # difference against covered_per_fn is the frontier -- functions we have
        # entered but not explored. Both halves come from real data: the static
        # map from AFL_LLVM_DOCUMENT_IDS, the covered set from replayed seeds.
        total_per_fn: dict[str, int] = {}
        for fn in self.edge_map.values():
            total_per_fn[fn] = total_per_fn.get(fn, 0) + 1

        partial = []
        for fn, covered in covered_per_fn.items():
            total = total_per_fn.get(fn, covered)
            uncovered = total - covered
            if uncovered > 0:
                partial.append(
                    {
                        "function": fn,
                        "covered": covered,
                        "total": total,
                        "uncovered": uncovered,
                        "fraction": round(covered / total, 3) if total else 0.0,
                    }
                )
        partial.sort(key=lambda f: f["uncovered"], reverse=True)

        singleton = sum(1 for c in self.edge_counts.values() if c == 1)

        return PlanObservation(
            run_id=self.run_id,
            target=self.target,
            elapsed_s=time.monotonic() - self.t0,
            corpus={
                "analysed_seeds": self.seeds_seen,
                "total_edges": len(self.covered_edges),
                "singleton_edges": singleton,
                "instrumented_functions": len(total_per_fn),
                "functions_reached": len(covered_per_fn),
            },
            uncovered_functions=uncovered_functions(self.covered_edges, self.edge_map),
            covered_functions=covered_per_fn,
            partial_functions=partial,
            coverage_trend=self.coverage_trend,
            crashes=self.read_crashes(),
            previous_plan=self.last_plan.as_dict() if self.last_plan else None,
            edges_since_last_plan=len(self.covered_edges) - self.edges_at_last_plan,
        )

    def read_crashes(self) -> list[dict]:
        """Deduplicated crash summaries, if triage has produced any yet."""
        path = os.path.join(self.state_dir, "crashes.json")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return []
        return data if isinstance(data, list) else []

    # ------------------------------------------------------------------ publish

    def publish(self, plan: Plan) -> None:
        """Write plan.json atomically.

        Temp file + ``os.replace``: the reader is a live fuzzing run polling this
        path, and it must never observe a half-written file.
        """
        payload = plan.as_dict()
        tmp = self.plan_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp, self.plan_path)
        except OSError:
            return

        try:
            with open(self.log_path, "a", encoding="utf-8", buffering=1) as fh:
                fh.write(json.dumps({"t": round(time.monotonic() - self.t0, 2), **payload}) + "\n")
        except OSError:
            pass

    # ---------------------------------------------------------------- main loop

    def should_plan(self, now: float, last_plan_at: float) -> bool:
        if self.calls >= self.max_calls:
            return False
        if now - last_plan_at < self.interval_s:
            return False
        new_edges = len(self.covered_edges) - self.edges_at_last_plan
        # First plan always fires once there is anything at all to look at;
        # after that, require real progress so a plateaued run stops paying.
        if self.calls == 0:
            return len(self.covered_edges) > 0
        return new_edges >= self.min_new_edges

    def run(self, duration_s: float, poll_s: float = 5.0) -> None:
        deadline = time.monotonic() + duration_s if duration_s > 0 else float("inf")
        last_plan_at = -1e9

        while time.monotonic() < deadline:
            self.ingest()
            now = time.monotonic()
            if self.should_plan(now, last_plan_at):
                last_plan_at = now
                self.seq += 1
                self.calls += 1
                obs = self.build_observation()

                cpu0 = time.process_time()
                plan = self.planner.plan(obs, self.seq)
                cpu_s = time.process_time() - cpu0

                self.edges_at_last_plan = len(self.covered_edges)
                self.total_cost_usd += plan.cost_usd
                if not plan.is_fallback:
                    self.last_plan = plan
                self.publish(plan)

                print(
                    f"[planner] seq={plan.seq} fallback={plan.is_fallback} "
                    f"regions={len(plan.region_priorities)} "
                    f"latency={plan.latency_s:.2f}s cpu={cpu_s:.3f}s "
                    f"cost=${plan.cost_usd:.4f} total=${self.total_cost_usd:.4f}"
                    + (f" err={plan.error}" if plan.error else ""),
                    flush=True,
                )
            time.sleep(poll_s)

        # Final accounting, read by the analysis pass to charge strategy D.
        summary = {
            "calls": self.calls,
            "total_cost_usd": self.total_cost_usd,
            "planner": getattr(self.planner, "name", "unknown"),
            "interval_s": self.interval_s,
            "min_new_edges": self.min_new_edges,
        }
        try:
            with open(
                os.path.join(self.state_dir, "planner_summary.json"), "w", encoding="utf-8"
            ) as fh:
                json.dump(summary, fh, indent=2)
        except OSError:
            pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="run_config.json written by afl_runner")
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--poll", type=float, default=5.0)
    args = ap.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    PlannerDaemon(cfg).run(duration_s=args.duration, poll_s=args.poll)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

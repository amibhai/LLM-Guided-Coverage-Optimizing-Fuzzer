"""Sidecar corpus analyser: extracts per-seed edge sets with ``afl-showmap``.

Runs as its own process alongside afl-fuzz. It watches the queue directory, replays
each new seed under ``afl-showmap``, and appends the resulting edge set to
``analysis.jsonl``, which the in-process bridge tails.

Why a separate process
----------------------
``afl-showmap`` forks and executes the target -- on the order of milliseconds per
seed, occasionally much worse on a slow input. Doing that inside ``queue_get``
would put a fork-exec on AFL++'s scheduling path and wreck the throughput numbers
the study is trying to measure. Out of process, a slow replay only delays *our*
knowledge of a seed; AFL++ keeps fuzzing at full speed and strategy C falls back to
its documented neutral prior for seeds not yet analysed.

Batching: ``afl-showmap -i <dir> -o <dir>`` processes a whole directory in one
invocation, amortising process startup across every pending seed. We copy pending
seeds into a staging directory rather than pointing showmap at the live queue,
because the queue mutates underneath us while AFL++ runs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from harness.queue_names import parse_queue_name  # noqa: E402


def parse_showmap_file(path: str) -> tuple[frozenset[int], int]:
    """Read an ``afl-showmap`` output file.

    Format is one ``edge_id:hitcount`` per line. Returns the edge set and the sum
    of hit counts (a cheap proxy for how much work the input did, logged for
    analysis but not used in scoring).
    """
    edges: set[int] = set()
    total_hits = 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                key, _, val = line.partition(":")
                try:
                    edges.add(int(key))
                    total_hits += int(val) if val else 0
                except ValueError:
                    continue
    except OSError:
        return frozenset(), 0
    return frozenset(edges), total_hits


class CorpusAnalyzer:
    def __init__(
        self,
        queue_dir: str,
        out_path: str,
        target_cmd: list[str],
        showmap: str = "afl-showmap",
        timeout_ms: int = 1000,
        mem_limit: str = "none",
        staging_dir: str | None = None,
        batch_size: int = 64,
    ) -> None:
        self.queue_dir = queue_dir
        self.out_path = out_path
        self.target_cmd = target_cmd
        self.showmap = showmap
        self.timeout_ms = timeout_ms
        self.mem_limit = mem_limit
        self.batch_size = batch_size
        self.staging = staging_dir or os.path.join(
            os.path.dirname(out_path), "_showmap_stage"
        )
        self.seen: set[str] = set()

    # ------------------------------------------------------------------ scanning

    def pending(self) -> list[str]:
        try:
            names = os.listdir(self.queue_dir)
        except OSError:
            return []
        out = []
        for n in names:
            if n in self.seen or not n.startswith("id:"):
                continue
            out.append(n)
        return sorted(out)[: self.batch_size]

    # ------------------------------------------------------------------ analysis

    def analyse_batch(self, names: list[str]) -> list[dict]:
        if not names:
            return []

        stage_in = os.path.join(self.staging, "in")
        stage_out = os.path.join(self.staging, "out")
        shutil.rmtree(self.staging, ignore_errors=True)
        os.makedirs(stage_in, exist_ok=True)
        os.makedirs(stage_out, exist_ok=True)

        staged: dict[str, str] = {}
        for n in names:
            src = os.path.join(self.queue_dir, n)
            # Flatten the AFL++ name: showmap echoes input filenames into its
            # output dir, and ':' / ',' in them are awkward to round-trip.
            safe = f"s{parse_queue_name(n).seed_id:08d}" if parse_queue_name(n) else None
            if safe is None:
                self.seen.add(n)
                continue
            try:
                shutil.copyfile(src, os.path.join(stage_in, safe))
            except OSError:
                continue  # AFL++ may have replaced the file mid-copy; retry later
            staged[safe] = n

        if not staged:
            return []

        cmd = [
            self.showmap,
            "-i", stage_in,
            "-o", stage_out,
            "-t", str(self.timeout_ms),
            "-m", self.mem_limit,
            "-Z",   # quiet / no UI
            "--",
        ] + self.target_cmd

        t0 = time.perf_counter()
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=max(30, self.timeout_ms * len(staged) / 1000 * 4),
                check=False,
                env={**os.environ, "AFL_QUIET": "1", "AFL_NO_UI": "1"},
            )
        except (subprocess.TimeoutExpired, OSError):
            # Mark them seen anyway: a batch that cannot be replayed will not
            # become replayable, and retrying forever would starve new seeds.
            for n in staged.values():
                self.seen.add(n)
            return []
        batch_wall_s = time.perf_counter() - t0

        # showmap gives us one aggregate wall time for the batch, not per-seed
        # timings. Dividing evenly is an approximation, and it is labelled as one:
        # the field is named exec_us_approx in the output record.
        per_seed_us = (batch_wall_s / max(len(staged), 1)) * 1e6

        results: list[dict] = []
        for safe, original in staged.items():
            self.seen.add(original)
            qn = parse_queue_name(original)
            if qn is None:
                continue
            map_path = os.path.join(stage_out, safe)
            if not os.path.exists(map_path):
                # Crashing or timing-out seed: showmap wrote nothing. Record it
                # with an empty edge set so the bridge stops waiting on it.
                results.append(
                    {
                        "seed_id": qn.seed_id,
                        "edges": [],
                        "exec_us": per_seed_us,
                        "status": "no_map",
                    }
                )
                continue
            edges, hits = parse_showmap_file(map_path)
            results.append(
                {
                    "seed_id": qn.seed_id,
                    "edges": sorted(edges),
                    "exec_us": per_seed_us,
                    "exec_us_approx": True,
                    "total_hits": hits,
                    "status": "ok",
                }
            )
        return results

    # ---------------------------------------------------------------- main loop

    def run(self, duration_s: float, poll_s: float = 1.0) -> None:
        deadline = time.monotonic() + duration_s if duration_s > 0 else float("inf")
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        with open(self.out_path, "a", encoding="utf-8", buffering=1) as out:
            while time.monotonic() < deadline:
                names = self.pending()
                if not names:
                    time.sleep(poll_s)
                    continue
                for rec in self.analyse_batch(names):
                    out.write(json.dumps(rec) + "\n")
        shutil.rmtree(self.staging, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queue-dir", required=True)
    ap.add_argument("--out", required=True, help="path to analysis.jsonl")
    ap.add_argument("--showmap", default="afl-showmap")
    ap.add_argument("--timeout-ms", type=int, default=1000)
    ap.add_argument("--mem-limit", default="none")
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--poll", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("target", nargs=argparse.REMAINDER, help="-- target argv")
    args = ap.parse_args(argv)

    target = args.target
    if target and target[0] == "--":
        target = target[1:]
    if not target:
        ap.error("target command required after --")

    CorpusAnalyzer(
        queue_dir=args.queue_dir,
        out_path=args.out,
        target_cmd=target,
        showmap=args.showmap,
        timeout_ms=args.timeout_ms,
        mem_limit=args.mem_limit,
        batch_size=args.batch_size,
    ).run(duration_s=args.duration, poll_s=args.poll)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

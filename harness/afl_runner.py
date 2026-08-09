"""Launches one fuzzing run: afl-fuzz + sidecar analyser + (for D) planner daemon.

One run = one (target, strategy, trial) cell of the experiment matrix. This module
is what the experiment runner invokes inside the container; it is deliberately
runnable standalone so a single arm can be debugged without the matrix around it.

Process layout::

    afl_runner (this)
      |-- afl-fuzz  ......... embeds CPython, imports harness.afl_bridge
      |                        -> queue_get / queue_new_entry decisions
      |-- analyzer  ......... afl-showmap replays -> analysis.jsonl
      `-- planner_daemon .... strategy D only; LLM calls -> plan.json

Keeping the analyser and planner out of the afl-fuzz process is what allows the
throughput comparison to mean anything: neither one can block ``fuzz_one()``.

Fair-comparison invariants, enforced here rather than left to convention:

* Every arm gets the same target binary, same seed corpus, same dictionary, same
  wall-clock budget (``-V``), and the same CPU binding policy.
* ``AFL_PYTHON_MODULE`` is set **only** when the strategy declares
  ``uses_queue_filter``. Strategy B therefore runs as unmodified AFL++, with no
  Python in the process at all.
* The RNG seed is derived from (strategy, trial) so trials are reproducible and
  distinct, and so arm A's randomness is not accidentally shared with arm C's.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from strategies.registry import build_strategy  # noqa: E402

DEFAULT_FUZZER_NAME = "default"


class RunSpec:
    """Everything needed to launch one arm. Serialised to the config JSON."""

    def __init__(
        self,
        run_id: str,
        target: str,
        strategy: str,
        trial: int,
        duration_s: float,
        target_cmd: list[str],
        seeds_dir: str,
        out_dir: str,
        dict_path: str | None = None,
        edge_ids_file: str | None = None,
        timeout_ms: int = 1000,
        mem_limit: str = "none",
        strategy_config: dict | None = None,
        controller: dict | None = None,
        planner: dict | None = None,
        afl_fuzz: str = "afl-fuzz",
        showmap: str = "afl-showmap",
    ) -> None:
        self.run_id = run_id
        self.target = target
        self.strategy = strategy
        self.trial = trial
        self.duration_s = duration_s
        self.target_cmd = target_cmd
        self.seeds_dir = seeds_dir
        self.out_dir = out_dir
        self.dict_path = dict_path
        self.edge_ids_file = edge_ids_file
        self.timeout_ms = timeout_ms
        self.mem_limit = mem_limit
        self.strategy_config = strategy_config or {}
        self.controller = controller or {}
        self.planner = planner or {}
        self.afl_fuzz = afl_fuzz
        self.showmap = showmap
        self.state_dir = os.path.join(out_dir, "harness")

    @property
    def rng_seed(self) -> int:
        """Deterministic per (target, strategy, trial), so runs are reproducible.

        Uses blake2b rather than ``hash()``: Python salts string hashing per
        process unless PYTHONHASHSEED is pinned, so ``hash()`` would hand the same
        trial a different seed on every invocation and quietly destroy the
        reproducibility this whole repo is built to provide.
        """
        key = f"{self.target}|{self.strategy}|{self.trial}".encode()
        digest = hashlib.blake2b(key, digest_size=8).digest()
        return (int.from_bytes(digest, "big") % (2**31 - 1)) or 1

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "target": self.target,
            "strategy": self.strategy,
            "trial": self.trial,
            "seed": self.rng_seed,
            "duration_s": self.duration_s,
            "target_cmd": self.target_cmd,
            "seeds_dir": self.seeds_dir,
            "out_dir": self.out_dir,
            "state_dir": self.state_dir,
            "fuzzer_name": DEFAULT_FUZZER_NAME,
            "dict_path": self.dict_path,
            "edge_ids_file": self.edge_ids_file,
            "timeout_ms": self.timeout_ms,
            "mem_limit": self.mem_limit,
            "strategy_config": self.strategy_config,
            "controller": self.controller,
            "planner": self.planner,
        }


def build_afl_command(spec: RunSpec, extra_args: tuple[str, ...]) -> list[str]:
    cmd = [
        spec.afl_fuzz,
        "-i", spec.seeds_dir,
        "-o", spec.out_dir,
        "-t", str(spec.timeout_ms),
        "-m", spec.mem_limit,
        # -V bounds the run by wall clock, which is the budget the study compares
        # arms on. Using an exec-count bound instead would hide exactly the
        # throughput differences we are trying to measure.
        "-V", str(int(spec.duration_s)),
    ]
    if spec.dict_path and os.path.exists(spec.dict_path):
        cmd += ["-x", spec.dict_path]
    cmd += list(extra_args)
    cmd += ["--"] + spec.target_cmd
    return cmd


def build_afl_env(spec: RunSpec, use_python_hook: bool) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "AFL_NO_UI": "1",             # we parse plot_data/fuzzer_stats, not the TUI
            "AFL_SKIP_CPUFREQ": "1",      # containers cannot set the governor
            "AFL_NO_AFFINITY": "1",       # the experiment runner does the pinning
            "AFL_AUTORESUME": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    if use_python_hook:
        # AFL++ imports by module name and needs the repo root importable.
        env["AFL_PYTHON_MODULE"] = "harness.afl_bridge"
        env["PYTHONPATH"] = _ROOT + os.pathsep + env.get("PYTHONPATH", "")
        env["FUZZHARNESS_CONFIG"] = os.path.join(spec.state_dir, "run_config.json")
        env["FUZZHARNESS_STATE"] = spec.state_dir
    else:
        # Belt and braces: make sure an inherited value cannot silently install
        # our hook into the supposedly-native baseline.
        env.pop("AFL_PYTHON_MODULE", None)
    return env


class RunLauncher:
    def __init__(self, spec: RunSpec) -> None:
        self.spec = spec
        self.procs: dict[str, subprocess.Popen] = {}

    def prepare(self) -> tuple[str, dict[str, str], list[str]]:
        spec = self.spec
        os.makedirs(spec.state_dir, exist_ok=True)

        strategy = build_strategy(spec.strategy, spec.strategy_config)
        use_hook = strategy.uses_queue_filter

        cfg_path = os.path.join(spec.state_dir, "run_config.json")
        with open(cfg_path, "w", encoding="utf-8") as fh:
            json.dump(spec.as_dict(), fh, indent=2)

        cmd = build_afl_command(spec, strategy.afl_extra_args)
        env = build_afl_env(spec, use_hook)

        with open(os.path.join(spec.state_dir, "launch.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "argv": cmd,
                    "python_hook": use_hook,
                    "strategy": strategy.manifest(),
                    "afl_env": {
                        k: v for k, v in env.items() if k.startswith(("AFL_", "FUZZHARNESS_"))
                    },
                    "started_at": time.time(),
                },
                fh,
                indent=2,
            )
        return cfg_path, env, cmd

    def start_sidecars(self) -> None:
        spec = self.spec
        queue_dir = os.path.join(spec.out_dir, DEFAULT_FUZZER_NAME, "queue")

        analyzer_cmd = [
            sys.executable, "-m", "harness.analyzer",
            "--queue-dir", queue_dir,
            "--out", os.path.join(spec.state_dir, "analysis.jsonl"),
            "--showmap", spec.showmap,
            "--timeout-ms", str(spec.timeout_ms),
            "--mem-limit", spec.mem_limit,
            # Outlive afl-fuzz slightly so the last seeds still get analysed.
            "--duration", str(spec.duration_s + 30),
            "--",
        ] + spec.target_cmd
        self.procs["analyzer"] = subprocess.Popen(
            analyzer_cmd,
            cwd=_ROOT,
            env={**os.environ, "PYTHONPATH": _ROOT},
            stdout=open(os.path.join(spec.state_dir, "analyzer.log"), "w"),
            stderr=subprocess.STDOUT,
        )

        if spec.strategy == "llm_guided":
            planner_cmd = [
                sys.executable, "-m", "planner.planner_daemon",
                "--config", os.path.join(spec.state_dir, "run_config.json"),
                "--duration", str(spec.duration_s),
            ]
            self.procs["planner"] = subprocess.Popen(
                planner_cmd,
                cwd=_ROOT,
                env={**os.environ, "PYTHONPATH": _ROOT},
                stdout=open(os.path.join(spec.state_dir, "planner.log"), "w"),
                stderr=subprocess.STDOUT,
            )

    def run(self) -> int:
        spec = self.spec
        _, env, cmd = self.prepare()

        # afl-fuzz creates out_dir itself and refuses to reuse a dirty one.
        if os.path.exists(spec.out_dir) and os.listdir(spec.out_dir):
            shutil.rmtree(spec.out_dir, ignore_errors=True)
        os.makedirs(spec.state_dir, exist_ok=True)

        self.start_sidecars()

        afl_log = open(os.path.join(spec.state_dir, "afl.log"), "w")
        t0 = time.time()
        proc = subprocess.Popen(
            cmd, cwd=_ROOT, env=env, stdout=afl_log, stderr=subprocess.STDOUT
        )
        self.procs["afl"] = proc

        try:
            # Generous grace beyond -V; afl-fuzz exits on its own timer.
            rc = proc.wait(timeout=spec.duration_s + 120)
        except subprocess.TimeoutExpired:
            proc.send_signal(signal.SIGINT)
            try:
                rc = proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = -9
        finally:
            elapsed = time.time() - t0
            self._stop_sidecars()
            afl_log.close()
            with open(os.path.join(spec.state_dir, "result.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {"return_code": rc, "elapsed_s": elapsed, "finished_at": time.time()},
                    fh,
                    indent=2,
                )
        return rc

    def _stop_sidecars(self) -> None:
        for name, p in self.procs.items():
            if name == "afl" or p.poll() is not None:
                continue
            p.terminate()
        deadline = time.time() + 45
        for name, p in self.procs.items():
            if name == "afl":
                continue
            try:
                p.wait(timeout=max(1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                p.kill()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run one (target, strategy, trial) cell.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--duration", type=float, required=True, help="seconds")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dict", dest="dict_path", default=None)
    ap.add_argument("--edge-ids", dest="edge_ids_file", default=None)
    ap.add_argument("--timeout-ms", type=int, default=1000)
    ap.add_argument("--mem-limit", default="none")
    ap.add_argument("--strategy-config", default=None, help="JSON string or @file")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- target argv")
    args = ap.parse_args(argv)

    target_cmd = args.cmd
    if target_cmd and target_cmd[0] == "--":
        target_cmd = target_cmd[1:]
    if not target_cmd:
        ap.error("target command required after --")

    scfg: dict = {}
    if args.strategy_config:
        raw = args.strategy_config
        if raw.startswith("@"):
            with open(raw[1:], "r", encoding="utf-8") as fh:
                scfg = json.load(fh)
        else:
            scfg = json.loads(raw)

    spec = RunSpec(
        run_id=args.run_id,
        target=args.target,
        strategy=args.strategy,
        trial=args.trial,
        duration_s=args.duration,
        target_cmd=target_cmd,
        seeds_dir=args.seeds,
        out_dir=args.out,
        dict_path=args.dict_path,
        edge_ids_file=args.edge_ids_file,
        timeout_ms=args.timeout_ms,
        mem_limit=args.mem_limit,
        strategy_config=scfg,
    )
    return RunLauncher(spec).run()


if __name__ == "__main__":
    raise SystemExit(main())

"""Edge id -> function name mapping, the grounding for strategy D.

AFL++ writes this file at *compile* time when ``AFL_LLVM_DOCUMENT_IDS=<path>`` is
set and the target is built with ``afl-clang-lto``. From the AFL++ docs:

    "AFL_LLVM_DOCUMENT_IDS=file will document to a file which edge ID was given
     to which function."

This is why the build script uses LTO mode. Without it, edge ids are assigned by a
random hash at instrumentation time and carry no stable relationship to source
locations -- there would be no honest way to tell the planner which *functions* are
uncovered, and strategy D's "prioritise these code regions" output could not be
mapped back onto seeds. Strategy D detects the empty map and degrades to strategy C
rather than guessing.

The file's exact column layout has shifted between AFL++ releases, so the parser
below is intentionally permissive: it looks for an integer and a symbol on each
line rather than assuming fixed positions.
"""

from __future__ import annotations

import os
import re

_LINE_RE = re.compile(
    r"^\s*(?:edge\s*)?(?P<id>\d+)\s*[:=,\t ]\s*(?P<sym>[^\s,]+)", re.IGNORECASE
)


def load_edge_function_map(path: str | None) -> dict[int, str]:
    """Parse an AFL_LLVM_DOCUMENT_IDS file into ``{edge_id: function_name}``.

    Returns an empty dict when the file is missing or unparseable -- callers treat
    that as "no grounding available" rather than an error, because a target built
    without LTO is a legitimate (if less capable) configuration.
    """
    if not path or not os.path.exists(path):
        return {}

    out: dict[int, str] = {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                m = _LINE_RE.match(line)
                if not m:
                    continue
                try:
                    edge_id = int(m.group("id"))
                except ValueError:
                    continue
                sym = m.group("sym").strip().strip("\"'")
                if sym:
                    out[edge_id] = sym
    except OSError:
        return {}
    return out


def function_coverage(
    covered_edges, edge_to_function: dict[int, str]
) -> dict[str, int]:
    """Count covered edges per function -- the 'what has been reached' half of the
    planner prompt."""
    out: dict[str, int] = {}
    for e in covered_edges:
        fn = edge_to_function.get(e)
        if fn:
            out[fn] = out.get(fn, 0) + 1
    return out


def uncovered_functions(
    covered_edges, edge_to_function: dict[int, str]
) -> list[str]:
    """Functions with instrumented edges, none of which have been covered.

    This is the 'what has *not* been reached' half of the planner prompt, and it is
    the single most useful thing we can tell the model: it is derived from the
    static instrumentation map intersected with real observed coverage, so it is a
    fact about the run rather than a guess about the program.
    """
    covered = set(covered_edges)
    seen: dict[str, bool] = {}
    for edge, fn in edge_to_function.items():
        hit = edge in covered
        seen[fn] = seen.get(fn, False) or hit
    return sorted(fn for fn, hit in seen.items() if not hit)

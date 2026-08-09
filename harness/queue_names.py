"""Parser for AFL++ queue/crash filenames.

AFL++ builds these names in ``describe_op()`` (src/afl-fuzz-bitmap.c). The format is
a comma-separated key:value list, e.g.::

    id:000123,src:000045,time:98765,execs:1234567,op:havoc,rep:8,+cov
    id:000007,src:000001+000003,time:4210,execs:9912,op:splice,rep:4
    id:000002,sync:strategy_c,src:000019,time:5000,execs:100

Every field below is written by AFL++ itself -- nothing here is inferred or
synthesised. Two fields matter most to us:

``src``
    The *parent* queue entry (or two parents, joined by ``+``, for splice).
    This gives exact parent->child attribution for the yield term in strategy C,
    straight from the fuzzer rather than from a heuristic guess.

``+cov``
    Appended only when ``new_bits == 2`` in ``save_if_interesting()``, i.e. the
    input hit an edge that was *never* covered before -- not merely a new hitcount
    bucket on a known edge. This is the ground-truth "new coverage" event.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# AFL++ zero-pads ids to 6 digits, but don't rely on width.
_ID_RE = re.compile(r"^\d+$")


@dataclass(frozen=True)
class QueueName:
    """Structured view of one AFL++ queue entry filename."""

    raw: str
    seed_id: int
    parents: tuple[int, ...] = ()
    #: ms since fuzz start, as recorded by AFL++ (not our own clock)
    time_ms: int | None = None
    #: total execs performed by AFL++ when this entry was saved
    execs: int | None = None
    #: the mutation stage that produced it ("havoc", "splice", "int8", ...)
    op: str | None = None
    #: havoc stacking repeat count, when present
    rep: int | None = None
    #: True when AFL++ tagged the entry ",+cov" => a brand-new edge
    new_cov: bool = False
    #: set for entries imported from another instance via -M/-S sync
    sync_from: str | None = None
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def is_imported(self) -> bool:
        return self.sync_from is not None

    @property
    def parent_id(self) -> int | None:
        """Primary parent, or None for initial-corpus entries."""
        return self.parents[0] if self.parents else None


def _parse_ids(value: str) -> tuple[int, ...]:
    out: list[int] = []
    for part in value.split("+"):
        part = part.strip()
        if _ID_RE.match(part):
            out.append(int(part))
    return tuple(out)


def parse_queue_name(name: str) -> QueueName | None:
    """Parse a queue filename. Returns None if it isn't an AFL++ entry.

    Accepts a bare basename or a full path. Unknown ``key:value`` pairs are kept
    in ``extra`` rather than dropped, so a future AFL++ version that adds a field
    degrades to "we saw it but ignored it" instead of a parse failure.
    """
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    if not base.startswith("id:"):
        return None

    seed_id: int | None = None
    parents: tuple[int, ...] = ()
    time_ms: int | None = None
    execs: int | None = None
    op: str | None = None
    rep: int | None = None
    new_cov = False
    sync_from: str | None = None
    extra: dict[str, str] = {}

    for token in base.split(","):
        token = token.strip()
        if not token:
            continue
        if token == "+cov":
            new_cov = True
            continue
        key, sep, value = token.partition(":")
        if not sep:
            extra[token] = ""
            continue
        if key == "id":
            seed_id = int(value) if _ID_RE.match(value) else None
        elif key == "src":
            parents = _parse_ids(value)
        elif key == "time":
            time_ms = int(value) if _ID_RE.match(value) else None
        elif key == "execs":
            execs = int(value) if _ID_RE.match(value) else None
        elif key == "op":
            op = value
        elif key == "rep":
            rep = int(value) if _ID_RE.match(value) else None
        elif key == "sync":
            sync_from = value
        else:
            extra[key] = value

    if seed_id is None:
        return None

    return QueueName(
        raw=base,
        seed_id=seed_id,
        parents=parents,
        time_ms=time_ms,
        execs=execs,
        op=op,
        rep=rep,
        new_cov=new_cov,
        sync_from=sync_from,
        extra=extra,
    )

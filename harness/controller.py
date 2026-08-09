"""Turns a strategy's desired priorities into ``queue_get`` accept/reject decisions.

Why this exists
---------------
A strategy says "seed 42 deserves 3x the budget of seed 7". AFL++ instead hands us
seeds drawn from *its own* alias-table distribution and asks yes/no. Those two
things only line up if something corrects for the base distribution -- otherwise a
strategy that asks for a uniform schedule would still inherit all of AFL++'s
weighting (exec time, bitmap size, ``n_fuzz``, favored bonus; see
create_alias_table in src/afl-fuzz-queue.c).

We cannot read AFL++'s internal weights from Python. But we do not need to: every
call to ``queue_get`` *is* a sample from that distribution. So we estimate the base
empirically from offer counts and run a closed loop:

    accept_prob(s)  proportional to  desired_share(s) / observed_offer_share(s)

then clamp into ``[floor, 1.0]`` and rescale so the most-wanted seed sits at 1.0
(rejecting is not free -- each rejection burns a ``fuzz_one`` entry -- so we keep
the overall acceptance rate as high as the shape allows).

This is standard rejection sampling with an online estimate of the proposal
density. It converges to the target distribution as long as AFL++ offers every
seed with non-zero probability, which its alias table guarantees.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class ControllerConfig:
    #: Never let acceptance fall below this, so a starved seed can recover once
    #: the strategy's opinion of it changes. Also prevents livelock when the
    #: strategy wants a seed AFL++ rarely offers.
    accept_floor: float = 0.02
    #: Pseudo-count for the offer-share estimate. Higher = slower, steadier
    #: correction; lower = twitchier early on when counts are tiny.
    offer_prior: float = 5.0
    #: Priorities are recomputed at most this often (seconds). Between refreshes
    #: we reuse cached accept probabilities, keeping queue_get O(1).
    refresh_interval_s: float = 1.0
    #: Cap on the correction factor, so one rarely-offered seed cannot pin the
    #: whole schedule to itself.
    max_correction: float = 20.0


@dataclass
class ControllerStats:
    offers: int = 0
    accepts: int = 0
    refreshes: int = 0
    last_refresh_s: float = 0.0
    #: seed_id -> accept probability at last refresh
    accept_probs: dict[int, float] = field(default_factory=dict)

    @property
    def accept_rate(self) -> float:
        return self.accepts / self.offers if self.offers else 0.0


class AcceptanceController:
    """Maps priorities -> acceptance probabilities, with online base correction."""

    def __init__(self, config: ControllerConfig | None = None, rng_seed: int = 0) -> None:
        self.cfg = config or ControllerConfig()
        self.rng = random.Random(rng_seed)
        self.stats = ControllerStats()
        self._offer_counts: dict[int, int] = {}
        self._accept_prob: dict[int, float] = {}
        self._dirty = True

    def note_new_seed(self, seed_id: int) -> None:
        """A new seed exists; give it a provisional accept probability of 1.0.

        New arrivals are optimistically accepted until the next refresh so that a
        freshly discovered seed is never blocked for a full refresh interval.
        """
        self._accept_prob.setdefault(seed_id, 1.0)
        self._dirty = True

    def refresh(self, priorities: dict[int, float], now_s: float) -> None:
        """Recompute acceptance probabilities from the strategy's priorities."""
        cfg = self.cfg
        self.stats.refreshes += 1
        self.stats.last_refresh_s = now_s

        if not priorities:
            self._accept_prob = {}
            return

        total_p = sum(max(p, 0.0) for p in priorities.values())
        if total_p <= 0.0:
            # Strategy wants nothing; fall back to uniform rather than deadlock.
            self._accept_prob = {sid: 1.0 for sid in priorities}
            self.stats.accept_probs = dict(self._accept_prob)
            return

        total_offers = sum(self._offer_counts.values())
        n = len(priorities)
        prior = cfg.offer_prior

        raw: dict[int, float] = {}
        for sid, p in priorities.items():
            desired = max(p, 0.0) / total_p
            # Smoothed empirical offer share. The prior term makes an unseen seed
            # look "averagely offered" instead of infinitely rare, which would
            # otherwise produce a huge correction on its very first appearance.
            offered = (self._offer_counts.get(sid, 0) + prior) / (total_offers + prior * n)
            ratio = desired / offered if offered > 0 else cfg.max_correction
            raw[sid] = min(ratio, cfg.max_correction)

        peak = max(raw.values())
        if peak <= 0:
            self._accept_prob = {sid: 1.0 for sid in priorities}
        else:
            self._accept_prob = {
                sid: max(cfg.accept_floor, min(1.0, r / peak)) for sid, r in raw.items()
            }

        self.stats.accept_probs = dict(self._accept_prob)
        self._dirty = False

    def decide(self, seed_id: int) -> tuple[bool, float]:
        """The hot path. Returns ``(accept, accept_prob)``.

        Deliberately does no scoring work -- it consumes whatever the last
        :meth:`refresh` produced.
        """
        self._offer_counts[seed_id] = self._offer_counts.get(seed_id, 0) + 1
        self.stats.offers += 1
        prob = self._accept_prob.get(seed_id, 1.0)
        accepted = prob >= 1.0 or self.rng.random() < prob
        if accepted:
            self.stats.accepts += 1
        return accepted, prob

    def needs_refresh(self, now_s: float) -> bool:
        return (
            self._dirty
            or (now_s - self.stats.last_refresh_s) >= self.cfg.refresh_interval_s
        )

    def realised_shares(self) -> dict[int, float]:
        """Empirical offer share per seed -- reported alongside desired shares so
        the writeup can show how well the controller actually tracked its target."""
        total = sum(self._offer_counts.values())
        if not total:
            return {}
        return {sid: c / total for sid, c in self._offer_counts.items()}

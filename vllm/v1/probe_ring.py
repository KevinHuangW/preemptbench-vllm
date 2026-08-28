"""
PROBE Growth Ring — online adaptive histogram for remaining growth estimation.

M4 implementation of the ring data structure described in the PROBE ROADMAP.

Usage (scheduler side):

    from vllm.v1.probe_ring import GrowthRing

    ring = GrowthRing(n_buckets=8)

    # During victim selection:
    est = ring.query(req.num_computed_tokens, g_slope_val, memory_pressure=p)

    # After a preempted request eventually completes:
    ring.update(num_computed_at_preempt, g_slope_at_preempt, actual_remaining)

Design notes:
- 2D key: (len_bucket, g_bucket)
  * current_length axis: log2 bucketing (works well — spans 100s–10000s tokens)
  * g_score axis: online percentile bucketing (g_slope spans 1–50, not orders
    of magnitude, so log2 collapses ~95% of updates into 5 of 16 buckets)
- Each cell stores the observed actual_remaining values from completed requests.
- query() falls back to g_rough when too few samples exist (cold-start).
  Cold-start threshold: int(2 / (1-q)) — requires 2 samples above the target
  quantile; scales automatically with pressure without any magic floor.
- memory_pressure ∈ [0,1] → q ∈ [0.5, q_max] where q_max is variance-adaptive:
    q_max = 1 − 1/(1+CV), clamped to [0.70, 0.95]
  High-variance cells (uncertain distribution) → higher conservative quantile.
  Low-variance cells (tight distribution) → closer to median.
- n_buckets controls granularity along each axis.
- g_bucket boundaries are recomputed from observed data every
  G_BUCKET_REFIT_EVERY updates; uses quantiles of seen g_score values so the
  n_buckets resolution knob is decoupled from g_score magnitude.
"""

from __future__ import annotations

import bisect
import math
from collections import defaultdict


class GrowthRing:
    """Online bucketed histogram for remaining-growth estimation."""

    # How many g_score updates between percentile boundary refits.
    G_BUCKET_REFIT_EVERY: int = 50

    def __init__(self, n_buckets: int = 8) -> None:
        """
        Parameters
        ----------
        n_buckets:
            Number of buckets along each axis.
            current_length axis: log2 discretisation (boundaries at powers of 2).
            g_score axis: online percentile discretisation (boundaries refit
            from observed data every G_BUCKET_REFIT_EVERY updates).
        """
        if n_buckets < 1:
            raise ValueError("n_buckets must be >= 1")
        self.n_buckets = n_buckets
        # (len_bucket, g_bucket) -> sorted list of observed actual_remaining
        self._samples: dict[tuple[int, int], list[float]] = defaultdict(list)
        # total updates received (for stats / logging)
        self._total_updates: int = 0
        self._total_queries: int = 0
        self._cold_queries:  int = 0

        # Online percentile tracking for g_score axis.
        # _g_obs: sorted list of all observed g_score values seen in update().
        # _g_boundaries: list of (n_buckets - 1) breakpoints; bucket i covers
        #   [_g_boundaries[i-1], _g_boundaries[i]).
        self._g_obs: list[float] = []
        self._g_boundaries: list[float] = []  # empty → fall back to log2

    # ── discretisation ────────────────────────────────────────────────────

    def _len_bucket(self, val: float) -> int:
        """Log2 bucket for current_length (spans 100s–10000s tokens)."""
        if val <= 0.0:
            return 0
        return min(int(math.log2(max(1.0, val))), self.n_buckets - 1)

    def _g_bucket(self, val: float) -> int:
        """Percentile bucket for g_score (spans 1–50, not log-scale)."""
        if not self._g_boundaries:
            # Cold: fall back to log2 until enough data for a refit.
            if val <= 0.0:
                return 0
            return min(int(math.log2(max(1.0, val))), self.n_buckets - 1)
        # Binary search into current boundaries.
        idx = bisect.bisect_right(self._g_boundaries, val)
        return min(idx, self.n_buckets - 1)

    def _refit_g_boundaries(self) -> None:
        """Set g_score bucket boundaries from observed data (done once, then frozen).

        Boundaries are computed once from the first G_BUCKET_REFIT_EVERY
        observations and never changed.  This keeps all cells under a single
        consistent partitioning — refitting after data has been bucketed would
        orphan earlier samples in stale cells.
        """
        if self._g_boundaries:
            return  # frozen after first fit
        n = len(self._g_obs)
        if n < self.n_buckets:
            return  # not enough data yet
        boundaries: list[float] = []
        for i in range(1, self.n_buckets):
            # i/n_buckets quantile of observed g values
            frac = i / self.n_buckets
            idx = min(int(frac * n), n - 1)
            boundaries.append(self._g_obs[idx])
        self._g_boundaries = boundaries

    def _key(self, current_length: float, g_score: float) -> tuple[int, int]:
        return (self._len_bucket(current_length), self._g_bucket(g_score))

    # ── public API ────────────────────────────────────────────────────────

    def query(
        self,
        current_length: float,
        g_rough: float,
        memory_pressure: float = 0.5,
    ) -> float:
        """Return estimated remaining growth.

        Parameters
        ----------
        current_length:
            Number of tokens computed so far (prompt + output).
        g_rough:
            Entropy-derived rough estimate of remaining tokens (g_slope score).
        memory_pressure:
            Scalar ∈ [0, 1].  0 = throughput-optimised (median quantile);
            1 = conservative (variance-adaptive upper quantile).

        Returns
        -------
        float
            Estimated remaining growth.  Falls back to ``g_rough`` when the
            cell has fewer than int(2 / (1 − q)) samples (cold-start).
        """
        self._total_queries += 1
        key = self._key(current_length, g_rough)
        samples = self._samples.get(key)

        # Variance-adaptive upper quantile bound.
        # CV = std/mean measures distribution spread.  High spread → higher
        # conservative upper bound; tight distribution → stay near median.
        # q_max clamped to [0.70, 0.95] to avoid extremes.
        if samples is not None and len(samples) >= 2:
            n = len(samples)
            mean = sum(samples) / n
            variance = sum((x - mean) ** 2 for x in samples) / n
            cv = math.sqrt(variance) / (mean + 1e-9)
            q_max = 1.0 - 1.0 / (1.0 + cv)
            q_max = max(0.70, min(0.95, q_max))
        else:
            q_max = 0.90  # default before we have variance info

        q = 0.5 + (q_max - 0.5) * max(0.0, min(1.0, memory_pressure))

        # Cold-start threshold: need int(2 / (1-q)) total samples so that 2
        # fall in the top (1-q) fraction.  No magic floor — formula governs.
        min_samples = int(2.0 / (1.0 - q + 1e-9))
        min_samples = max(1, min_samples)  # always require at least 1 sample

        if samples is None or len(samples) < min_samples:
            self._cold_queries += 1
            return g_rough  # cold-start: fall back to caller's score

        idx = min(int(q * len(samples)), len(samples) - 1)
        return samples[idx]  # already sorted by update()

    def update(
        self,
        current_length: float,
        g_rough: float,
        actual_remaining: float,
    ) -> None:
        """Record an observed remaining-growth measurement.

        Parameters
        ----------
        current_length:
            ``num_computed_tokens`` at the moment of preemption.
        g_rough:
            g_slope score at the moment of preemption.
        actual_remaining:
            Tokens the request actually generated after the preemption point
            (i.e. final_output_len − output_len_at_preemption).
        """
        if actual_remaining < 0:
            return  # sanity guard

        # Update g_score percentile tracker before computing the bucket key
        # so this observation participates in the next refit.
        bisect.insort(self._g_obs, g_rough)
        self._total_updates += 1

        # Refit g boundaries every G_BUCKET_REFIT_EVERY observations.
        if self._total_updates % self.G_BUCKET_REFIT_EVERY == 0:
            self._refit_g_boundaries()

        key = self._key(current_length, g_rough)
        bisect.insort(self._samples[key], actual_remaining)

    # ── introspection ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return a snapshot of ring statistics (for logging / analysis)."""
        return {
            "n_buckets":     self.n_buckets,
            "n_cells":       len(self._samples),
            "total_updates": self._total_updates,
            "total_queries": self._total_queries,
            "cold_queries":  self._cold_queries,
            "warm_rate":     (
                1.0 - self._cold_queries / max(1, self._total_queries)
            ),
            "g_boundaries":  list(self._g_boundaries),
            "cells": {
                str(k): len(v) for k, v in self._samples.items()
            },
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"GrowthRing(n_buckets={self.n_buckets}, "
            f"cells={s['n_cells']}, "
            f"updates={s['total_updates']}, "
            f"warm_rate={s['warm_rate']:.1%})"
        )

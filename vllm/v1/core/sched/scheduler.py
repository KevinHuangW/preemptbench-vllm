# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import json
import math
import os
import time
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import numpy as np

from vllm import envs
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.v1 import probe_preempt_logger as _ppl

# PROBE victim selection policy (set via PROBE_VICTIM env var).
# PROBE_VICTIM=default              → vLLM FCFS (newest-first LIFO, unchanged)            [baseline: vLLM SOSP'23]
# PROBE_VICTIM=shortest_first       → smallest current KV cache (min num_computed_tokens) [baseline: B1 / InferMax arXiv 2411.07447]
# PROBE_VICTIM=least_decoded        → fewest output tokens so far (min num_output_tokens)  [baseline: B2 least-decoded-first]
# PROBE_VICTIM=largest_kv           → largest current KV cache (max num_computed_tokens)   [baseline: B3 largest-KV-first]
# PROBE_VICTIM=input_length         → longest input prompt (max num_prompt_tokens)         [baseline: B4 / FastServe arXiv 2305.05920]
# PROBE_VICTIM=benefit_cost         → max(kv_freed / recompute_cost) = num_computed/num_prompt [baseline: B5 / DBMS-inspired]
# PROBE_VICTIM=probe_slope          → improved g_rough: log-length bias + fast EMA + slope penalty [ours: P3]
# PROBE_VICTIM=probe_spike          → log-length × EMA-of-spike-indicator (H>1.0); replaces EMA slope [ours: P5]
# PROBE_VICTIM=oracle               → score = expected_total_tokens − num_computed (cheats: uses precomputed
#                                     prompt-hash → length lookup at PROBE_ORACLE_PATH). Upper bound for any
#                                     length-aware policy.
#
# Orthogonal admission control:
# PROBE_ADMISSION=oracle            → before admitting a waiting request, sum projected remaining KV blocks
#                                     across running requests using the same oracle lookup; if this exceeds
#                                     free pool blocks (with a small margin), skip admission this step.
#                                     Combined with PROBE_VICTIM=default this gives "oracle_admit"; combined
#                                     with PROBE_VICTIM=oracle this gives "oracle_full".  Tests whether
#                                     admission control alone (without smart preemption) suffices to dodge
#                                     the discard-mode recompute thrash.
# PROBE_ADMISSION_MARGIN=<float>    → fraction of free blocks reserved for the admitted request itself
#                                     (default 0.10).
#   Changes vs. probe (P1):
#     C — length multiplier replaced with log(1+tokens): kills quadratic length bias while
#         retaining a tiebreaker signal among sequences at the same entropy level.
#     A — uses fast EMA (α=0.3) in the base term so victim score drops quickly when a
#         sequence transitions to low-entropy (converging / answering) phase, and adds
#         λ·max(fast_h−slow_h, 0) slope bonus to further prioritise sequences whose
#         entropy is actively rising. Both changes are model-agnostic: they detect
#         phase transitions by entropy trajectory, not by token identity.
#   Tune λ via PROBE_SLOPE_LAMBDA env var (default 1.0).
# PROBE_VICTIM=probe_slope_ring     → P3 scorer (g_slope) + ring calibration with live KV pressure [ours: P4]
#   Ring keyed on g_slope; memory_pressure (1−free/total) injected into ring.query() to drive
#   pressure-adaptive quantile (0.5→0.9); n_buckets=16 spans 128→16384 tokens.
# PROBE_VICTIM=probe_slope_peos     → P3 variant using P(EOS) instead of token entropy [ablation: A1]
# PROBE_VICTIM=random               → uniform random victim among running reqs, seeded by
#                                     PROBE_VICTIM_SEED (defaults to PROBE_POISSON_SEED) for
#                                     reproducibility [control: does victim identity matter at all?]
_PROBE_VICTIM = os.environ.get("PROBE_VICTIM", "default")
_PROBE_VICTIM_SEED = int(os.environ.get(
    "PROBE_VICTIM_SEED", os.environ.get("PROBE_POISSON_SEED", "42")))
import random as _random_mod
_victim_rng = _random_mod.Random(_PROBE_VICTIM_SEED)  # used only by PROBE_VICTIM=random

# PROBE M3 constants.
# H_max = ln(vocab_size); GPT-OSS-120B uses tiktoken cl100k vocab (128 256 tokens)
# ln(128256) ≈ 11.762
_PROBE_H_MAX: float = 11.762
_PROBE_EPS: float = 1e-6
_PROBE_EMA_ALPHA: float = 0.1        # slow EMA — stable level signal
_PROBE_EMA_ALPHA_FAST: float = 0.3   # fast EMA — tracks recent entropy; responds in ~7 tokens
_PROBE_SPIKE_ALPHA: float = 0.05     # spike-rate EMA — ~20-token recency window
_PROBE_SPIKE_THRESHOLD: float = float(os.environ.get("PROBE_SPIKE_THRESHOLD", "1.0"))  # H > threshold = decision point
_PROBE_ZERO_THRESHOLD: float = float(os.environ.get("PROBE_ZERO_THRESHOLD", "0.05"))    # H < threshold = high-confidence step (paired with spike for 2-feature predictor)
_PROBE_SLOPE_LAMBDA: float = float(os.environ.get("PROBE_SLOPE_LAMBDA", "1.0"))
# Slope penalty weight for probe_slope.  Adds λ·max(fast_h−slow_h, 0) to the
# victim score, boosting priority of sequences whose entropy is actively rising
# (still expanding) vs. those that have already peaked and are converging.
# λ=1.0 keeps the bonus on the same scale as the log-length base term at
# moderate entropy; tune via PROBE_SLOPE_LAMBDA env var.
# Proactive trigger: fire when free KV blocks drop below this fraction of total.
_PROBE_HEADROOM_FRAC: float = float(os.environ.get("PROBE_HEADROOM_FRAC", "0.15"))

# P3e: weight for the symmetric convergence discount term.
# Adds -γ·max(slow_h - fast_h, 0) to make actively-converging sequences easier to evict.
_PROBE_SLOPE_GAMMA: float = float(os.environ.get("PROBE_SLOPE_GAMMA", "0.5"))
# P3f: weight for the P(EOS) fusion discount.
# Subtracts δ·fast_ema_p_eos from the entropy score (high P(EOS) → near termination → easier to evict).
_PROBE_PEOS_DELTA: float = float(os.environ.get("PROBE_PEOS_DELTA", "1.0"))
# P3g: exponential decay rate for the peak-entropy discount.
# score × exp(-β · peak_gap / H_max); β=2 → 37% weight when gap = H_max/2.
_PROBE_PEAK_BETA: float = float(os.environ.get("PROBE_PEAK_BETA", "2.0"))
# P3c: KV utilization thresholds for adaptive λ scaling.
# λ_eff = λ at kv_util ≤ LAM_LOW; λ_eff = 0 at kv_util ≥ LAM_HIGH; linear between.
_PROBE_ADAPTIVE_LAM_LOW: float = float(os.environ.get("PROBE_ADAPTIVE_LAM_LOW", "0.80"))
_PROBE_ADAPTIVE_LAM_HIGH: float = float(os.environ.get("PROBE_ADAPTIVE_LAM_HIGH", "0.95"))
# Module-level KV pressure for P3c (updated each schedule() step).
_probe_kv_pressure: float = 0.0
# Warm-init reset value for P3b (H_max when victim is probe_slope_b, else 0.0).
_PROBE_RESET_H: float = _PROBE_H_MAX if _PROBE_VICTIM == "probe_slope_b" else 0.0

# PROBE P4: probe_slope_ring — ring calibrated on g_slope with pressure injection.
# n_buckets=16: len buckets span up to 2^15=32768 tokens so AIME sequences
# (500–8000 tokens) spread across buckets 9–13 instead of all landing in bucket 7.
_PROBE_SLOPE_RING = None
if _PROBE_VICTIM == "probe_slope_ring":
    from vllm.v1.probe_ring import GrowthRing as _GrowthRing  # noqa: F811
    _PROBE_SLOPE_RING = _GrowthRing(
        n_buckets=int(os.environ.get("PROBE_SLOPE_RING_BUCKETS", "16"))
    )

# P4 sustained-pressure EMA: updated every scheduling step so it tracks ambient
# KV occupancy, not the instantaneous spike at preemption time (which is always
# ≈1.0 since preemption only fires when allocation fails).
# Hysteresis gate: ring activates when EMA ≥ ACTIVATE (0.90) and stays active
# until EMA drops below DEACTIVATE (0.85).  The 5-point band prevents the ring
# from flipping on/off at the saturation boundary (N≈88–96).  Below the band
# the ring is fully passive (pure P3 behaviour); above it the ring is fully
# active with pressure-adaptive calibration.
_probe_kv_ema: float = 0.0
_probe_ring_active: bool = False          # hysteresis state
_PROBE_KV_EMA_ALPHA: float = 0.02        # ~50-step time constant; ~1 s at 20 Hz
_PROBE_SLOPE_RING_ACTIVATE: float = float(
    os.environ.get("PROBE_SLOPE_RING_ACTIVATE", "0.90")
)
_PROBE_SLOPE_RING_DEACTIVATE: float = float(
    os.environ.get("PROBE_SLOPE_RING_DEACTIVATE", "0.85")
)

# Pending ring updates: req_id → (num_computed_at_preempt, g_score_at_preempt,
#                                  num_output_tokens_at_preempt)
# Populated when a request is preempted; consumed when it eventually finishes.
_probe_slope_ring_pending: dict[str, tuple[float, float, int]] = {}


def _probe_g_spike(request) -> float:  # type: ignore[return]
    """Victim score for probe_spike: log-length × EMA-of-spike-indicator.

    Replaces the fast-slow EMA slope with a recency-weighted count of
    high-entropy (H > 1.0 nat) tokens — decision-point bursts in reasoning
    traces.  Spike-rate EMA (α=0.05, ~20-token window) has Spearman ρ ≈ 0.58
    with total decode length vs ρ ≈ 0 for the EMA slope (see
    entropy/length_pred_study/REPORT.md).  No calibration coefficients needed.
    """
    return math.log1p(request.num_computed_tokens) * request.probe_spike_ema


# Oracle policy state (lazy-loaded on first scoring call to avoid load cost
# in policies that don't use it).  PROBE_ORACLE_PATH points to a JSON file
# produced by scripts/build_oracle_lookup.py.  The lookup keys are SHA-1
# hashes of the prompt's first/last 32 token IDs.
import hashlib as _hashlib
_PROBE_ORACLE_PATH = os.environ.get("PROBE_ORACLE_PATH", "")
_ORACLE_LOOKUP: dict[str, dict] | None = None
_ORACLE_FALLBACK: int = 12000
_ORACLE_HASH_PREFIX_LEN: int = 32
_ORACLE_HASH_SUFFIX_LEN: int = 32
# direct_id schema constants (populated when scheme == "direct_id"):
_ORACLE_USER_CONTENT_START: int = 74
_ORACLE_SLICE_LENGTH: int = 32
_ORACLE_REQ_CACHE: dict[str, int] = {}  # req_id -> resolved expected_total


def _oracle_hash_window(token_ids, start: int, length: int) -> str:
    if start >= 0:
        sub = tuple(token_ids[start:start + length])
    else:
        sub = tuple(token_ids[start:])
    return _hashlib.sha1(repr(sub).encode()).hexdigest()[:16]


_ORACLE_HASH_SCHEME = "prefix_suffix"  # or "full_prompt", "len_plus_suffix", "direct_id"


def _oracle_load_if_needed() -> None:
    global _ORACLE_LOOKUP, _ORACLE_FALLBACK
    global _ORACLE_HASH_PREFIX_LEN, _ORACLE_HASH_SUFFIX_LEN, _ORACLE_HASH_SCHEME
    global _ORACLE_USER_CONTENT_START, _ORACLE_SLICE_LENGTH
    if _ORACLE_LOOKUP is not None or not _PROBE_ORACLE_PATH:
        return
    try:
        with open(_PROBE_ORACLE_PATH) as fh:
            data = json.load(fh)
        # New direct_id schema (slice-based lookup) vs old hash-based schemas
        schema = str(data.get("schema", ""))
        if schema == "direct_id":
            _ORACLE_LOOKUP = dict(data.get("slice_to_entry", {}))
            _ORACLE_HASH_SCHEME = "direct_id"
            _ORACLE_USER_CONTENT_START = int(data.get("user_content_start", 74))
            _ORACLE_SLICE_LENGTH = int(data.get("slice_length", 32))
        else:
            _ORACLE_LOOKUP = dict(data.get("lookup", {}))
            _ORACLE_HASH_SCHEME = str(data.get("hash_scheme", "prefix_suffix"))
            _ORACLE_HASH_PREFIX_LEN = int(data.get("hash_prefix_len", 32))
            _ORACLE_HASH_SUFFIX_LEN = int(data.get("hash_suffix_len", 32))
        _ORACLE_FALLBACK = int(data.get("fallback_expected_total", 12000))
        extra = ""
        if _ORACLE_HASH_SCHEME == "direct_id":
            extra = (f", user_content_start={_ORACLE_USER_CONTENT_START}, "
                     f"slice_length={_ORACLE_SLICE_LENGTH}")
        print(f"[PROBE oracle] loaded {len(_ORACLE_LOOKUP)} entries, "
              f"scheme={_ORACLE_HASH_SCHEME}, fallback={_ORACLE_FALLBACK}{extra}",
              flush=True)
    except Exception as e:
        # Fail loud once and fall back to the static default; don't crash sched.
        print(f"[PROBE oracle] failed to load {_PROBE_ORACLE_PATH}: {e}",
              flush=True)
        _ORACLE_LOOKUP = {}


_ORACLE_HIT_CACHE: dict[str, bool] = {}  # req_id -> True if lookup hit (not fallback)
_PROBE_ORACLE_LOG_MISS = os.environ.get("PROBE_ORACLE_LOG_MISS", "1") not in ("0", "false", "False", "")


def _oracle_resolve(request) -> int:
    """Return expected_total_tokens for this request (with caching).

    On the FIRST resolution per request (not cached), if the lookup misses
    (falls back to default), prints a "[PROBE oracle] MISS" line so silent
    fallback doesn't go unnoticed. Subsequent calls hit the cache and are
    silent. Gate via PROBE_ORACLE_LOG_MISS=0 if log volume becomes a problem.
    """
    rid = request.request_id
    if rid in _ORACLE_REQ_CACHE:
        return _ORACLE_REQ_CACHE[rid]
    _oracle_load_if_needed()
    expected = _ORACLE_FALLBACK
    is_hit = False
    diag_slice = None
    if _ORACLE_LOOKUP:
        ids = request.prompt_token_ids
        if _ORACLE_HASH_SCHEME == "direct_id":
            # Slice the fixed (offset, length) window of prompt token IDs.
            # Sweep client prepends "[TASK aime=NNN attempt=KK]\n\n" to user
            # content, so this slice uniquely identifies (problem, attempt).
            start = _ORACLE_USER_CONTENT_START
            end = start + _ORACLE_SLICE_LENGTH
            slice_key = repr(tuple(ids[start:end]))
            diag_slice = slice_key
            entry = _ORACLE_LOOKUP.get(slice_key)
        elif _ORACLE_HASH_SCHEME == "full_prompt":
            h = _hashlib.sha1(repr(tuple(ids)).encode()).hexdigest()[:16]
            entry = _ORACLE_LOOKUP.get(h)
        elif _ORACLE_HASH_SCHEME == "len_plus_suffix":
            suf = tuple(ids[-_ORACLE_HASH_SUFFIX_LEN:])
            h = _hashlib.sha1(repr((len(ids), suf)).encode()).hexdigest()[:16]
            entry = _ORACLE_LOOKUP.get(h)
        elif _ORACLE_HASH_SCHEME == "suffix":
            entry = _ORACLE_LOOKUP.get(
                _oracle_hash_window(ids, -_ORACLE_HASH_SUFFIX_LEN,
                                    _ORACLE_HASH_SUFFIX_LEN))
        else:
            h_pref = _oracle_hash_window(ids, 0, _ORACLE_HASH_PREFIX_LEN)
            entry = _ORACLE_LOOKUP.get(h_pref)
            if entry is None:
                h_suff = _oracle_hash_window(ids, -_ORACLE_HASH_SUFFIX_LEN,
                                             _ORACLE_HASH_SUFFIX_LEN)
                entry = _ORACLE_LOOKUP.get(h_suff)
        if entry and entry.get("expected_total") is not None:
            expected = int(entry["expected_total"])
            is_hit = True
    if not is_hit and _PROBE_ORACLE_LOG_MISS:
        # First-time miss for this rid → log so silent fallback is visible.
        diag = ""
        if diag_slice is not None:
            diag = f" slice={diag_slice[:96]}{'...' if len(diag_slice) > 96 else ''}"
        print(f"[PROBE oracle] MISS rid={rid} scheme={_ORACLE_HASH_SCHEME} "
              f"→ fallback={_ORACLE_FALLBACK}{diag}",
              flush=True)
    _ORACLE_REQ_CACHE[rid] = expected
    _ORACLE_HIT_CACHE[rid] = is_hit
    return expected


def _oracle_log_decision(victim, running_reqs) -> None:
    """Log one preempt-decision record for offline verification.

    Writes to PROBE_ORACLE_DECISION_LOG (one JSON line per oracle preempt):
      victim       : (req_id, expected_total, num_output_tokens, score, hit)
      pool_stats   : n, n_hit, n_fallback, score_p50, score_max
      victim_rank  : 0 = longest remaining among hits, ... (− if fallback)
    """
    path = os.environ.get("PROBE_ORACLE_DECISION_LOG", "")
    if not path:
        return
    try:
        # Compute scores for ALL running requests (cheap: cached lookup).
        scores = []
        for r in running_reqs:
            exp = _oracle_resolve(r)
            sc = float(exp - r.num_output_tokens)
            hit = _ORACLE_HIT_CACHE.get(r.request_id, False)
            scores.append((sc, hit, r.request_id, exp, r.num_output_tokens))
        scores.sort(reverse=True)  # highest score first
        # Find victim's rank in the sorted list
        v_rid = victim.request_id
        v_rank = next((i for i, s in enumerate(scores) if s[2] == v_rid), -1)

        n = len(scores)
        n_hit = sum(1 for s in scores if s[1])
        n_fb = n - n_hit
        if n > 0:
            sc_vals = [s[0] for s in scores]
            p50 = sc_vals[n // 2]
            mx = sc_vals[0]
        else:
            p50 = mx = 0.0

        v_exp = _oracle_resolve(victim)
        v_dec = victim.num_output_tokens
        v_hit = _ORACLE_HIT_CACHE.get(v_rid, False)
        rec = {
            "t": time.time(),
            "victim": {
                "req_id": v_rid,
                "expected_total": v_exp,
                "num_output_tokens": v_dec,
                "score": float(v_exp - v_dec),
                "hit": v_hit,
                "rank_in_pool": v_rank,
            },
            "pool": {"n": n, "n_hit": n_hit, "n_fallback": n_fb,
                     "score_p50": p50, "score_max": mx},
        }
        with open(path, "a", buffering=1) as f:
            f.write(json.dumps(rec) + "\n")
    except Exception as e:
        # Never let logging crash scheduling.
        print(f"[PROBE oracle-decision-log] {e}", flush=True)


def _probe_g_oracle(request) -> float:  # type: ignore[return]
    """Victim score for oracle: predicted remaining tokens.

    score = expected_total_decode_tokens - num_computed_tokens
    Higher score = more work left = preferred preemption victim.

    expected_total_decode_tokens is looked up by hashing the prompt token IDs
    against a precomputed table built from previous T=0 runs of the same
    workload.  Greedy decoding makes T=0 lengths ~deterministic per prompt,
    so this is genuinely an oracle ceiling for length-aware victim selection.
    """
    expected_total = _oracle_resolve(request)
    # Oracle gives expected total OUTPUT (completion) tokens; num_computed
    # includes prompt + generated, so we use num_output_tokens for fairness.
    decoded = request.num_output_tokens
    return float(expected_total - decoded)


# Orthogonal admission control.
_PROBE_ADMISSION = os.environ.get("PROBE_ADMISSION", "")

# PROBE: reactive-only offload mode.
# When PROBE_REACTIVE_OFFLOAD=1 and the offloading connector is attached,
# preempted requests stash their remaining KV blocks to CPU via async D2H
# (using the existing free_to_drain + PRE_OFFLOADING + proactive-store
# pipeline that was previously dead code).  The trigger condition is
# UNCHANGED from DISCARD mode — only the per-preempt cost changes
# (PCIe transfer instead of full re-prefill on resume).  Designed for the
# clean experiment: identical victim selection + identical event counts
# vs DISCARD, only resume cost differs.
_PROBE_REACTIVE_OFFLOAD: bool = (
    os.environ.get("PROBE_REACTIVE_OFFLOAD", "0") == "1"
)
_PROBE_ADMISSION_MARGIN = float(os.environ.get("PROBE_ADMISSION_MARGIN", "0.10"))

# Co-designed admission+preempt for discard mode.
# beta penalises decoded work so we don't evict requests that have paid
# heavy recompute cost.  threshold lets admission ask "is there a cheap
# victim available?" before refusing — the two halves work together.
_PROBE_PREEMPT_BETA = float(os.environ.get("PROBE_PREEMPT_BETA", "1.0"))
_PROBE_ADMIT_THRESHOLD = float(os.environ.get("PROBE_ADMIT_THRESHOLD", "0.0"))
# Offload-mode swap-back-cost horizon: requests with remaining decode tokens
# below this are protected from eviction (their KV would be paged out and back
# very soon, making the swap a net loss).  Used by kv_remaining_swapaware_*.
_PROBE_SWAP_C = float(os.environ.get("PROBE_SWAP_C", "1000"))
# SF-aware admission: refuse if the SF-chosen victim (argmin num_computed) still
# has > this many tokens of expected remaining work — protects long fresh jobs
# from getting evicted by SF and causing recompute thrash.
_PROBE_ADMIT_SF_REMAINING_MAX = int(os.environ.get("PROBE_ADMIT_SF_REMAINING_MAX", "2000"))

# Adaptive β: scales between BETA_MIN and BETA_MAX as util^BETA_K
_PROBE_BETA_MIN = float(os.environ.get("PROBE_BETA_MIN", "0.5"))
_PROBE_BETA_MAX = float(os.environ.get("PROBE_BETA_MAX", "2.5"))
_PROBE_BETA_K = float(os.environ.get("PROBE_BETA_K", "2.0"))

# Predictive admission: simulate forward H seconds, refuse if peak > capacity.
# Decode rate model: 1 token per request per scheduling step (vLLM batch decode).
_PROBE_PRED_HORIZON_STEPS = int(os.environ.get("PROBE_PRED_HORIZON_STEPS", "1000"))


def _kv_utilization(block_pool) -> float:
    free = block_pool.get_num_free_blocks()
    total = getattr(block_pool, "num_gpu_blocks", None)
    if total is None or total <= 0:
        return 0.0
    return max(0.0, 1.0 - free / total)


def _adaptive_beta(util: float) -> float:
    return _PROBE_BETA_MIN + (_PROBE_BETA_MAX - _PROBE_BETA_MIN) * (util ** _PROBE_BETA_K)


def _probe_g_oracle_co_adapt(request, block_pool=None) -> float:
    """oracle_co with adaptive β tracking KV utilization.

    Uses module-global _CURRENT_ADAPTIVE_BETA which the dispatch site updates
    each schedule() round. β starts low (admit eagerly) and ramps up as KV
    pool fills, naturally tightening admission and protecting decoded work
    when the pool is hot.
    """
    expected_total = _oracle_resolve(request)
    decoded = request.num_output_tokens
    return float(expected_total - (1.0 + _CURRENT_ADAPTIVE_BETA) * decoded)


_CURRENT_ADAPTIVE_BETA: float = 1.0  # updated each schedule() round


def _oracle_co_adapt_admission_should_refuse(running_reqs, block_pool, block_size):
    """Same shape as oracle_co admission, but using adaptive β for victim score."""
    free = block_pool.get_num_free_blocks()
    if free <= 0:
        return True
    projected = 0
    for req in running_reqs:
        exp = _oracle_resolve(req)
        rem = exp - req.num_output_tokens
        if rem > 0:
            projected += (rem + block_size - 1) // block_size
    available = max(1, int(free * (1.0 - _PROBE_ADMISSION_MARGIN)))
    if projected <= available:
        return False
    if not running_reqs:
        return True
    best_score = max(_probe_g_oracle_co_adapt(r) for r in running_reqs)
    return best_score < _PROBE_ADMIT_THRESHOLD


# Method 5 — Two-level priority queues (long/short classification by oracle).
# Long requests (expected_total >= LONG_THRESHOLD) get a hard concurrent-count
# cap; shorts always admit. Preempt side is default LIFO (which naturally
# evicts shorts as they're newest).
_PROBE_LONG_THRESHOLD = int(os.environ.get("PROBE_LONG_THRESHOLD", "25000"))
_PROBE_MAX_LONG_INFLIGHT = int(os.environ.get("PROBE_MAX_LONG_INFLIGHT", "4"))


def _is_long_request(request) -> bool:
    return _oracle_resolve(request) >= _PROBE_LONG_THRESHOLD


def _two_level_should_skip_arrival(arrival, running_reqs) -> bool:
    """Per-arrival check: returns True if this arrival should be skipped
    (deferred to skipped_waiting) because it's a long request and the
    long-lane is at capacity. Shorts are always admitted."""
    if not _is_long_request(arrival):
        return False
    n_long = sum(1 for r in running_reqs if _is_long_request(r))
    return n_long >= _PROBE_MAX_LONG_INFLIGHT


def _sf_oracle_tiebreak_key(request) -> tuple[int, float]:
    """SF + oracle remaining-decode tiebreaker.

    Primary: num_computed_tokens (same as shortest_first — minimize recompute
    cost on resume).  Secondary (tiebreak among equal-or-near-equal primary):
    negative remaining-decode, so among cheap victims we prefer the one with
    the LONGEST remaining work (most "opportunity" to delay).

    Used with `min(running, key=...)` — smaller primary wins; among ties,
    smaller secondary (= larger remaining_decode) wins.

    Admission and trigger condition: unchanged from shortest_first.
    """
    decoded = request.num_output_tokens
    expected_total = _oracle_resolve(request)
    remaining = max(0, expected_total - decoded)
    # Tiebreak field is negative remaining, scaled down so it only affects
    # ordering when primary values are very close.  Negative → min picks
    # LARGEST remaining among ties.
    return (request.num_computed_tokens, -float(remaining))


def _probe_g_smart_lifo(request) -> float:
    """Method 3 — Smart-LIFO scoring: among requests with low decoded
    (cheap to recompute), favor those with largest expected_total. Score
    = expected_total / max(1, decoded). High score = fresh long-tail
    request — best preempt victim because it's both cheap to redo AND
    has the most pending KV pressure to release.
    """
    expected = _oracle_resolve(request)
    decoded = max(1, request.num_output_tokens)
    return float(expected / decoded)


def _oracle_reservation_admission_should_refuse(running_reqs, block_pool, block_size):
    """Method 4 — Reservation-based admission. Each in-flight request
    "reserves" its full peak KV (= expected_total / block_size).  If the
    sum of reservations exceeds (1 - margin) * total capacity, refuse new
    admissions.  No preempt-side change — uses whatever PROBE_VICTIM is
    set to (default LIFO if the synthetic policy is reservation_admit).

    Difference from oracle_admit: that one summed *remaining* blocks (which
    drops as decode progresses); this one sums *peak* blocks (constant per
    request) so it's a stricter ceiling.  The trade-off is whether to gate
    based on what's left to do (admission_overcommit) or what could ever be
    needed (this).  Stricter usually = fewer preemptions but more queue wait.
    """
    free = block_pool.get_num_free_blocks()
    total = getattr(block_pool, "num_gpu_blocks", None)
    if total is None or total <= 0:
        return False
    peak_reserved = 0
    for req in running_reqs:
        expected = _oracle_resolve(req)
        peak_reserved += (expected + block_size - 1) // block_size
    capacity = int(total * (1.0 - _PROBE_ADMISSION_MARGIN))
    return peak_reserved > capacity


def _oracle_pred_admission_should_refuse(running_reqs, block_pool, block_size):
    """Predictive admission: simulate KV trajectory for H steps and refuse if
    peak utilization exceeds capacity AND no cheap victim is available.

    Per-step model: each running request grows by 1 token (i.e. ceil(steps/block_size)
    new blocks in H steps).  We project total blocks needed at horizon H,
    compare to capacity, and refuse if overflow predicted unless a cheap
    victim could absorb it.
    """
    free = block_pool.get_num_free_blocks()
    total = getattr(block_pool, "num_gpu_blocks", None) or (free * 2)
    if free <= 0:
        return True
    H = _PROBE_PRED_HORIZON_STEPS
    # Sum projected additional blocks: each running request grows by min(remaining, H) tokens
    projected_growth = 0
    for req in running_reqs:
        exp = _oracle_resolve(req)
        rem = max(0, exp - req.num_output_tokens)
        # Within horizon, request decodes at most min(rem, H) more tokens
        gain = min(rem, H)
        projected_growth += (gain + block_size - 1) // block_size
    used = total - free
    projected_peak = used + projected_growth
    if projected_peak <= total * (1.0 - _PROBE_ADMISSION_MARGIN):
        return False  # no overflow predicted within horizon
    if not running_reqs:
        return True
    # Overflow predicted: check if a cheap victim is available
    best_score = max(_probe_g_oracle_co(r) for r in running_reqs)  # uses static β=PROBE_PREEMPT_BETA
    return best_score < _PROBE_ADMIT_THRESHOLD


def _probe_g_oracle_co(request) -> float:
    """Cost-aware oracle score: penalises decoded work to protect sunk cost.

    score = (expected_total - decoded) - beta * decoded
          = expected_total - (1 + beta) * decoded

    Higher score = better preempt victim (long remaining + low decoded).
    With beta=0 this is identical to _probe_g_oracle. Larger beta protects
    high-progress requests more aggressively.
    """
    expected_total = _oracle_resolve(request)
    decoded = request.num_output_tokens
    return float(expected_total - (1.0 + _PROBE_PREEMPT_BETA) * decoded)


def _probe_g_longest_remaining(request) -> float:
    """P1: argmax(expected_total - decoded). Picks the request with the most
    decode work still to do. Derived from §4.2 of eviction_economics.md as the
    profit-maximising victim under throughput-bound DISCARD: evicting it frees
    KV blocks that would otherwise be tied up for the longest natural duration.
    Has no no-oracle equivalent."""
    return float(max(0, _oracle_resolve(request) - request.num_output_tokens))


def _probe_g_kv_remaining_product(request) -> float:
    """P2: argmax(K_R · remaining). Full §4.3 ROI score: balances current KV
    footprint (block-seconds freed per eviction) with remaining work (so we
    don't evict near-done requests)."""
    decoded = request.num_output_tokens
    expected = _oracle_resolve(request)
    remaining = max(0, expected - decoded)
    kv = len(request.prompt_token_ids) + decoded
    return float(kv * remaining)


def _probe_g_kv_remaining_co(request) -> float:
    """P2-v2: K_R · (E − (1+β)·N). Combines kv_remaining_product's K-weighting
    with oracle_co's sunk-cost discount.

    Reasoning:
      P2 (kv_remaining_product) wins at low λ by picking the most "valuable to
      free" victim (big K_R × long remaining) — but it has the highest C per
      preempt of any policy, so under SLO discipline it breaks down past
      λ=0.035. The score below adds oracle_co's sunk-cost penalty:

          score = K_R · (E − (1+β)·N)

      At N = 0 this reduces to K · E (full kv_remaining_product weight).
      At N = E/(1+β) the second factor is 0 (request becomes ineligible).
      At N > E/(1+β) the score is negative (request is filtered out).

      With β = PROBE_PREEMPT_BETA (default 1.0):
        - threshold for ineligibility is N > E/2 (50% done)
        - protects near-completion victims absolutely
        - retains K-weighted choice among low-progress candidates

      Predicted behaviour:
        - λ=0.030: ~identical to P2 (most candidates have N ≪ E/2 anyway)
        - λ=0.035-0.040: lower C than P2 (eligible set excludes mid-progress
          requests), should stay Tier-1 longer
        - λ=0.045: behaves closer to longest_remaining with K-weight
    """
    decoded = request.num_output_tokens
    expected = _oracle_resolve(request)
    kv = len(request.prompt_token_ids) + decoded
    # second factor goes through zero at N = E/(1+β), then negative
    cost_adjusted_remaining = expected - (1.0 + _PROBE_PREEMPT_BETA) * decoded
    return float(kv * cost_adjusted_remaining)


# Cap on per-request preempt count (anti-thrash shield / §8-B). After a request
# has been preempted >= MAX times it becomes INELIGIBLE for further preempt
# (the score function dispatch filters it out before argmax). This breaks the
# cascade attractor that all length-aware victim functions are vulnerable to
# (rev-4 scratch pad finding). Set 0 to disable (default 4).
_PROBE_MAX_PREEMPTS_PER_REQ: int = int(os.environ.get("PROBE_MAX_PREEMPTS_PER_REQ", "4"))

# Near-completion shield (§8-A). When >0, requests whose entropy-spike predictor
# says they have fewer than K remaining tokens become INELIGIBLE for preempt.
# Only fires when the predictor is active (num_computed >= MIN_DECODED).
# Set 0 to disable (default 0 = off).
_PROBE_SHIELD_NEAR_COMPLETE_K: int = int(
    os.environ.get("PROBE_SHIELD_NEAR_COMPLETE_K", "0")
)

# Whether the default LIFO branch (PROBE_VICTIM=default) should apply the
# shield filters. Off by default to preserve baseline LIFO behavior;
# opt-in via PROBE_DEFAULT_LIFO_SHIELD=1 in the `default_ab` recipe.
_PROBE_DEFAULT_LIFO_SHIELD: bool = (
    os.environ.get("PROBE_DEFAULT_LIFO_SHIELD", "0").lower()
    not in ("0", "false", "no", "")
)


def _near_completion(r) -> bool:
    """True if the predictor says this request is within K tokens of finishing
    and PROBE_SHIELD_NEAR_COMPLETE_K is set. Returns False when the predictor
    has no signal yet (num_computed < MIN_DECODED) so early-stage requests
    aren't falsely shielded by the fallback constant."""
    if _PROBE_SHIELD_NEAR_COMPLETE_K <= 0:
        return False
    n = r.num_computed_tokens
    if n < _SPIKE_E_MIN_DECODED:
        return False
    e_hat = _spike_estimate_E(r)
    return (e_hat - n) < _PROBE_SHIELD_NEAR_COMPLETE_K


def _eligible_for_preempt(running):
    """Filter running pool to requests that haven't hit the per-request preempt
    cap (§8-B) AND aren't predicted to be near completion (§8-A).
    Falls back to the full pool if every request is shielded (rare but possible
    at deep saturation — we still need to preempt SOMEONE)."""
    eligible = running
    if _PROBE_MAX_PREEMPTS_PER_REQ > 0:
        eligible = [r for r in eligible
                    if getattr(r, "num_preemptions", 0) < _PROBE_MAX_PREEMPTS_PER_REQ]
    if _PROBE_SHIELD_NEAR_COMPLETE_K > 0:
        eligible = [r for r in eligible if not _near_completion(r)]
    return eligible if eligible else running


# S2 online length predictor coefficients (fitted on round-1 calibration data
# at checkpoint cp=2048; see docs/scratch_pad.md). Tunable via env vars so we
# can swap in fits from longer or shorter trajectories.
_SPIKE_E_INTERCEPT: float = float(os.environ.get("PROBE_SPIKE_E_INTERCEPT", "8.55"))
_SPIKE_E_SLOPE: float = float(os.environ.get("PROBE_SPIKE_E_SLOPE", "6.08"))
# 2-feature spike+zero predictor: adds near_zero_rate term. Defaults to 0.0 so
# the 1-feature spike-only predictor is preserved when this env var is unset.
# On HLE, spike-only has r≈0 but spike+zero recovers r≈0.39 (see scratch_pad §15).
_SPIKE_E_ZERO_SLOPE: float = float(os.environ.get("PROBE_SPIKE_E_ZERO_SLOPE", "0.0"))
_SPIKE_E_MIN_DECODED: int = int(os.environ.get("PROBE_SPIKE_E_MIN_DECODED", "256"))
_SPIKE_E_FALLBACK: int = int(os.environ.get("PROBE_SPIKE_E_FALLBACK", "12760"))
# Gate mode for early-stage requests (N < MIN_DECODED):
#   "constant"  — return _SPIKE_E_FALLBACK (legacy behavior; fits AIME's long decodes)
#   "gate"      — return N+1, marking the request as "no signal yet"; co-design
#                 score formula then makes them ineligible (E-(1+β)·N goes negative
#                 for any non-trivial N). Recommended for short-decode workloads
#                 like GSM8K where the constant fallback wildly inflates early-stage E.
_SPIKE_E_GATE_MODE: str = os.environ.get("PROBE_SPIKE_E_GATE_MODE", "constant")


def _spike_estimate_E(request) -> int:
    """Online entropy-spike length predictor (paper-deployable; no offline calib).

    T̂ = exp(a + b · spike_count / num_computed)  when num_computed ≥ MIN_DECODED
    else either FALLBACK constant or a gate value (PROBE_SPIKE_E_GATE_MODE).

    Coefficients fit on round-1 AIME calibration data (n=2312 stop/length
    completions joined with entropy.jsonl per-token entropy): a=8.55, b=6.08
    at cp=2048, R²=0.29, MAPE ≈ 46%, AUC@10K = 0.79.

    Result is clamped to at least N+1 (a request can never have fewer remaining
    tokens than already decoded) so the score functions don't get negative
    "remaining" inputs from a noisy predictor.
    """
    import math as _math
    N = request.num_computed_tokens
    if N < _SPIKE_E_MIN_DECODED:
        if _SPIKE_E_GATE_MODE == "gate":
            return N + 1
        return _SPIKE_E_FALLBACK
    rate = getattr(request, "probe_spike_count", 0) / N
    log_T = _SPIKE_E_INTERCEPT + _SPIKE_E_SLOPE * rate
    if _SPIKE_E_ZERO_SLOPE != 0.0:
        zero_rate = getattr(request, "probe_near_zero_count", 0) / N
        log_T += _SPIKE_E_ZERO_SLOPE * zero_rate
    T_hat = _math.exp(log_T)
    return int(max(T_hat, N + 1))


def _probe_g_kv_remaining_co_entropy(request) -> float:
    """Same victim score as kv_remaining_co (cost-aware K_R·(E−(1+β)·N))
    but with E sourced from the S2 online entropy-spike predictor instead
    of the offline oracle. This is the deployment-grade version of the
    method — no per-workload calibration required.
    """
    decoded = request.num_output_tokens
    expected = _spike_estimate_E(request)
    kv = len(request.prompt_token_ids) + decoded
    cost_adjusted_remaining = expected - (1.0 + _PROBE_PREEMPT_BETA) * decoded
    return float(kv * cost_adjusted_remaining)


def _probe_g_entropy_remaining(request) -> float:
    """Pure remaining-decode (no co-design): argmax(E − N), with E from the
    S2 entropy-spike predictor. Mirror of _probe_g_longest_remaining but using
    the online predictor instead of the oracle. Strips out the (P+N) prefix
    K-weight and the (1+β) sunk-cost discount used in kv_remaining_co_entropy.
    """
    expected = _spike_estimate_E(request)
    decoded = request.num_output_tokens
    return float(max(0, expected - decoded))


def _probe_g_entropy_least_remaining(request) -> float:
    """SJF mirror of _probe_g_entropy_remaining: argmax(N − E) = preempt the
    request closest to finishing (most decoded relative to predicted total).
    Direction-flip experiment per scratch_pad_slo.md §19. Predicted to be
    anti-pattern because preempting near-completion wastes max sunk compute.
    """
    expected = _spike_estimate_E(request)
    decoded = request.num_output_tokens
    return float(decoded - expected)


def _probe_g_oracle_least_remaining(request) -> float:
    """SJF mirror of _probe_g_oracle: argmax(N − E_true). Perfect-knowledge
    direction-flip — preempt the request truly closest to completion."""
    expected = _oracle_resolve(request)
    decoded = request.num_output_tokens
    return float(decoded - expected)


def _probe_g_kv_least_remaining_co_entropy(request) -> float:
    """SJF + KV co-design mirror of _probe_g_kv_remaining_co_entropy.
    score = K_R · ((1+β)·N − E)  (sign-flipped LJF co-design)
    argmax picks: big-KV + high decoded + low predicted = "predictor-overrun
    weighted by capacity gain". See scratch_pad_slo §19h Design A.
    """
    decoded = request.num_output_tokens
    expected = _spike_estimate_E(request)
    kv = len(request.prompt_token_ids) + decoded
    cost_adjusted_overrun = (1.0 + _PROBE_PREEMPT_BETA) * decoded - expected
    return float(kv * cost_adjusted_overrun)


def _probe_g_kv_least_remaining_co_oracle(request) -> float:
    """SJF + KV co-design with perfect knowledge. Mirror of _probe_g_oracle_co.
    score = (expected − (1+β)·decoded) flipped: ((1+β)·decoded − expected) × kv
    """
    decoded = request.num_output_tokens
    expected = _oracle_resolve(request)
    kv = len(request.prompt_token_ids) + decoded
    cost_adjusted_overrun = (1.0 + _PROBE_PREEMPT_BETA) * decoded - expected
    return float(kv * cost_adjusted_overrun)


def _probe_g_kv_remaining_swapaware_entropy(request) -> float:
    """Offload-mode-tuned victim score: KV · [1 − β · C / (remaining + ε)].

    Uses the S2 entropy-spike predictor for remaining-decode-length. Differs
    from kv_remaining_co_entropy in cost model — designed for OFFLOAD mode
    where preempted KV is paged to CPU (no recompute cost), so the dominant
    eviction cost is the swap-back transfer + scheduling delay.

    Near-done requests have small (remaining), so the bracket goes strongly
    negative — they are protected from eviction (let them finish). Long-
    remaining requests have large (remaining), bracket → 1 — they expose
    full KV value, freely evictable.
    """
    decoded = request.num_output_tokens
    expected = _spike_estimate_E(request)
    kv = len(request.prompt_token_ids) + decoded
    remaining = max(1, expected - decoded)
    swap_back_p = min(1.0, _PROBE_SWAP_C / remaining)
    return float(kv * (1.0 - _PROBE_PREEMPT_BETA * swap_back_p))


def _probe_g_age_weighted(request) -> float:
    """P4: KV-remaining-product divided by 1 + num_preemptions.
    Anti-starvation: deprioritise requests that have already been preempted."""
    decoded = request.num_output_tokens
    expected = _oracle_resolve(request)
    remaining = max(0, expected - decoded)
    kv = len(request.prompt_token_ids) + decoded
    age = 1 + getattr(request, "num_preemptions", 0)
    return float((kv * remaining) / age)


def _probe_g_longest_decoded(request) -> float:
    """P5: argmax(num_computed_tokens). Largest current KV footprint by
    decoded length only (ignores prompt). Predicted to win on bimodal
    workloads by routing preempt pressure to long-decode jobs and protecting
    short jobs from thrash."""
    return float(request.num_computed_tokens)


def _oracle_co_admission_should_refuse(running_reqs, block_pool, block_size: int) -> bool:
    """Co-designed admission check.

    Refuse admission only when:
      (a) the in-flight pool's projected remaining KV demand exceeds free blocks
          (preemption would be forced if we admit), AND
      (b) no acceptable victim exists — i.e. max_co_score across running
          requests is below ADMIT_THRESHOLD (all candidates would be costly to
          evict because they're all high-progress).

    If pool has headroom OR there's a cheap victim available, admit.
    """
    if not _ORACLE_LOOKUP and _PROBE_ADMISSION == "oracle_co":
        _oracle_load_if_needed()
    free_blocks = block_pool.get_num_free_blocks()
    if free_blocks <= 0:
        return True
    projected = 0
    for req in running_reqs:
        expected = _oracle_resolve(req)
        remaining = expected - req.num_output_tokens
        if remaining > 0:
            projected += (remaining + block_size - 1) // block_size
    available = max(1, int(free_blocks * (1.0 - _PROBE_ADMISSION_MARGIN)))
    if projected <= available:
        # Pool will fit naturally — admit.
        return False
    # Pool would overflow. Check if a cheap victim exists.
    if not running_reqs:
        return True
    best_score = max(_probe_g_oracle_co(r) for r in running_reqs)
    return best_score < _PROBE_ADMIT_THRESHOLD


def _oracle_co_lifo_admission_should_refuse(running_reqs, block_pool, block_size: int) -> bool:
    """LIFO-aware admission gate (mirror of oracle_co_sf for the default victim).

    vLLM's default preempt is `running.pop()` — the most-recently-admitted
    in-flight request. Refuse admission if that victim still has long
    expected remaining decode work (high recompute cost on DISCARD).
    """
    if not _ORACLE_LOOKUP and _PROBE_ADMISSION == "oracle_co_lifo":
        _oracle_load_if_needed()
    free_blocks = block_pool.get_num_free_blocks()
    if free_blocks <= 0:
        return True
    projected = 0
    for req in running_reqs:
        expected = _oracle_resolve(req)
        remaining = expected - req.num_output_tokens
        if remaining > 0:
            projected += (remaining + block_size - 1) // block_size
    available = max(1, int(free_blocks * (1.0 - _PROBE_ADMISSION_MARGIN)))
    if projected <= available:
        return False
    if not running_reqs:
        return True
    lifo_victim = running_reqs[-1]
    lifo_victim_remaining = max(0, _oracle_resolve(lifo_victim) - lifo_victim.num_output_tokens)
    return lifo_victim_remaining > _PROBE_ADMIT_SF_REMAINING_MAX


def _oracle_co_il_admission_should_refuse(running_reqs, block_pool, block_size: int) -> bool:
    """input_length-aware admission gate (mirror of oracle_co_sf for IL victim).

    input_length preempt selects argmax(len(prompt_token_ids)). Refuse
    admission if the IL-chosen victim's expected remaining is large.
    """
    if not _ORACLE_LOOKUP and _PROBE_ADMISSION == "oracle_co_il":
        _oracle_load_if_needed()
    free_blocks = block_pool.get_num_free_blocks()
    if free_blocks <= 0:
        return True
    projected = 0
    for req in running_reqs:
        expected = _oracle_resolve(req)
        remaining = expected - req.num_output_tokens
        if remaining > 0:
            projected += (remaining + block_size - 1) // block_size
    available = max(1, int(free_blocks * (1.0 - _PROBE_ADMISSION_MARGIN)))
    if projected <= available:
        return False
    if not running_reqs:
        return True
    il_victim = max(running_reqs, key=lambda r: len(r.prompt_token_ids))
    il_victim_remaining = max(0, _oracle_resolve(il_victim) - il_victim.num_output_tokens)
    return il_victim_remaining > _PROBE_ADMIT_SF_REMAINING_MAX


def _oracle_co_lr_admission_should_refuse(running_reqs, block_pool, block_size: int) -> bool:
    """LR-aware admission gate (mirror of oracle_co_sf for the longest_remaining victim).

    Refuse admission only when overflow is projected AND the LR-chosen victim
    (argmax remaining) has remaining work below the admit-cost threshold —
    i.e. when admitting now would force the policy to evict a request whose
    remaining work is small relative to recompute cost, i.e. a poor trade.
    """
    if not _ORACLE_LOOKUP and _PROBE_ADMISSION == "oracle_co_lr":
        _oracle_load_if_needed()
    free_blocks = block_pool.get_num_free_blocks()
    if free_blocks <= 0:
        return True
    projected = 0
    for req in running_reqs:
        expected = _oracle_resolve(req)
        remaining = expected - req.num_output_tokens
        if remaining > 0:
            projected += (remaining + block_size - 1) // block_size
    available = max(1, int(free_blocks * (1.0 - _PROBE_ADMISSION_MARGIN)))
    if projected <= available:
        return False
    if not running_reqs:
        return True
    # LR victim = argmax remaining; refuse if even that "best" victim isn't worth evicting.
    lr_victim = max(running_reqs, key=lambda r: _oracle_resolve(r) - r.num_output_tokens)
    lr_remaining = max(0, _oracle_resolve(lr_victim) - lr_victim.num_output_tokens)
    # Refuse when the LR victim's remaining is BELOW threshold — i.e. the
    # cheapest-to-keep victim is also near-done and shouldn't be evicted.
    return lr_remaining < _PROBE_ADMIT_SF_REMAINING_MAX


def _oracle_co_sf_admission_should_refuse(running_reqs, block_pool, block_size: int) -> bool:
    """SF-aware admission gate.

    The shortest_first preempt selects argmin(num_computed_tokens) — i.e. the
    freshest request — as its victim, regardless of expected_total. The vanilla
    oracle_co gate asks "does ANY running request look like a cheap victim?";
    that's a weak filter when paired with SF because SF will not evict the
    chosen "cheap" one — it evicts the freshest one, which often has long
    expected work ahead. This gate computes what SF would actually pick and
    refuses admission if that victim's expected remaining is large.
    """
    if not _ORACLE_LOOKUP and _PROBE_ADMISSION == "oracle_co_sf":
        _oracle_load_if_needed()
    free_blocks = block_pool.get_num_free_blocks()
    if free_blocks <= 0:
        return True
    projected = 0
    for req in running_reqs:
        expected = _oracle_resolve(req)
        remaining = expected - req.num_output_tokens
        if remaining > 0:
            projected += (remaining + block_size - 1) // block_size
    available = max(1, int(free_blocks * (1.0 - _PROBE_ADMISSION_MARGIN)))
    if projected <= available:
        return False
    if not running_reqs:
        return True
    sf_victim = min(running_reqs, key=lambda r: r.num_computed_tokens)
    sf_victim_remaining = max(0, _oracle_resolve(sf_victim) - sf_victim.num_output_tokens)
    return sf_victim_remaining > _PROBE_ADMIT_SF_REMAINING_MAX


def _entropy_co_sf_admission_should_refuse(running_reqs, block_pool, block_size: int) -> bool:
    """Deployment-grade twin of `_oracle_co_sf_admission_should_refuse`:
    same logic, but uses the S2 online entropy-spike length predictor
    instead of the offline oracle. No calibration required.

    Refuses admission only when:
      (a) projected remaining KV demand (sum over running) > free_blocks, AND
      (b) the SF would-be victim (argmin num_computed) has predicted remaining
          > PROBE_ADMIT_SF_REMAINING_MAX — i.e. SF preempt would target a request
          with lots of expected work, making the preempt expensive.
    """
    free_blocks = block_pool.get_num_free_blocks()
    if free_blocks <= 0:
        return True
    projected = 0
    for req in running_reqs:
        expected = _spike_estimate_E(req)
        remaining = expected - req.num_output_tokens
        if remaining > 0:
            projected += (remaining + block_size - 1) // block_size
    available = max(1, int(free_blocks * (1.0 - _PROBE_ADMISSION_MARGIN)))
    if projected <= available:
        return False
    if not running_reqs:
        return True
    sf_victim = min(running_reqs, key=lambda r: r.num_computed_tokens)
    sf_victim_remaining = max(0, _spike_estimate_E(sf_victim) - sf_victim.num_output_tokens)
    return sf_victim_remaining > _PROBE_ADMIT_SF_REMAINING_MAX


def _entropy_co_admission_should_refuse(running_reqs, block_pool, block_size: int) -> bool:
    """Entropy analog of `_oracle_co_admission_should_refuse` (generic gate,
    not SF-aware). Refuses admission only when:
      (a) projected remaining KV demand > free_blocks, AND
      (b) no acceptable victim exists — max entropy-co score across running
          requests is below ADMIT_THRESHOLD.
    """
    free_blocks = block_pool.get_num_free_blocks()
    if free_blocks <= 0:
        return True
    projected = 0
    for req in running_reqs:
        expected = _spike_estimate_E(req)
        remaining = expected - req.num_output_tokens
        if remaining > 0:
            projected += (remaining + block_size - 1) // block_size
    available = max(1, int(free_blocks * (1.0 - _PROBE_ADMISSION_MARGIN)))
    if projected <= available:
        return False
    if not running_reqs:
        return True
    best_score = max(_probe_g_kv_remaining_co_entropy(r) for r in running_reqs)
    return best_score < _PROBE_ADMIT_THRESHOLD


def _oracle_admission_overcommit(running_reqs, block_pool, block_size: int) -> bool:
    """Returns True if the projected remaining KV demand of the in-flight
    pool exceeds free blocks (minus a margin for the would-be arrival).

    For each running request, oracle says how many more tokens it'll generate;
    we convert to additional blocks and sum.  If sum > free*(1-margin), refuse
    admission this step — let the in-flight pool drain rather than admit a
    new request that will trigger preemption.
    """
    if not _ORACLE_LOOKUP and _PROBE_ADMISSION == "oracle":
        _oracle_load_if_needed()
    free_blocks = block_pool.get_num_free_blocks()
    if free_blocks <= 0:
        return True
    projected = 0
    for req in running_reqs:
        expected_total = _oracle_resolve(req)
        remaining = expected_total - req.num_output_tokens
        if remaining > 0:
            projected += (remaining + block_size - 1) // block_size
    available = max(1, int(free_blocks * (1.0 - _PROBE_ADMISSION_MARGIN)))
    return projected > available


def _probe_g_slope(request) -> float:  # type: ignore[return]
    """Victim score for probe_slope (P3): log-length × fast-EMA-entropy + slope bonus.

    log(1+len) de-biases long sequences that have already entered low-entropy
    (converging) phase.  fast_ema_h (α=0.3) detects the entropy cliff in ~7
    tokens.  Slope bonus λ·max(fast_h−slow_h,0) boosts sequences still
    actively expanding into exploration.
    """
    fast_h = request.probe_ema_h_fast
    slow_h = request.probe_ema_h
    base = (math.log1p(request.num_computed_tokens) * fast_h
            / (_PROBE_H_MAX - fast_h + _PROBE_EPS))
    slope_bonus = _PROBE_SLOPE_LAMBDA * max(fast_h - slow_h, 0.0)
    return base + slope_bonus


def _probe_g_peos(request) -> float:  # type: ignore[return]
    """P3 ablation using P(EOS) instead of token entropy (A1).

    g_peos = log(1+len) × (1 - fast_p_eos) / (fast_p_eos + ε)
           + λ × max(slow_p_eos - fast_p_eos, 0)

    High P(EOS) → model wants to stop → low remaining growth → low score.
    Slope bonus fires when fast_p_eos drops below slow (p_eos declining ≡
    entropy rising), identical semantics to P3's entropy slope penalty.
    """
    fast_p = request.probe_ema_p_eos_fast
    slow_p = request.probe_ema_p_eos
    base = (math.log1p(request.num_computed_tokens) * (1.0 - fast_p)
            / (fast_p + _PROBE_EPS))
    slope_bonus = _PROBE_SLOPE_LAMBDA * max(slow_p - fast_p, 0.0)
    return base + slope_bonus


def _probe_g_slope_b(request) -> float:
    """P3b: identical scorer to P3; improvement is warm EMA init in Request.__init__."""
    return _probe_g_slope(request)


def _probe_g_slope_c(request) -> float:
    """P3c: P3 with pressure-adaptive λ — slope bonus scales to zero under extreme KV load.

    λ_eff = λ at kv_util ≤ LAM_LOW (0.80); 0 at kv_util ≥ LAM_HIGH (0.95).
    Prevents P3's entropy-protection from causing head-of-line blocking above N*.
    Reads _probe_kv_pressure, updated each schedule() step.
    """
    fast_h = request.probe_ema_h_fast
    slow_h = request.probe_ema_h
    base = (math.log1p(request.num_computed_tokens) * fast_h
            / (_PROBE_H_MAX - fast_h + _PROBE_EPS))
    lam_eff = _PROBE_SLOPE_LAMBDA * max(
        0.0,
        1.0 - (_probe_kv_pressure - _PROBE_ADAPTIVE_LAM_LOW)
              / (_PROBE_ADAPTIVE_LAM_HIGH - _PROBE_ADAPTIVE_LAM_LOW),
    )
    return base + lam_eff * max(fast_h - slow_h, 0.0)


def _probe_g_slope_e(request) -> float:
    """P3e: symmetric slope — bonus for rising entropy, discount for falling entropy.

    Adds -γ·max(slow_h - fast_h, 0) to make actively-converging sequences
    easier to evict, symmetric with the existing exploration bonus.
    """
    fast_h = request.probe_ema_h_fast
    slow_h = request.probe_ema_h
    base = (math.log1p(request.num_computed_tokens) * fast_h
            / (_PROBE_H_MAX - fast_h + _PROBE_EPS))
    slope_bonus = _PROBE_SLOPE_LAMBDA * max(fast_h - slow_h, 0.0)
    convergence_discount = _PROBE_SLOPE_GAMMA * max(slow_h - fast_h, 0.0)
    return base + slope_bonus - convergence_discount


def _probe_g_slope_f(request) -> float:
    """P3f: entropy + P(EOS) fusion — P(EOS) used as a convergence detector on top of entropy.

    Subtracts δ·fast_ema_p_eos from the entropy score. High P(EOS) signals
    imminent termination; reduces victim score independent of entropy level.
    Leverages the P(EOS) signal already extracted for the A1 ablation.
    """
    fast_h = request.probe_ema_h_fast
    slow_h = request.probe_ema_h
    base = (math.log1p(request.num_computed_tokens) * fast_h
            / (_PROBE_H_MAX - fast_h + _PROBE_EPS))
    slope_bonus = _PROBE_SLOPE_LAMBDA * max(fast_h - slow_h, 0.0)
    peos_discount = _PROBE_PEOS_DELTA * request.probe_ema_p_eos_fast
    return base + slope_bonus - peos_discount


def _probe_g_slope_g(request) -> float:
    """P3g: peak-entropy tracking — discounts sequences that have fallen below their own peak.

    Multiplies base+slope by exp(-β · peak_gap / H_max) where peak_gap is how
    far fast_h has dropped below the sequence's maximum observed entropy.
    Sequences still near their peak are fully protected; sequences well past
    their peak (answer-writing phase) are heavily discounted.
    """
    fast_h = request.probe_ema_h_fast
    slow_h = request.probe_ema_h
    peak_h = request.probe_peak_h
    base = (math.log1p(request.num_computed_tokens) * fast_h
            / (_PROBE_H_MAX - fast_h + _PROBE_EPS))
    slope_bonus = _PROBE_SLOPE_LAMBDA * max(fast_h - slow_h, 0.0)
    peak_gap = max(peak_h - fast_h, 0.0)
    peak_discount = math.exp(-_PROBE_PEAK_BETA * peak_gap / _PROBE_H_MAX)
    return (base + slope_bonus) * peak_discount


def _probe_g_slope_e2(request) -> float:
    """P3e-γ0.2: symmetric slope with lighter convergence discount (γ=0.2 vs 0.5).

    Reduces N=84-88 clogging observed with γ=0.5 while retaining the N=76-80
    signal improvement. Hardcoded to avoid per-run env var contamination.
    """
    fast_h = request.probe_ema_h_fast
    slow_h = request.probe_ema_h
    base = (math.log1p(request.num_computed_tokens) * fast_h
            / (_PROBE_H_MAX - fast_h + _PROBE_EPS))
    slope_bonus = _PROBE_SLOPE_LAMBDA * max(fast_h - slow_h, 0.0)
    convergence_discount = 0.2 * max(slow_h - fast_h, 0.0)
    return base + slope_bonus - convergence_discount


def _probe_g_slope_e1(request) -> float:
    """P3e-γ0.1: symmetric slope with minimal convergence discount (γ=0.1).

    Most conservative tuning — intended to preserve P3's N=84-88 behaviour
    while testing whether any convergence signal benefit survives at N=76-80.
    """
    fast_h = request.probe_ema_h_fast
    slow_h = request.probe_ema_h
    base = (math.log1p(request.num_computed_tokens) * fast_h
            / (_PROBE_H_MAX - fast_h + _PROBE_EPS))
    slope_bonus = _PROBE_SLOPE_LAMBDA * max(fast_h - slow_h, 0.0)
    convergence_discount = 0.1 * max(slow_h - fast_h, 0.0)
    return base + slope_bonus - convergence_discount


def _probe_g_slope_c2(request) -> float:
    """P3c-LAM88: pressure-adaptive λ with higher activation threshold (LAM_LOW=0.88).

    Preserves full λ until KV utilization reaches 88% (vs 80% in P3c), keeping
    P3's entropy discrimination active deeper into the pressure zone. λ reaches
    zero at 95% util (same high end as P3c).
    """
    fast_h = request.probe_ema_h_fast
    slow_h = request.probe_ema_h
    base = (math.log1p(request.num_computed_tokens) * fast_h
            / (_PROBE_H_MAX - fast_h + _PROBE_EPS))
    lam_eff = _PROBE_SLOPE_LAMBDA * max(
        0.0,
        1.0 - (_probe_kv_pressure - 0.88) / (_PROBE_ADAPTIVE_LAM_HIGH - 0.88),
    )
    return base + lam_eff * max(fast_h - slow_h, 0.0)


from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsReader,
)
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    EncoderDecoderCacheManager,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.core.sched.request_queue import (
    RequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig
from vllm.v1.metrics.perf import ModelMetrics, PerfStats
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)


class Scheduler(SchedulerInterface):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.observability_config = vllm_config.observability_config
        self.kv_metrics_collector: KVCacheMetricsCollector | None = None
        if self.observability_config.kv_cache_metrics:
            self.kv_metrics_collector = KVCacheMetricsCollector(
                self.observability_config.kv_cache_metrics_sample,
            )
        self.structured_output_manager = structured_output_manager
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.finished_req_ids_dict: dict[int, set[str]] | None = (
            defaultdict(set) if include_finished_set else None
        )
        self.prev_step_scheduled_req_ids: set[str] = set()

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = (
            self.scheduler_config.max_num_scheduled_tokens
            if self.scheduler_config.max_num_scheduled_tokens
            else self.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events
        )

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        self.connector_prefix_cache_stats: PrefixCacheStats | None = None
        self.recompute_kv_load_failures = True
        if self.vllm_config.kv_transfer_config is not None:
            assert not self.is_encoder_decoder, (
                "Encoder-decoder models are not currently supported with KV connectors"
            )
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config,
                role=KVConnectorRole.SCHEDULER,
                kv_cache_config=self.kv_cache_config,
            )
            if self.log_stats:
                self.connector_prefix_cache_stats = PrefixCacheStats()
            kv_load_failure_policy = (
                self.vllm_config.kv_transfer_config.kv_load_failure_policy
            )
            self.recompute_kv_load_failures = kv_load_failure_policy == "recompute"

        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_index,
        )
        self.ec_connector = None
        if self.vllm_config.ec_transfer_config is not None:
            self.ec_connector = ECConnectorFactory.create_connector(
                config=self.vllm_config, role=ECConnectorRole.SCHEDULER
            )

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = block_size
        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Scheduling policy
        try:
            self.policy = SchedulingPolicy(self.scheduler_config.policy)
        except ValueError as e:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}"
            ) from e
        # Priority queues for requests.
        self.waiting = create_request_queue(self.policy)
        # requests skipped in waiting flow due async deps or constraints.
        self.skipped_waiting = create_request_queue(self.policy)
        self.running: list[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # Counter for requests waiting for streaming input. Used to calculate
        # number of unfinished requests
        self.num_waiting_for_streaming_input: int = 0

        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()
        self.failed_recving_kv_req_ids: set[str] = set()

        # PROBE proactive offload: requests in PRE_OFFLOADING state.
        # GPU blocks are freed to the draining pool immediately; the
        # offloading connector is streaming KV to CPU.  The draining blocks
        # are released to the free queue once finished_sending is received.
        self._probe_pre_offloading: dict[str, Request] = {}
        # Reactive-offload (PROBE_REACTIVE_OFFLOAD): set of req_ids that
        # entered PRE_OFFLOADING this step.  Drained into
        # scheduler_output.proactive_preempted_req_ids and cleared each step.
        self._step_proactive_preempts: set[str] = set()
        # Block IDs currently in draining pool (per proactive pre-offload).
        self._probe_drain_block_ids: dict[str, list[int]] = {}

        # Encoder-related.
        # Calculate encoder cache size if applicable
        supports_mm_inputs = mm_registry.supports_multimodal_inputs(
            vllm_config.model_config
        )
        mm_budget = (
            MultiModalBudget(vllm_config, mm_registry) if supports_mm_inputs else None
        )

        # NOTE: Text-only encoder-decoder models are implemented as
        # multi-modal models for convenience
        # Example: https://github.com/vllm-project/bart-plugin
        if self.is_encoder_decoder:
            assert mm_budget and len(mm_budget.mm_max_toks_per_item) <= 1, (
                "Encoder-decoder models are expected to implement the "
                "multimodal interface with at most one modality."
            )

        self.max_num_encoder_input_tokens = (
            mm_budget.encoder_compute_budget if mm_budget else 0
        )
        encoder_cache_size = mm_budget.encoder_cache_size if mm_budget else 0
        self.encoder_cache_manager = (
            EncoderDecoderCacheManager(cache_size=encoder_cache_size)
            if self.is_encoder_decoder
            else EncoderCacheManager(cache_size=encoder_cache_size)
        )

        speculative_config = vllm_config.speculative_config
        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens
            if speculative_config.uses_draft_model():
                self.num_lookahead_tokens = self.num_spec_tokens

        # Create the KV cache manager.
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=self.pcp_world_size,
            hash_block_size=self.block_size,
            metrics_collector=self.kv_metrics_collector,
        )
        # Bind GPU block pool to the KV connector. This must happen after
        # kv_cache_manager is constructed so block_pool is available.
        if self.connector is not None and hasattr(
            self.connector, "bind_gpu_block_pool"
        ):
            self.connector.bind_gpu_block_pool(self.kv_cache_manager.block_pool)

        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        self.use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER
        self.scheduler_reserve_full_isl = (
            self.scheduler_config.scheduler_reserve_full_isl
        )

        self.has_mamba_layers = kv_cache_config.has_mamba_layers
        self.needs_kv_cache_zeroing = kv_cache_config.needs_kv_cache_zeroing
        self.need_mamba_block_aligned_split = (
            self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align"
        )
        self.perf_metrics: ModelMetrics | None = None
        if self.log_stats and vllm_config.observability_config.enable_mfu_metrics:
            self.perf_metrics = ModelMetrics(vllm_config)

        if self.vllm_config.model_config.enable_return_routed_experts:
            assert self.dcp_world_size == 1 and self.pcp_world_size == 1, (
                "enable_return_routed_experts does not support context parallelism "
                "(dcp_world_size > 1 or pcp_world_size > 1)"
            )

            self.routed_experts_reader = RoutedExpertsReader.create()

            assert len(kv_cache_config.kv_cache_groups) > 0, (
                "enable_return_routed_experts requires at least one kv cache group"
            )
            # Find the attention group for routed experts indexing.
            self.routed_experts_attn_gid = 0
            for gid, group in enumerate(kv_cache_config.kv_cache_groups):
                if isinstance(group.kv_cache_spec, AttentionSpec):
                    self.routed_experts_attn_gid = gid
                    break
            min_block_size = min(
                [
                    group.kv_cache_spec.block_size
                    for group in kv_cache_config.kv_cache_groups
                ]
            )
            num_groups = len(kv_cache_config.kv_cache_groups)
            self.max_num_kv_tokens = (
                kv_cache_config.num_blocks // num_groups
            ) * min_block_size
            dcp_size = self.vllm_config.parallel_config.decode_context_parallel_size
            pcp_size = self.vllm_config.parallel_config.prefill_context_parallel_size
            if pcp_size * dcp_size > 1:
                self.max_num_kv_tokens *= pcp_size * dcp_size

            self.routed_experts_reader.attach_buffer(
                max_num_kv_tokens=self.max_num_kv_tokens,
                vllm_config=self.vllm_config,
            )

        self._pause_state: PauseState = PauseState.UNPAUSED

    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        assert num_external_computed_tokens == 0, (
            "External KV connector is not verified yet"
        )
        num_computed_tokens = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        # Perform block-aligned splitting at prefill phase, including:
        # * non-resumed requests: num_computed_tokens < num_prompt_tokens + 0
        # * resumed requests: num_computed_tokens < (
        #                       num_prompt_tokens + num_output_tokens
        #                     )
        # NOTE: Use `request.num_tokens - 1` to bypass normal decoding.
        if num_computed_tokens < max(request.num_prompt_tokens, request.num_tokens - 1):
            # To enable block-aligned caching of the Mamba state, `num_new_tokens`
            # must be a multiple of `block_size`.
            # As an exception, if `num_new_tokens` is less than `block_size`, the
            # state is simply not cached, requiring no special handling.
            # Additionally, when Eagle mode is enabled, FullAttn prunes the last
            # matching block. To prevent this from causing a Mamba cache miss, the
            # last chunk must be not smaller than `block_size`.
            block_size = self.cache_config.block_size
            last_cache_position = request.num_tokens - request.num_tokens % block_size
            # eagle prune
            if self.use_eagle:
                last_cache_position = max(last_cache_position - block_size, 0)
            num_computed_tokens_after_sched = num_computed_tokens + num_new_tokens
            if num_computed_tokens_after_sched < last_cache_position:
                # align to block_size
                num_new_tokens = num_new_tokens // block_size * block_size
            elif (
                num_computed_tokens
                < last_cache_position
                < num_computed_tokens_after_sched
            ):
                # force to cache the last chunk
                num_new_tokens = last_cache_position - num_computed_tokens
            else:
                # prefill the last few tokens
                pass
        return num_new_tokens

    def schedule(self) -> SchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            # Do not schedule any requests when paused.
            token_budget = 0

        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()

        self.kv_cache_manager.new_step_starts()

        # PROBE P3c / P4: update KV-pressure globals every step.
        if _PROBE_VICTIM in ("probe_slope_c", "probe_slope_c2", "probe_slope_ring"):
            global _probe_kv_pressure, _probe_kv_ema, _probe_ring_active
            _bp_tot = self.kv_cache_manager.block_pool.num_gpu_blocks
            if _bp_tot > 0:
                _bp_free = self.kv_cache_manager.block_pool.get_num_free_blocks()
                _inst_p = 1.0 - _bp_free / _bp_tot
                _probe_kv_pressure = _inst_p  # P3c: instantaneous utilization
                if _PROBE_VICTIM == "probe_slope_ring":
                    _probe_kv_ema = (_PROBE_KV_EMA_ALPHA * _inst_p
                                     + (1.0 - _PROBE_KV_EMA_ALPHA) * _probe_kv_ema)
            # P4 hysteresis gate.
            if _PROBE_VICTIM == "probe_slope_ring":
                if not _probe_ring_active and _probe_kv_ema >= _PROBE_SLOPE_RING_ACTIVATE:
                    _probe_ring_active = True
                elif _probe_ring_active and _probe_kv_ema < _PROBE_SLOPE_RING_DEACTIVATE:
                    _probe_ring_active = False

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            if (
                request.num_output_placeholders > 0
                # This is (num_computed_tokens + 1) - (num_output_placeholders - 1).
                # Since output placeholders are also included in the computed tokens
                # count, we subtract (num_output_placeholders - 1) to remove any draft
                # tokens, so that we can be sure no further steps are needed even if
                # they are all rejected.
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                # Async scheduling: Avoid scheduling an extra step when we are sure that
                # the previous step has reached request.max_tokens. We don't schedule
                # partial draft tokens since this prevents uniform decode optimizations.
                req_index += 1
                continue

            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens
            )

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            external_load_encoder_input: list[int] = []
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                    external_load_encoder_input,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                    shift_computed_tokens=1 if self.use_eagle else 0,
                )

            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # 4. Insufficient budget for a block-aligned chunk in hybrid
                #    models with mamba cache mode \"align\".
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            # Schedule newly needed KV blocks for the request.
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                    if new_blocks is not None:
                        # The request can be scheduled.
                        break

                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens.pop(preempted_req_id)
                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req_id, None
                            )
                            if preempted_encoder_inputs:
                                # Restore encoder compute budget if the preempted
                                # request had encoder inputs scheduled in this step.
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                            req_index -= 1
                    elif _PROBE_VICTIM in (
                        "oracle", "oracle_co", "oracle_co_adapt", "oracle_co_pred",
                        "oracle_smart_lifo",
                        "longest_remaining", "kv_remaining_product",
                        "kv_remaining_co", "kv_remaining_co_entropy",
                        "kv_remaining_swapaware_entropy",
                        "entropy_remaining",
                        "entropy_least_remaining", "oracle_least_remaining",
                        "entropy_kv_least_remaining_co",
                        "oracle_kv_least_remaining_co",
                        "age_weighted_oracle", "longest_decoded",
                    ):
                        if _PROBE_VICTIM == "oracle":
                            score_fn = _probe_g_oracle
                        elif _PROBE_VICTIM == "oracle_co_adapt":
                            global _CURRENT_ADAPTIVE_BETA
                            _CURRENT_ADAPTIVE_BETA = _adaptive_beta(
                                _kv_utilization(self.kv_cache_manager.block_pool))
                            score_fn = _probe_g_oracle_co_adapt
                        elif _PROBE_VICTIM == "oracle_smart_lifo":
                            score_fn = _probe_g_smart_lifo
                        elif _PROBE_VICTIM == "longest_remaining":
                            score_fn = _probe_g_longest_remaining
                        elif _PROBE_VICTIM == "kv_remaining_product":
                            score_fn = _probe_g_kv_remaining_product
                        elif _PROBE_VICTIM == "kv_remaining_co":
                            score_fn = _probe_g_kv_remaining_co
                        elif _PROBE_VICTIM == "kv_remaining_co_entropy":
                            score_fn = _probe_g_kv_remaining_co_entropy
                        elif _PROBE_VICTIM == "kv_remaining_swapaware_entropy":
                            score_fn = _probe_g_kv_remaining_swapaware_entropy
                        elif _PROBE_VICTIM == "entropy_remaining":
                            score_fn = _probe_g_entropy_remaining
                        elif _PROBE_VICTIM == "entropy_least_remaining":
                            score_fn = _probe_g_entropy_least_remaining
                        elif _PROBE_VICTIM == "oracle_least_remaining":
                            score_fn = _probe_g_oracle_least_remaining
                        elif _PROBE_VICTIM == "entropy_kv_least_remaining_co":
                            score_fn = _probe_g_kv_least_remaining_co_entropy
                        elif _PROBE_VICTIM == "oracle_kv_least_remaining_co":
                            score_fn = _probe_g_kv_least_remaining_co_oracle
                        elif _PROBE_VICTIM == "age_weighted_oracle":
                            score_fn = _probe_g_age_weighted
                        elif _PROBE_VICTIM == "longest_decoded":
                            score_fn = _probe_g_longest_decoded
                        else:
                            # oracle_co and oracle_co_pred both use the static-β cost-aware score
                            score_fn = _probe_g_oracle_co
                        preempted_req = max(_eligible_for_preempt(self.running), key=score_fn)
                        _oracle_log_decision(preempted_req, self.running)
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens.pop(preempted_req_id)
                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req_id, None
                            )
                            if preempted_encoder_inputs:
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                            req_index -= 1
                    elif _PROBE_VICTIM == "probe_spike":
                        preempted_req = max(_eligible_for_preempt(self.running), key=_probe_g_spike)
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens.pop(preempted_req_id)
                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req_id, None
                            )
                            if preempted_encoder_inputs:
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                            req_index -= 1
                    elif _PROBE_VICTIM in (
                        "probe_slope", "probe_slope_peos",
                        "probe_slope_b", "probe_slope_c", "probe_slope_e",
                        "probe_slope_f", "probe_slope_g",
                        "probe_slope_e2", "probe_slope_e1", "probe_slope_c2",
                    ):
                        # PROBE P3 family: entropy-based victim selection.
                        _score_fn = {
                            "probe_slope_peos": _probe_g_peos,
                            "probe_slope_b": _probe_g_slope_b,
                            "probe_slope_c": _probe_g_slope_c,
                            "probe_slope_e": _probe_g_slope_e,
                            "probe_slope_f": _probe_g_slope_f,
                            "probe_slope_g": _probe_g_slope_g,
                            "probe_slope_e2": _probe_g_slope_e2,
                            "probe_slope_e1": _probe_g_slope_e1,
                            "probe_slope_c2": _probe_g_slope_c2,
                        }.get(_PROBE_VICTIM, _probe_g_slope)
                        preempted_req = max(
                            self.running,
                            key=_score_fn,
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens.pop(preempted_req_id)
                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req_id, None
                            )
                            if preempted_encoder_inputs:
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                            req_index -= 1
                    elif _PROBE_VICTIM == "probe_slope_ring":
                        # PROBE P4: hysteresis-gated ring with g_slope + sustained pressure.
                        #
                        # _probe_ring_active latches on at EMA ≥ 0.90 and off at < 0.85.
                        # Below the band → pure g_slope (P3 behaviour); ring still books
                        # samples silently so it warms up before saturation begins.
                        # Inside the band → state is sticky (no flip-flopping).
                        assert _PROBE_SLOPE_RING is not None
                        if not _probe_ring_active:
                            # Bookkeeping mode: P3 behaviour, ring is passive.
                            preempted_req = max(
                                self.running,
                                key=_probe_g_slope,
                            )
                        else:
                            # Active mode: ring-calibrated score with EMA pressure.
                            preempted_req = max(
                                self.running,
                                key=lambda r: _PROBE_SLOPE_RING.query(  # type: ignore[union-attr]
                                    r.num_computed_tokens,
                                    _probe_g_slope(r),
                                    memory_pressure=_probe_kv_ema,
                                ),
                            )
                        _probe_slope_ring_pending[preempted_req.request_id] = (
                            float(preempted_req.num_computed_tokens),
                            _probe_g_slope(preempted_req),
                            preempted_req.num_output_tokens,
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens.pop(preempted_req_id)
                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req_id, None
                            )
                            if preempted_encoder_inputs:
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                            req_index -= 1
                    elif _PROBE_VICTIM == "shortest_first":
                        # Baseline: InferMax (arXiv 2411.07447) — evict the
                        # request with the smallest current KV cache to
                        # minimise recompute cost on resumption.
                        preempted_req = min(
                            self.running,
                            key=lambda r: r.num_computed_tokens,
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens.pop(preempted_req_id)
                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req_id, None
                            )
                            if preempted_encoder_inputs:
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                            req_index -= 1
                    elif _PROBE_VICTIM == "random":
                        # Control: uniform random victim, seeded RNG (see
                        # _PROBE_VICTIM_SEED). Separates "any non-pathological
                        # pick works" from genuine predictor value.
                        preempted_req = _victim_rng.choice(self.running)
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens.pop(preempted_req_id)
                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req_id, None
                            )
                            if preempted_encoder_inputs:
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                            req_index -= 1
                    elif _PROBE_VICTIM == "sf_oracle_tiebreak":
                        # SF primary (minimize recompute cost) with oracle
                        # remaining-decode as a tie-breaker (prefer evicting
                        # the cheap-victim with the longest tail).
                        preempted_req = min(
                            self.running,
                            key=_sf_oracle_tiebreak_key,
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens.pop(preempted_req_id)
                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req_id, None
                            )
                            if preempted_encoder_inputs:
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                            req_index -= 1
                    elif _PROBE_VICTIM == "input_length":
                        # Baseline: FastServe (arXiv 2305.05920) — evict the
                        # request with the longest input prompt, using prompt
                        # length as a static proxy for total output length.
                        preempted_req = max(
                            self.running,
                            key=lambda r: r.num_prompt_tokens,
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens.pop(preempted_req_id)
                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req_id, None
                            )
                            if preempted_encoder_inputs:
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                            req_index -= 1
                    elif _PROBE_VICTIM == "least_decoded":
                        # Baseline B2: least-decoded-first — evict the request
                        # with the fewest generated tokens, isolating decode
                        # progress from prompt length.
                        preempted_req = min(
                            self.running,
                            key=lambda r: r.num_output_tokens,
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens.pop(preempted_req_id)
                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req_id, None
                            )
                            if preempted_encoder_inputs:
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                            req_index -= 1
                    elif _PROBE_VICTIM == "largest_kv":
                        # Baseline B3: largest-KV-first — evict the request
                        # with the most computed tokens to free the maximum
                        # KV-cache memory per preemption event.
                        preempted_req = max(
                            self.running,
                            key=lambda r: r.num_computed_tokens,
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens.pop(preempted_req_id)
                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req_id, None
                            )
                            if preempted_encoder_inputs:
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                            req_index -= 1
                    elif _PROBE_VICTIM == "benefit_cost":
                        # Baseline B5: DBMS-inspired benefit/cost ratio.
                        # benefit = KV blocks freed ∝ num_computed_tokens
                        # cost    = recompute cost  ∝ num_prompt_tokens
                        # Evict the request that maximises KV freed per unit
                        # of recompute cost; avoids the degenerate case where
                        # benefit and cost are both proportional to KV size.
                        preempted_req = max(
                            self.running,
                            key=lambda r: r.num_computed_tokens / max(r.num_prompt_tokens, 1),
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens.pop(preempted_req_id)
                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req_id, None
                            )
                            if preempted_encoder_inputs:
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                            req_index -= 1
                    else:
                        # Default FCFS: evict most recently admitted request.
                        # When PROBE_DEFAULT_LIFO_SHIELD=1 (default_ab recipe),
                        # preserve LIFO order but skip shielded requests.
                        if _PROBE_DEFAULT_LIFO_SHIELD:
                            eligible_set = set(id(r) for r in _eligible_for_preempt(self.running))
                            preempted_req = None
                            for r in reversed(self.running):
                                if id(r) in eligible_set:
                                    preempted_req = r
                                    break
                            if preempted_req is None:
                                preempted_req = self.running.pop()
                            else:
                                self.running.remove(preempted_req)
                        else:
                            preempted_req = self.running.pop()

                    # PROBE M2: log this preemption event.
                    _ppl.log_preemption(
                        req_id=preempted_req.request_id,
                        num_computed=preempted_req.num_computed_tokens,
                        num_preemptions_after=preempted_req.num_preemptions + 1,
                        cascade_pos=len(preempted_reqs),
                        timestamp=scheduled_timestamp,
                    )

                    self._preempt_request(preempted_req, scheduled_timestamp)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt. Cannot schedule this request.
                        break

            if new_blocks is None:
                # Cannot schedule this request.
                break

            # Schedule the request.
            scheduled_running_reqs.append(request)
            request_id = request.request_id
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (
                    num_new_tokens
                    + request.num_computed_tokens
                    - request.num_tokens
                    - request.num_output_placeholders
                )
                if num_scheduled_spec_tokens > 0:
                    spec_token_ids = request.spec_token_ids
                    if len(spec_token_ids) > num_scheduled_spec_tokens:
                        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                    scheduled_spec_decode_tokens[request.request_id] = spec_token_ids

                # New spec tokens will be set in `update_draft_token_ids` before the
                # next step when applicable.
                request.spec_token_ids = []

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)
                encoder_compute_budget = new_encoder_compute_budget
            if external_load_encoder_input:
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # PROBE M2: advance cascade counter once per schedule() call.
        if preempted_reqs:
            _ppl.advance_cascade()

        # Next, schedule the WAITING requests.
        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
            # PROBE: oracle admission gate. If the in-flight pool's projected
            # remaining KV demand already exceeds free capacity, refuse to
            # admit anything this step — let existing requests drain first.
            # This prevents the recompute-thrash that pure preemption can't.
            if _PROBE_ADMISSION == "oracle":
                admission_blocked = _oracle_admission_overcommit(
                    self.running,
                    self.kv_cache_manager.block_pool,
                    self.block_size,
                )
            elif _PROBE_ADMISSION == "oracle_co":
                admission_blocked = _oracle_co_admission_should_refuse(
                    self.running,
                    self.kv_cache_manager.block_pool,
                    self.block_size,
                )
            elif _PROBE_ADMISSION == "oracle_co_sf":
                admission_blocked = _oracle_co_sf_admission_should_refuse(
                    self.running,
                    self.kv_cache_manager.block_pool,
                    self.block_size,
                )
            elif _PROBE_ADMISSION == "entropy_co_sf":
                admission_blocked = _entropy_co_sf_admission_should_refuse(
                    self.running,
                    self.kv_cache_manager.block_pool,
                    self.block_size,
                )
            elif _PROBE_ADMISSION == "entropy_co":
                admission_blocked = _entropy_co_admission_should_refuse(
                    self.running,
                    self.kv_cache_manager.block_pool,
                    self.block_size,
                )
            elif _PROBE_ADMISSION == "oracle_co_lifo":
                admission_blocked = _oracle_co_lifo_admission_should_refuse(
                    self.running,
                    self.kv_cache_manager.block_pool,
                    self.block_size,
                )
            elif _PROBE_ADMISSION == "oracle_co_il":
                admission_blocked = _oracle_co_il_admission_should_refuse(
                    self.running,
                    self.kv_cache_manager.block_pool,
                    self.block_size,
                )
            elif _PROBE_ADMISSION == "oracle_co_lr":
                admission_blocked = _oracle_co_lr_admission_should_refuse(
                    self.running,
                    self.kv_cache_manager.block_pool,
                    self.block_size,
                )
            elif _PROBE_ADMISSION == "oracle_co_adapt":
                admission_blocked = _oracle_co_adapt_admission_should_refuse(
                    self.running,
                    self.kv_cache_manager.block_pool,
                    self.block_size,
                )
            elif _PROBE_ADMISSION == "oracle_co_pred":
                admission_blocked = _oracle_pred_admission_should_refuse(
                    self.running,
                    self.kv_cache_manager.block_pool,
                    self.block_size,
                )
            elif _PROBE_ADMISSION == "oracle_reservation":
                admission_blocked = _oracle_reservation_admission_should_refuse(
                    self.running,
                    self.kv_cache_manager.block_pool,
                    self.block_size,
                )
            else:
                admission_blocked = False

            step_skipped_waiting = create_request_queue(self.policy)

            while (
                not admission_blocked
                and (self.waiting or self.skipped_waiting)
                and token_budget > 0
            ):
                if len(self.running) == self.max_num_running_reqs:
                    break

                request_queue = self._select_waiting_queue_for_scheduling()
                assert request_queue is not None

                request = request_queue.peek_request()
                request_id = request.request_id

                # try to promote blocked statuses while traversing skipped queue.
                if self._is_blocked_waiting_status(
                    request.status
                ) and not self._try_promote_blocked_waiting_request(request):
                    if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request_id,
                        )
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                # Check that adding the request still respects the max_loras
                # constraint.
                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                    )
                ):
                    # Scheduling would exceed max_loras, skip.
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                # PROBE Method 5 — two-level admission: defer long arrivals
                # if long-lane is at capacity. Shorts always admit.
                if (_PROBE_ADMISSION == "oracle_two_level"
                        and _two_level_should_skip_arrival(request, self.running)):
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False
                connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    # Get locally-cached tokens.
                    new_computed_blocks, num_new_local_computed_tokens = (
                        self.kv_cache_manager.get_computed_blocks(request)
                    )

                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        ext_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens
                            )
                        )

                        if ext_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue

                        request.num_external_computed_tokens = ext_tokens
                        num_external_computed_tokens = ext_tokens

                        connector_prefix_cache_queries = (
                            request.num_tokens - num_new_local_computed_tokens
                        )
                        connector_prefix_cache_hits = num_external_computed_tokens

                    # Total computed tokens (local + external).
                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens
                    )
                    assert num_computed_tokens <= request.num_tokens
                else:
                    # KVTransfer: WAITING reqs have num_computed_tokens > 0
                    # after async KV recvs are completed.
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None
                external_load_encoder_input = []
                new_encoder_compute_budget = encoder_compute_budget

                if load_kv_async:
                    # KVTransfer: loading remote KV, do not allocate for new work.
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                else:
                    # Number of tokens to be scheduled.
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if (
                        not self.scheduler_config.enable_chunked_prefill
                        and num_new_tokens > token_budget
                    ):
                        # If chunked_prefill is disabled,
                        # we can stop the scheduling here.
                        break

                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                    # Schedule encoder inputs.
                    if request.has_encoder_inputs:
                        (
                            encoder_inputs_to_schedule,
                            num_new_tokens,
                            new_encoder_compute_budget,
                            external_load_encoder_input,
                        ) = self._try_schedule_encoder_inputs(
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            encoder_compute_budget,
                            shift_computed_tokens=1 if self.use_eagle else 0,
                        )
                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            break

                if self.need_mamba_block_aligned_split:
                    num_new_tokens = self._mamba_block_aligned_split(
                        request,
                        num_new_tokens,
                        num_new_local_computed_tokens,
                        num_external_computed_tokens,
                    )
                    if num_new_tokens == 0:
                        break

                # Handles an edge case when P/D Disaggregation
                # is used with Spec Decoding where an
                # extra block gets allocated which
                # creates a mismatch between the number
                # of local and remote blocks.
                effective_lookahead_tokens = (
                    0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens
                )

                # Determine if we need to allocate cross-attention blocks.
                num_encoder_tokens = 0
                if (
                    self.is_encoder_decoder
                    and request.has_encoder_inputs
                    and encoder_inputs_to_schedule
                ):
                    num_encoder_tokens = sum(
                        request.get_num_encoder_embeds(i)
                        for i in encoder_inputs_to_schedule
                    )

                if (
                    self.scheduler_reserve_full_isl
                    and not self.kv_cache_manager.can_fit_full_sequence(
                        request,
                        num_new_computed_tokens=num_new_local_computed_tokens,
                        new_computed_blocks=new_computed_blocks,
                        num_external_computed_tokens=num_external_computed_tokens,
                        num_encoder_tokens=num_encoder_tokens,
                    )
                ):
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_new_computed_tokens=num_new_local_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    num_external_computed_tokens=num_external_computed_tokens,
                    delay_cache_blocks=load_kv_async,
                    num_encoder_tokens=num_encoder_tokens,
                )

                if new_blocks is None:
                    # The request cannot be scheduled.

                    # NOTE: we need to untouch the request from the encode cache
                    # manager
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        self.kv_cache_manager.get_blocks(request_id),
                        num_external_computed_tokens,
                    )
                    if (
                        self.connector_prefix_cache_stats is not None
                        and connector_prefix_cache_queries != 0
                    ):
                        self.connector_prefix_cache_stats.record(
                            num_tokens=connector_prefix_cache_queries,
                            num_hits=connector_prefix_cache_hits,
                            preempted=request.num_preemptions > 0,
                        )

                request = request_queue.pop_request()
                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    step_skipped_waiting.prepend_request(request)
                    # Set num_computed_tokens even though KVs are not yet loaded.
                    # request.num_computed_tokens will not be used anywhere until
                    # the request finished the KV transfer.
                    #
                    # If a transfer error is reported by the connector,
                    # request.num_computed_tokens will be re-set accordingly in
                    # _update_requests_with_invalid_blocks.
                    #
                    # When the transfer is finished, either successfully or not,
                    # request.num_computed_tokens will correctly reflect the number
                    # of computed tokens.
                    # _update_waiting_for_remote_kv will then cache
                    # only the successfully loaded tokens.
                    request.num_computed_tokens = num_computed_tokens
                    continue

                self.running.append(request)
                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )
                # PROBE lifecycle: log first-admit and resume-after-preempt
                # events. Combined with preempt.jsonl (when each request leaves
                # running) this gives full request lifecycle for bottleneck
                # analysis (queue wait + active session durations).
                _is_resume = request.status == RequestStatus.PREEMPTED
                _ppl.log_admit(
                    req_id=request.request_id,
                    prompt_tokens=len(request.prompt_token_ids),
                    timestamp=scheduled_timestamp,
                    is_resume=_is_resume,
                    num_computed_at_admit=request.num_computed_tokens,
                )
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
                    request_id
                )
                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # Count the number of prefix cached tokens.
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens
                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)
                    encoder_compute_budget = new_encoder_compute_budget
                # Allocate for external load encoder cache
                if external_load_encoder_input:
                    for i in external_load_encoder_input:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)

            # re-queue requests skipped in this pass ahead of older skipped items.
            if step_skipped_waiting:
                self.skipped_waiting.prepend_requests(step_skipped_waiting)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens

        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs
        ) <= len(self.running)

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
            if self.running:
                any_request_id = self.running[0].request_id
                num_common_prefix_blocks = (
                    self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)
                )

        # Construct the scheduler output.
        if self.use_v2_model_runner:
            scheduled_new_reqs = scheduled_new_reqs + scheduled_resumed_reqs
            scheduled_resumed_reqs = []
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                NewRequestData.from_request(
                    req, req_to_new_blocks[req.request_id].get_block_ids()
                )
                for req in scheduled_new_reqs
            ]

        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )

        # Record the request ids that were scheduled in this step.
        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        new_block_ids_to_zero = (
            (self.kv_cache_manager.take_new_block_ids() or None)
            if self.needs_kv_cache_zeroing
            else None
        )

        # Split preempted_reqs into reactive-offload (PRE_OFFLOADING) and
        # immediate-free (PREEMPTED). The offload connector consumes the
        # former via _get_proactive_store_specs; the latter resets block_ids
        # in _get_reqs_to_store.
        if self._step_proactive_preempts:
            preempted_immediate_ids = {
                req.request_id for req in preempted_reqs
                if req.request_id not in self._step_proactive_preempts
            }
            proactive_ids = set(self._step_proactive_preempts)
            self._step_proactive_preempts.clear()
        else:
            preempted_immediate_ids = {req.request_id for req in preempted_reqs}
            proactive_ids = None

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            preempted_req_ids=preempted_immediate_ids,
            proactive_preempted_req_ids=proactive_ids,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            new_block_ids_to_zero=new_block_ids_to_zero,
        )

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self._build_kv_connector_meta(self.connector, scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # Build the connector meta for ECConnector
        if self.ec_connector is not None:
            ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.ec_connector_metadata = ec_meta

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output

    def _build_kv_connector_meta(
        self, connector: KVConnectorBase_V1, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        return connector.build_connector_meta(scheduler_output)

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        """Preempt a request and put it back to the waiting queue.

        NOTE: The request should be popped from the running queue outside of this
        method.

        Two modes:
          (a) Default (DISCARD-style): immediately free GPU blocks; request must
              re-prefill from scratch on resume.
          (b) PROBE_REACTIVE_OFFLOAD=1 with offload connector attached: route
              through the PRE_OFFLOADING state.  Blocks move to a draining
              pool; the connector schedules async D2H of remaining KV; on
              completion, _finalize_proactive_preemption releases the drain
              pool and moves the request to PREEMPTED in the waiting queue.
              On resume, the offload manager's prefix-cache lookup finds the
              CPU copy → cheap restore instead of full re-prefill.
              Trigger condition and event count are identical to (a).
        """
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )

        # Reactive-offload gating:
        # - free_to_drain moves blocks to a DRAIN POOL, not the free pool.
        #   They become available again only after async D2H completes
        #   (at earliest, next scheduler step).
        # - If we use reactive-offload during an admission cascade, the
        #   freed-but-draining blocks don't satisfy allocate_slots, so
        #   the loop cascades into preempting more victims → drain pool
        #   inflates, preempt count balloons.
        # - Fix: at most ONE reactive-offload preempt per scheduler step.
        #   Subsequent preempts in the same step fall back to immediate-
        #   discard so admission completes. Across steps, the drain pool
        #   clears (async D2H finishes) before the next reactive-offload
        #   fires. This caps drain-pool growth at 1 per step while still
        #   providing cheap resume for the most-recently-preempted req.
        if (
            _PROBE_REACTIVE_OFFLOAD
            and self.connector is not None
            and not self._step_proactive_preempts
        ):
            # Mode (b): reactive offload (first preempt of this step).
            drain_ids = self.kv_cache_manager.free_to_drain(request)
            self._probe_drain_block_ids[request.request_id] = drain_ids
            self._probe_pre_offloading[request.request_id] = request
            self.encoder_cache_manager.free(request)
            request.status = RequestStatus.PRE_OFFLOADING
            if request.spec_token_ids:
                request.spec_token_ids = []
            request.num_preemptions += 1
            if self.log_stats:
                request.record_event(EngineCoreEventType.PREEMPTED, timestamp)
            # Tell the connector to store this request's remaining blocks now
            # (consumed in scheduler_output.proactive_preempted_req_ids).
            self._step_proactive_preempts.add(request.request_id)
            # The request is NOT requeued to waiting yet — that happens in
            # _finalize_proactive_preemption once async D2H completes.
            return

        # Mode (a): existing DISCARD-style immediate-free path.
        self.kv_cache_manager.free(request)
        self.encoder_cache_manager.free(request)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        request.probe_ema_h = _PROBE_RESET_H   # PROBE M3: reset on requeue; P3b warms to H_max
        request.probe_ema_h_fast = _PROBE_RESET_H
        request.probe_ema_p_eos = 0.0          # PROBE A1: reset P(EOS) EMA
        request.probe_ema_p_eos_fast = 0.0
        request.probe_peak_h = 0.0             # PROBE P3g: reset peak entropy
        request.probe_spike_ema = 0.0          # probe_spike: reset spike-rate EMA
        request.probe_spike_count = 0          # S2: reset cumulative spike count
        request.probe_near_zero_count = 0      # 2-feature predictor: reset near-zero count
        if request.spec_token_ids:
            request.spec_token_ids = []
        request.num_preemptions += 1
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)

        # Put the request back to the waiting queue.
        self.waiting.prepend_request(request)

    def _finalize_proactive_preemption(
        self,
        request: Request,
        scheduled_timestamp: float | None,
    ) -> None:
        """Called when async D2H for a PRE_OFFLOADING request completes.

        GPU blocks were freed to the draining pool at pre-offload start.
        Now that D2H is confirmed done, release them to the free queue so
        new requests can use them.  Move the request to waiting as PREEMPTED.
        The CPU copy is available via the offloading connector for H2D on resume.
        """
        assert request.status == RequestStatus.PRE_OFFLOADING
        drain_ids = self._probe_drain_block_ids.pop(request.request_id, [])
        self.kv_cache_manager.release_draining(drain_ids)
        self.encoder_cache_manager.free(request)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        request.probe_ema_h = 0.0
        request.probe_ema_h_fast = 0.0
        request.probe_ema_p_eos = 0.0
        request.probe_ema_p_eos_fast = 0.0
        if request.spec_token_ids:
            request.spec_token_ids = []
        # num_preemptions was already incremented in _preempt_request at
        # trigger time; do NOT double-count here.
        if self.log_stats and scheduled_timestamp is not None:
            request.record_event(EngineCoreEventType.PREEMPTED, scheduled_timestamp)
        self._probe_pre_offloading.pop(request.request_id, None)
        self.waiting.prepend_request(request)
        logger.debug(
            "PROBE proactive pre-offload complete: req %s freed GPU blocks "
            "and moved to waiting (H2D on resume from CPU copy)",
            request.request_id,
        )

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token
            request.is_prefill_chunk = request.num_computed_tokens < (
                request.num_tokens + request.num_output_placeholders
            )
            scheduler_output.has_structured_output_requests |= (
                request.use_structured_output and not request.is_prefill_chunk
            )

            # NOTE: _free_encoder_inputs relies on num_computed_tokens, which
            # may be updated again in _update_from_output for speculative
            # decoding. However, it is safe to call the method here because
            # encoder inputs are always part of the prompt, not the output,
            # and thus are unaffected by speculative decoding.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

        # Clear the finished request IDs.
        # NOTE: We shouldn't do self.finished_req_ids.clear() here because
        # it will also affect the scheduler output.
        self.finished_req_ids = set()

    def _update_request_as_session(
        self, session: Request, update: StreamingUpdate
    ) -> None:
        """
        Updates the waiting session with the next streaming update.

        Discards the last sampled output token from the prior input chunk.
        """

        # Current streaming input behaviour: Keep only computed output tokens
        # (discard final sampled output token).
        num_computed_tokens = session.num_computed_tokens
        kept_output_tokens = session._all_token_ids[
            session.num_prompt_tokens : num_computed_tokens
        ]
        del session._all_token_ids[num_computed_tokens:]
        session._output_token_ids.clear()
        assert session.prompt_token_ids is not None
        # Extend prompt with kept output tokens.
        session.prompt_token_ids.extend(kept_output_tokens)

        if update.mm_features:
            base = session.num_tokens
            for mm_feature in update.mm_features:
                mm_feature.mm_position = replace(
                    mm_feature.mm_position, offset=mm_feature.mm_position.offset + base
                )
            session.mm_features.extend(update.mm_features)

        session._all_token_ids.extend(update.prompt_token_ids or ())
        session.prompt_token_ids.extend(update.prompt_token_ids or ())
        # Update block hashes for the new tokens.
        session.update_block_hashes()
        session.num_prompt_tokens = len(session.prompt_token_ids)
        session.arrival_time = update.arrival_time
        session.sampling_params = update.sampling_params
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            self.num_waiting_for_streaming_input -= 1
        session.status = RequestStatus.WAITING

        if self.log_stats:
            session.record_event(EngineCoreEventType.QUEUED)

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[tuple[list[int], ...] | None] = []
        all_token_ids: dict[str, list[int]] = {}
        num_computed_tokens: list[int] = []
        num_output_tokens: list[int] = []
        resumed_req_ids = set()

        num_running_reqs = len(running_reqs)
        for idx, req in enumerate(itertools.chain(running_reqs, resumed_reqs)):
            req_id = req.request_id
            req_ids.append(req_id)
            # NOTE: In PP+async scheduling, we consume token ids via a direct GPU
            # broadcast path (`input_batch.prev_sampled_token_ids`), so we can
            # omit this payload.
            if self.use_pp and not self.scheduler_config.async_scheduling:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker. Otherwise, we don't
                # need to send the sampled tokens back because the model runner
                # will cache them.
                num_tokens = num_scheduled_tokens[req_id] - len(
                    spec_decode_tokens.get(req_id, ())
                )
                token_ids = req.all_token_ids[
                    req.num_computed_tokens : req.num_computed_tokens + num_tokens
                ]
                new_token_ids.append(token_ids)
            scheduled_in_prev_step = req_id in self.prev_step_scheduled_req_ids
            if idx >= num_running_reqs:
                assert not scheduled_in_prev_step
                resumed_req_ids.add(req_id)
            if not scheduled_in_prev_step:
                all_token_ids[req_id] = req.all_token_ids.copy()
            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True)
            )
            num_computed_tokens.append(req.num_computed_tokens)
            num_output_tokens.append(
                req.num_output_tokens + req.num_output_placeholders
            )

        return CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=resumed_req_ids,
            new_token_ids=new_token_ids,
            all_token_ids=all_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            num_output_tokens=num_output_tokens,
        )

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
        shift_computed_tokens: int = 0,
    ) -> tuple[list[int], int, int, list[int]]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - It is not exist on remote encoder cache (via ECConnector)
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.

        Note that num_computed_tokens includes both locally cached
        blocks and externally cached blocks (via KVConnector).
        """
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_compute_budget, []
        encoder_inputs_to_schedule: list[int] = []
        mm_features = request.mm_features
        assert mm_features is not None
        assert len(mm_features) > 0
        external_load_encoder_input = []

        # NOTE: since scheduler operates on the request level (possibly with
        # multiple encoder inputs per request), we need to create temporary
        # trackers for accounting at the encoder input level.
        mm_hashes_to_schedule = set()
        num_embeds_to_schedule = 0
        for i, mm_feature in enumerate(mm_features):
            start_pos = mm_feature.mm_position.offset
            num_encoder_tokens = mm_feature.mm_position.length
            num_encoder_embeds = mm_feature.mm_position.get_num_embeds()
            item_identifier = mm_feature.identifier

            # The encoder output is needed if the two ranges overlap:
            # [num_computed_tokens, num_computed_tokens + num_new_tokens) and
            # [start_pos, start_pos + num_encoder_tokens)
            if (
                start_pos
                >= num_computed_tokens + num_new_tokens + shift_computed_tokens
            ):
                # The encoder input is not needed in this step.
                break

            if self.is_encoder_decoder and num_computed_tokens > 0:
                assert start_pos == 0, (
                    "Encoder input should be processed at the beginning of "
                    "the sequence when encoder-decoder models are used."
                )
                # Encoder input has already been computed
                # The calculation here is a bit different. We don't turn encoder
                # output into tokens that get processed by the decoder and
                # reflected in num_computed_tokens. Instead, start_pos reflects
                # the position where we need to ensure we calculate encoder
                # inputs. This should always be 0 to ensure we calculate encoder
                # inputs before running the decoder.  Once we've calculated some
                # decoder tokens (num_computed_tokens > 0), then we know we
                # already calculated encoder inputs and can skip here.
                continue
            elif start_pos + num_encoder_tokens <= num_computed_tokens:
                # The encoder input is already computed and stored
                # in the decoder's KV cache.
                continue

            if not self.is_encoder_decoder:
                # We are not using the encoder cache for encoder-decoder models,
                # yet.
                if item_identifier in mm_hashes_to_schedule:
                    # The same encoder input has already been scheduled in the
                    # current step.
                    continue

                if self.encoder_cache_manager.check_and_update_cache(request, i):
                    # The encoder input is already computed and cached from a
                    # previous step.
                    continue

            # If no encoder input chunking is allowed, we do not want to
            # partially schedule a multimodal item. If the scheduled range would
            # only cover part of the mm input, roll back to before the mm item.
            if (
                self.scheduler_config.disable_chunked_mm_input
                and num_computed_tokens < start_pos
                and (num_computed_tokens + num_new_tokens)
                < (start_pos + num_encoder_tokens)
            ):
                # Account for EAGLE shift when rolling back to avoid
                # encoder cache miss. This ensures the scheduled range
                # stops before start_pos even with the shift.
                num_new_tokens = max(
                    0, start_pos - (num_computed_tokens + shift_computed_tokens)
                )
                break
            if not self.encoder_cache_manager.can_allocate(
                request, i, encoder_compute_budget, num_embeds_to_schedule
            ):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                if num_computed_tokens + shift_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input.
                    num_new_tokens = start_pos - (
                        num_computed_tokens + shift_computed_tokens
                    )
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0
                break

            # Calculate the number of embeddings to schedule in the current range
            # of scheduled encoder placeholder tokens.
            start_idx_rel = max(0, num_computed_tokens - start_pos)
            end_idx_rel = min(
                num_encoder_tokens, num_computed_tokens + num_new_tokens - start_pos
            )
            curr_embeds_start, curr_embeds_end = (
                mm_feature.mm_position.get_embeds_indices_in_range(
                    start_idx_rel, end_idx_rel
                )
            )
            # There's no embeddings in the current range of encoder placeholder tokens
            # so we can skip the encoder input.
            if curr_embeds_end - curr_embeds_start == 0:
                continue

            if self.ec_connector is not None and self.ec_connector.has_cache_item(
                item_identifier
            ):
                mm_hashes_to_schedule.add(item_identifier)
                external_load_encoder_input.append(i)
                num_embeds_to_schedule += num_encoder_embeds
                continue

            num_embeds_to_schedule += num_encoder_embeds
            encoder_compute_budget -= num_encoder_embeds
            mm_hashes_to_schedule.add(item_identifier)
            encoder_inputs_to_schedule.append(i)

        return (
            encoder_inputs_to_schedule,
            num_new_tokens,
            encoder_compute_budget,
            external_load_encoder_input,
        )

    def get_grammar_bitmask(
        self, scheduler_output: SchedulerOutput
    ) -> GrammarOutput | None:
        # Collect list of scheduled request ids that use structured output.
        # The corresponding rows of the bitmask will be in this order.
        if not scheduler_output.has_structured_output_requests:
            return None

        structured_output_request_ids = [
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if (req := self.requests.get(req_id))
            and (req.use_structured_output and not req.is_prefill_chunk)
        ]
        if not structured_output_request_ids:
            return None

        bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            scheduler_output.scheduled_spec_decode_tokens,
        )
        return GrammarOutput(structured_output_request_ids, bitmask)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output
        cudagraph_stats = model_runner_output.cudagraph_stats

        perf_stats: PerfStats | None = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        if kv_connector_stats and self.connector:
            kv_stats = self.connector.get_kv_connector_stats()
            if kv_stats:
                kv_connector_stats = kv_connector_stats.aggregate(kv_stats)

        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
            failed_kv_load_req_ids = self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids,
                num_scheduled_tokens,
            )

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # skip failed or rescheduled requests from KV load failure
                continue
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism or in async scheduling).
                # NOTE(Kuntai): When delay_free_blocks=True (for async KV
                # cache transfer in KV connector), the aborted request will not
                # be set to None (in order to finish async KV transfer).
                # In this case, we use is_finished() to check.
                continue

            req_index = model_runner_output.req_id_to_index[req_id]

            # PROBE M3: update per-request EMA entropy for g_rough / g_slope.
            if (model_runner_output.token_entropy is not None
                    and req_index < len(model_runner_output.token_entropy)):
                h_t = model_runner_output.token_entropy[req_index]
                request.probe_ema_h = (
                    _PROBE_EMA_ALPHA * h_t
                    + (1.0 - _PROBE_EMA_ALPHA) * request.probe_ema_h
                )
                request.probe_ema_h_fast = (
                    _PROBE_EMA_ALPHA_FAST * h_t
                    + (1.0 - _PROBE_EMA_ALPHA_FAST) * request.probe_ema_h_fast
                )
                # P3g: track running maximum entropy for peak-gap discount.
                if h_t > request.probe_peak_h:
                    request.probe_peak_h = h_t
                # probe_spike: EMA of spike indicator.
                spike = 1.0 if h_t > _PROBE_SPIKE_THRESHOLD else 0.0
                request.probe_spike_ema = (
                    _PROBE_SPIKE_ALPHA * spike
                    + (1.0 - _PROBE_SPIKE_ALPHA) * request.probe_spike_ema
                )
                # S2 online length predictor: cumulative spike count.
                # rate = spike_count / num_computed is the feature for T̂.
                if spike:
                    request.probe_spike_count += 1
                # 2-feature predictor: cumulative near-zero (high-confidence) count.
                # zero_rate = near_zero_count / num_computed pairs with spike rate.
                if h_t < _PROBE_ZERO_THRESHOLD:
                    request.probe_near_zero_count += 1

            # PROBE A1: update per-request EMA P(EOS) for probe_slope_peos.
            if (model_runner_output.token_p_eos is not None
                    and req_index < len(model_runner_output.token_p_eos)):
                p_t = model_runner_output.token_p_eos[req_index]
                request.probe_ema_p_eos = (
                    _PROBE_EMA_ALPHA * p_t
                    + (1.0 - _PROBE_EMA_ALPHA) * request.probe_ema_p_eos
                )
                request.probe_ema_p_eos_fast = (
                    _PROBE_EMA_ALPHA_FAST * p_t
                    + (1.0 - _PROBE_EMA_ALPHA_FAST) * request.probe_ema_p_eos_fast
                )

            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []
            )

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            )
            if scheduled_spec_token_ids and generated_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens.
                if request.num_computed_tokens > 0:
                    request.num_computed_tokens -= num_rejected
                # If async scheduling, num_output_placeholders also includes
                # the scheduled spec tokens count and so is similarly adjusted.
                if request.num_output_placeholders > 0:
                    request.num_output_placeholders -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                    request_id=req_id,
                )

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None
            kv_transfer_params = None
            status_before_stop = request.status

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids
                )
            elif request.pooling_params and pooler_output is not None:
                # Pooling stops as soon as there is output.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True

            routed_experts = None
            finish_reason = None
            if stopped:
                routed_experts = self._get_routed_experts(request)

                # Capture finish_reason BEFORE _handle_stopped_request, which may
                # reset the status to WAITING for streaming requests that continue.
                finish_reason = request.get_finished_reason()
                finished = self._handle_stopped_request(request)
                if finished:
                    kv_transfer_params = self._free_request(request)
                    # PROBE lifecycle: completion event.
                    _ppl.log_complete(
                        req_id=request.request_id,
                        num_output_tokens=request.num_output_tokens,
                        timestamp=time.monotonic(),
                        finish_reason=str(finish_reason) if finish_reason else "unknown",
                    )

                    # PROBE M4: update growth ring with observed actual remaining.
                    # We stored a snapshot when this request was last preempted;
                    # now that it has finished we know how many tokens it actually
                    # generated after that checkpoint.
                    if _PROBE_SLOPE_RING is not None and req_id in _probe_slope_ring_pending:
                        snap_len, snap_g, snap_out = _probe_slope_ring_pending.pop(req_id)
                        actual_remaining = request.num_output_tokens - snap_out
                        _PROBE_SLOPE_RING.update(snap_len, snap_g, float(actual_remaining))

                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if (
                request.sampling_params is not None
                and request.sampling_params.logprobs is not None
                and logprobs
            ):
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

            if new_token_ids and self.structured_output_manager.should_advance(request):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                assert struct_output_request.grammar is not None
                ok = struct_output_request.grammar.accept_tokens(req_id, new_token_ids)
                if not ok:
                    logger.warning(
                        "Unexpected: grammar rejected tokens %s for request %s.",
                        new_token_ids,
                        req_id,
                    )

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if (
                new_token_ids
                or pooler_output is not None
                or kv_transfer_params
                or stopped
            ):
                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=finish_reason,
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                        num_external_computed_tokens=request.num_external_computed_tokens,
                        routed_experts=routed_experts,
                        num_nans_in_logits=request.num_nans_in_logits,
                    )
                )
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)

        if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
            requests = [self.requests[req_id] for req_id in failed_kv_load_req_ids]
            self.finish_requests(failed_kv_load_req_ids, RequestStatus.FINISHED_ERROR)
            for request in requests:
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=request.get_finished_reason(),
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                    )
                )

        # KV Connector: update state for finished KV Transfers.
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set
                    )
            finished_req_ids.clear()

        if (
            stats := self.make_stats(
                spec_decoding_stats, kv_connector_stats, cudagraph_stats, perf_stats
            )
        ) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs

    @staticmethod
    def _is_blocked_waiting_status(status: RequestStatus) -> bool:
        return status in (
            RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
            RequestStatus.WAITING_FOR_REMOTE_KVS,
            RequestStatus.WAITING_FOR_STREAMING_REQ,
        )

    def _enqueue_waiting_request(self, request: Request) -> None:
        if self._is_blocked_waiting_status(request.status):
            self.skipped_waiting.add_request(request)
        else:
            self.waiting.add_request(request)

    def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        if self.policy == SchedulingPolicy.FCFS:
            return self.skipped_waiting or self.waiting or None

        # PRIORITY mode: compare queue heads when both queues are non-empty.
        if self.waiting and self.skipped_waiting:
            waiting_req = self.waiting.peek_request()
            skipped_req = self.skipped_waiting.peek_request()
            return self.waiting if waiting_req < skipped_req else self.skipped_waiting

        return self.waiting or self.skipped_waiting or None

    def _handle_stopped_request(self, request: Request) -> bool:
        """Return True if finished (can be False for resumable requests)."""
        if not request.resumable:
            return True

        if request.streaming_queue:
            update = request.streaming_queue.popleft()
            if update is None:
                # Streaming request finished.
                return True
            self._update_request_as_session(request, update)
        else:
            request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
            self.num_waiting_for_streaming_input += 1

        self._enqueue_waiting_request(request)
        return False

    def _get_routed_experts(self, request: Request) -> np.ndarray | None:
        if not self.vllm_config.model_config.enable_return_routed_experts:
            return None

        kv_blocks = self.kv_cache_manager.get_blocks(request.request_id)
        block_ids = kv_blocks.get_block_ids()[self.routed_experts_attn_gid]
        num_tokens = request.num_tokens - 1

        # compute slot mapping using attention group's block_size
        block_ids_array = np.array(block_ids, dtype=np.int32)
        num_blocks = len(block_ids)
        attn_group = self.kv_cache_config.kv_cache_groups[self.routed_experts_attn_gid]
        block_size = attn_group.kv_cache_spec.block_size

        # generate block offsets
        block_offsets = np.arange(0, block_size)

        # compute slot mapping: slot = block_id * block_size + offset
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids_array.reshape((num_blocks, 1)) * block_size
        ).flatten()[:num_tokens]

        return self.routed_experts_reader.get_routed_experts(indices=slot_mapping)

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]  # Trim new tokens if needed.
                break
        return new_token_ids, stopped

    def _free_encoder_inputs(self, request: Request) -> None:
        cached_encoder_input_ids = self.encoder_cache_manager.get_cached_input_ids(
            request
        )
        # OPTIMIZATION: Avoid list(set) if the set is empty.
        if not cached_encoder_input_ids:
            return

        # Here, we use list(set) to avoid modifying the set while iterating
        # over it.
        for input_id in list(cached_encoder_input_ids):
            mm_feature = request.mm_features[input_id]
            start_pos = mm_feature.mm_position.offset
            num_tokens = mm_feature.mm_position.length
            if self.is_encoder_decoder and request.num_computed_tokens > 0:
                # With Whisper, as soon as we've generated a single token,
                # we know we're done with the encoder input. Cross Attention
                # KVs have been calculated and cached already.
                self.encoder_cache_manager.free_encoder_input(request, input_id)
            elif start_pos + num_tokens <= request.num_computed_tokens:
                # The encoder output is already processed and stored
                # in the decoder's KV cache.
                self.encoder_cache_manager.free_encoder_input(request, input_id)

    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            if request.is_prefill_chunk:
                # Ignore draft tokens for prefill chunks.
                if request.spec_token_ids:
                    request.spec_token_ids = []
                continue

            # Add newly generated spec token ids to the request.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]
            request.spec_token_ids = spec_token_ids

    def update_draft_token_ids_in_output(
        self, draft_token_ids: DraftTokenIds, scheduler_output: SchedulerOutput
    ) -> None:
        num_invalid_spec_tokens: dict[str, int] = {}

        sched_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            placeholder_spec_tokens = sched_spec_tokens.get(req_id)
            if not placeholder_spec_tokens:
                continue

            orig_num_spec_tokens = len(placeholder_spec_tokens)
            # Trim drafts to scheduled number of spec tokens
            # (needed for chunked prefill case for example).
            del spec_token_ids[orig_num_spec_tokens:]
            # Filter out spec tokens which do not adhere to the grammar.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                assert metadata is not None and metadata.grammar is not None
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)
            # Pad to original number of spec tokens.
            num_invalid_tokens = orig_num_spec_tokens - len(spec_token_ids)
            if num_invalid_tokens:
                spec_token_ids.extend([-1] * num_invalid_tokens)
                num_invalid_spec_tokens[req_id] = num_invalid_tokens

            sched_spec_tokens[req_id] = spec_token_ids

        scheduler_output.num_invalid_spec_tokens = num_invalid_spec_tokens

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        return len(self.running), len(self.waiting) + len(self.skipped_waiting)

    def add_request(self, request: Request) -> None:
        existing = self.requests.get(request.request_id)
        if existing is not None:
            update = StreamingUpdate.from_request(request)
            if existing.status != RequestStatus.WAITING_FOR_STREAMING_REQ:
                assert existing.streaming_queue is not None, "duplicate request id"
                # Queue next input chunk (or finished sentinel).
                existing.streaming_queue.append(update)
            elif update is not None:
                # Commence next input chunk.
                self._update_request_as_session(existing, update)
            else:
                # Streaming-input session finished.
                self.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
        else:
            if request.resumable:
                request.streaming_queue = deque()
            self._enqueue_waiting_request(request)
            self.requests[request.request_id] = request
            if self.log_stats:
                request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[tuple[str, int]]:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.

        If request_ids is None, all requests will be finished.

        Returns:
            Tuple of (req_id, client_index) for requests that were aborted. Will not
            include any that were already finished.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        elif request_ids is not None:
            request_ids = set(request_ids)
        else:
            request_ids = self.requests.keys()

        running_requests_to_remove = set()
        waiting_requests_to_remove = []
        valid_requests = []

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)
            elif request.status == RequestStatus.PRE_OFFLOADING:
                # Blocks are in the draining pool; release them now.
                # The request is NOT in self.running (was removed at pre-offload
                # trigger), so no running-queue removal is needed.  _free_request
                # will call kv_cache_manager.free() which is a no-op since
                # req_to_blocks was already cleared by free_to_drain().
                drain_ids = self._probe_drain_block_ids.pop(req_id, [])
                self.kv_cache_manager.release_draining(drain_ids)
                self._probe_pre_offloading.pop(req_id, None)
            else:
                if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                    self.num_waiting_for_streaming_input -= 1
                waiting_requests_to_remove.append(request)

        # Remove all requests from queues at once for better efficiency
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)
            self.skipped_waiting.remove_requests(waiting_requests_to_remove)

        # Second pass: set status and free requests
        for request in valid_requests:
            delay_free_blocks = False
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                delay_free_blocks = (
                    request.request_id not in self.finished_recving_kv_req_ids
                )
                self.finished_recving_kv_req_ids.discard(request.request_id)
                self.failed_recving_kv_req_ids.discard(request.request_id)

            request.status = finished_status
            self._free_request(request, delay_free_blocks=delay_free_blocks)

        return [(r.request_id, r.client_index) for r in valid_requests]

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> dict[str, Any] | None:
        assert request.is_finished()

        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        delay_free_blocks |= connector_delay_free_blocks
        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params

    def _free_blocks(self, request: Request):
        assert request.is_finished()
        self.kv_cache_manager.free(request)
        del self.requests[request.request_id]

    @property
    def pause_state(self) -> PauseState:
        return self._pause_state

    def set_pause_state(self, pause_state: PauseState) -> None:
        self._pause_state = pause_state

    def get_num_unfinished_requests(self) -> int:
        if self._pause_state == PauseState.PAUSED_ALL:
            return 0
        if self._pause_state == PauseState.PAUSED_NEW:
            return len(self.running)
        num_waiting = (
            len(self.waiting)
            + len(self.skipped_waiting)
            - self.num_waiting_for_streaming_input
        )
        return num_waiting + len(self.running)

    def has_finished_requests(self) -> bool:
        return len(self.finished_req_ids) > 0

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the KV prefix cache.

        If reset_running_requests is True, all the running requests will be
        preempted and moved to the waiting queue.
        Otherwise, this method will only reset the KV prefix cache when there
        is no running requests taking KV cache.
        """
        if reset_running_requests:
            # For logging.
            timestamp = time.monotonic()
            # Invalidate all the current running requests KV's by pushing them to
            # the waiting queue. In this case, we can reduce the ref count of all
            # the kv blocks to 0 and thus we can make sure the reset is successful.
            # Preempt in reverse order so the requests will be added back to the
            # running queue in FIFO order.
            while self.running:
                request = self.running.pop()
                self._preempt_request(request, timestamp)
                # NOTE(zhuohan): For async scheduling, we need to discard the latest
                # output token on the fly to avoid a redundant repetitive output token.
                request.num_output_placeholders = 0
                request.discard_latest_async_tokens = True

            # Clear scheduled request ids cache. Since we are forcing preemption
            # + resumption in the same step, we must act as if these requests were
            # not scheduled in the prior step. They will be flushed from the
            # persistent batch in the model runner.
            self.prev_step_scheduled_req_ids.clear()

        reset_successful = self.kv_cache_manager.reset_prefix_cache()
        if reset_running_requests and not reset_successful:
            raise RuntimeError(
                "Failed to reset KV cache even when all the running requests are "
                "preempted and moved to the waiting queue. This is likely due to "
                "the presence of running requests waiting for remote KV transfer, "
                "which is not supported yet."
            )

        if reset_connector:
            reset_successful = self.reset_connector_cache() and reset_successful

        return reset_successful

    def reset_connector_cache(self) -> bool:
        if self.connector is None:
            logger.warning("reset_connector called but no KV connector is configured.")
            return False

        if self.connector.reset_cache() is False:
            return False

        if self.log_stats:
            assert self.connector_prefix_cache_stats is not None
            self.connector_prefix_cache_stats.reset = True

        return True

    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings are not reused.
        """
        self.encoder_cache_manager.reset()

    def make_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None = None,
        kv_connector_stats: KVConnectorStats | None = None,
        cudagraph_stats: CUDAGraphStat | None = None,
        perf_stats: PerfStats | None = None,
    ) -> SchedulerStats | None:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        connector_prefix_cache_stats: PrefixCacheStats | None = None
        if self.connector_prefix_cache_stats is not None:
            connector_prefix_cache_stats = self.connector_prefix_cache_stats
            self.connector_prefix_cache_stats = PrefixCacheStats()
        eviction_events = (
            self.kv_metrics_collector.drain_events()
            if self.kv_metrics_collector is not None
            else []
        )
        spec_stats = spec_decoding_stats
        connector_stats_payload = (
            kv_connector_stats.data if kv_connector_stats else None
        )
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting) + len(self.skipped_waiting),
            kv_cache_usage=self.kv_cache_manager.usage,
            encoder_cache_usage=self._get_encoder_cache_usage(),
            prefix_cache_stats=prefix_cache_stats,
            connector_prefix_cache_stats=connector_prefix_cache_stats,
            kv_cache_eviction_events=eviction_events,
            spec_decoding_stats=spec_stats,
            kv_connector_stats=connector_stats_payload,
            cudagraph_stats=cudagraph_stats,
            perf_stats=perf_stats,
        )

    def _get_encoder_cache_usage(self) -> float:
        """Get encoder cache usage as a fraction (0.0 to 1.0)."""
        ecm = self.encoder_cache_manager
        if ecm.cache_size == 0:
            return 0.0
        used_slots = ecm.cache_size - ecm.num_free_slots
        return used_slots / ecm.cache_size

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        num_invalid_spec_tokens: dict[str, int] | None,
        request_id: str,
    ) -> SpecDecodingStats | None:
        if not self.log_stats or not num_draft_tokens:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        if num_invalid_spec_tokens:
            num_draft_tokens -= num_invalid_spec_tokens.get(request_id, 0)
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens, num_accepted_tokens=num_accepted_tokens
        )
        return spec_decoding_stats

    def shutdown(self) -> None:
        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()
        if self.connector is not None:
            self.connector.shutdown()

    ########################################################################
    # KV Connector Related Methods
    ########################################################################

    def get_kv_connector(self) -> KVConnectorBase_V1 | None:
        return self.connector

    def _connector_finished(
        self, request: Request
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        if self.connector is None:
            return False, None

        # Free any out-of-window prefix blocks before we hand the block table to
        # the connector.
        self.kv_cache_manager.remove_skipped_blocks(
            request_id=request.request_id,
            total_computed_tokens=request.num_tokens,
        )

        block_ids = self.kv_cache_manager.get_block_ids(request.request_id)

        if not isinstance(self.connector, SupportsHMA):
            # NOTE(Kuntai): We should deprecate this code path after we enforce
            # all connectors to support HMA.
            # Hybrid memory allocator should be already turned off for this
            # code path, but let's double-check here.
            assert len(self.kv_cache_config.kv_cache_groups) == 1
            return self.connector.request_finished(request, block_ids[0])

        return self.connector.request_finished_all_groups(request, block_ids)

    def _update_waiting_for_remote_kv(self, request: Request) -> None:
        """
        KV Connector: update request state after async recv is finished.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None

        if request.request_id in self.failed_recving_kv_req_ids:
            # Request had KV load failures; num_computed_tokens was already
            # updated in _update_requests_with_invalid_blocks
            if request.num_computed_tokens:
                # Cache any valid computed tokens.
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
            else:
                # No valid computed tokens, release allocated blocks.
                # There may be a local cache hit on retry.
                self.kv_cache_manager.free(request)

            self.failed_recving_kv_req_ids.remove(request.request_id)
        else:
            # Now that the blocks are ready, actually cache them.
            # This will cache the blocks iff caching is enabled.
            self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)

            # on a full prompt hit, we need to re-compute the last token
            # in order to be able to sample the next token
            if request.num_computed_tokens == request.num_tokens:
                request.num_computed_tokens = request.num_tokens - 1

            # Count the number of prefix cached tokens.
            if request.num_cached_tokens < 0:
                request.num_cached_tokens = request.num_computed_tokens

        self.finished_recving_kv_req_ids.remove(request.request_id)

    def _try_promote_blocked_waiting_request(self, request: Request) -> bool:
        """
        Try to promote a blocked waiting request back to schedulable states.
        """
        if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            # finished_recving_kv_req_ids is populated during
            # update_from_output(), based on worker-side connector signals
            # in KVConnectorOutput.finished_recving
            if request.request_id not in self.finished_recving_kv_req_ids:
                return False
            self._update_waiting_for_remote_kv(request)
            if request.num_preemptions:
                request.status = RequestStatus.PREEMPTED
            else:
                request.status = RequestStatus.WAITING
            return True

        if request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
            structured_output_req = request.structured_output_request
            if not (structured_output_req and structured_output_req.grammar):
                return False
            request.status = RequestStatus.WAITING
            return True

        if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            assert not request.streaming_queue
            return False

        raise AssertionError(
            "Unexpected blocked waiting status in promotion: "
            f"{request.status.name} for request {request.request_id}"
        )

    def _update_from_kv_xfer_finished(self, kv_connector_output: KVConnectorOutput):
        """
        KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            schedule the request during the next step.
        """

        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        # KV Connector:: update recv and send status from last step.
        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            assert req_id in self.requests
            req = self.requests[req_id]
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.finished_recving_kv_req_ids.add(req_id)
            else:
                assert RequestStatus.is_finished(req.status)
                self._free_blocks(self.requests[req_id])
        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            if req_id not in self.requests:
                # Request completed or was aborted before the D2H transfer
                # callback fired; blocks already freed at completion time.
                logger.debug("Skipping finished_sending for already-removed "
                             "request %s", req_id)
                continue
            req = self.requests[req_id]
            if req.status == RequestStatus.PRE_OFFLOADING:
                # Proactive D2H complete: GPU blocks are still valid (we held
                # them until now), so free them safely, then move to waiting.
                self._finalize_proactive_preemption(req, scheduled_timestamp=None)
            else:
                self._free_blocks(req)

    def _update_requests_with_invalid_blocks(
        self,
        requests: Iterable[Request],
        invalid_block_ids: set[int],
        num_scheduled_tokens: dict[str, int],
        evict_blocks: bool = True,
    ) -> tuple[set[str], int, set[int]]:
        """
        Identify and update requests affected by invalid KV cache blocks.

        This method scans the given requests, detects those with invalid blocks
        and adjusts their `num_computed_tokens` to the longest valid prefix.
        For observability, it also accumulates the total number of tokens that
        will need to be recomputed across all affected requests.

        Args:
            requests: The set of requests to scan for invalid blocks.
            invalid_block_ids: IDs of invalid blocks.
            num_scheduled_tokens: req_id -> number of scheduled tokens.
            evict_blocks: Whether to collect blocks for eviction (False for
                async requests which aren't cached yet).

        Returns:
            tuple:
                - affected_req_ids (set[str]): IDs of requests impacted by
                invalid blocks.
                - total_affected_tokens (int): Total number of tokens that must
                be recomputed across all affected requests.
                - blocks_to_evict (set[int]): Block IDs to evict from cache,
                including invalid blocks and downstream dependent blocks.
        """
        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        blocks_to_evict: set[int] = set()
        # If a block is invalid and shared by multiple requests in the batch,
        # these requests must be rescheduled, but only the first will recompute
        # it. This set tracks blocks already marked for recomputation.
        marked_invalid_block_ids: set[int] = set()
        for request in requests:
            is_affected = False
            marked_invalid_block = False
            req_id = request.request_id
            # TODO (davidb): add support for hybrid memory allocator
            (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
            # We iterate only over blocks that may contain externally computed
            # tokens
            req_num_computed_tokens = (
                request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)
            )

            req_num_computed_blocks = (
                req_num_computed_tokens + self.block_size - 1
            ) // self.block_size
            for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):
                if block_id not in invalid_block_ids:
                    continue

                is_affected = True

                if block_id in marked_invalid_block_ids:
                    # This invalid block is shared with a previous request
                    # and was already marked for recomputation.
                    # This means this request can still consider this block
                    # as computed when rescheduled.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    continue

                marked_invalid_block_ids.add(block_id)

                if marked_invalid_block:
                    # This request has already marked an invalid block for
                    # recomputation and updated its num_computed_tokens.
                    continue

                marked_invalid_block = True
                # Truncate the computed tokens at the first failed block
                request.num_computed_tokens = idx * self.block_size
                num_affected_tokens = (
                    req_num_computed_tokens - request.num_computed_tokens
                )
                total_affected_tokens += num_affected_tokens
                request.num_external_computed_tokens -= num_affected_tokens
                # collect invalid block and all downstream dependent blocks
                if evict_blocks:
                    blocks_to_evict.update(req_block_ids[idx:])

            if is_affected:
                if not marked_invalid_block:
                    # All invalid blocks of this request are shared with
                    # previous requests and will be recomputed by them.
                    # Revert to considering only cached tokens as computed.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    total_affected_tokens += (
                        request.num_computed_tokens - req_num_computed_tokens
                    )
                    request.num_computed_tokens = req_num_computed_tokens

                affected_req_ids.add(request.request_id)

        return affected_req_ids, total_affected_tokens, blocks_to_evict

    def _handle_invalid_blocks(
        self, invalid_block_ids: set[int], num_scheduled_tokens: dict[str, int]
    ) -> set[str]:
        """
        Handle requests affected by invalid KV cache blocks.

        Returns:
            Set of affected request IDs to skip in update_from_output main loop.
        """
        should_fail = not self.recompute_kv_load_failures

        # handle async KV loads (not cached yet, evict_blocks=False)
        async_load_reqs = (
            req
            for req in self.skipped_waiting
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        )
        async_failed_req_ids, num_failed_tokens, _ = (
            self._update_requests_with_invalid_blocks(
                async_load_reqs,
                invalid_block_ids,
                num_scheduled_tokens,
                evict_blocks=False,
            )
        )

        total_failed_requests = len(async_failed_req_ids)
        total_failed_tokens = num_failed_tokens

        # handle sync loads (may be cached, collect blocks for eviction)
        sync_failed_req_ids, num_failed_tokens, sync_blocks_to_evict = (
            self._update_requests_with_invalid_blocks(
                self.running, invalid_block_ids, num_scheduled_tokens, evict_blocks=True
            )
        )

        total_failed_requests += len(sync_failed_req_ids)
        total_failed_tokens += num_failed_tokens

        if not total_failed_requests:
            return set()

        # evict invalid blocks and downstream dependent blocks from cache
        # only when not using recompute policy (where blocks will be recomputed
        # and reused by other requests sharing them)
        if sync_blocks_to_evict and not self.recompute_kv_load_failures:
            self.kv_cache_manager.evict_blocks(sync_blocks_to_evict)

        if should_fail:
            all_failed_req_ids = async_failed_req_ids | sync_failed_req_ids
            logger.error(
                "Failing %d request(s) due to KV load failure "
                "(failure_policy=fail, %d tokens affected). Request IDs: %s",
                total_failed_requests,
                total_failed_tokens,
                all_failed_req_ids,
            )
            return all_failed_req_ids

        logger.warning(
            "Recovered from KV load failure: "
            "%d request(s) rescheduled (%d tokens affected).",
            total_failed_requests,
            total_failed_tokens,
        )

        # Mark async requests with KV load failures for retry once loading completes
        self.failed_recving_kv_req_ids |= async_failed_req_ids
        # Return sync affected IDs to skip in update_from_output
        return sync_failed_req_ids

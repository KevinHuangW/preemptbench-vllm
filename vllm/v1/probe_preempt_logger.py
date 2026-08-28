# SPDX-License-Identifier: Apache-2.0
# PROBE M2: preemption event logger.
#
# Activated by PROBE_PREEMPT_LOG env var (path to JSONL output file).
#
# Per-preemption record:
#   {"t": <unix_ts>, "req_id": "...", "num_computed": N,
#    "num_preemptions": K, "cascade_id": C, "cascade_pos": P}
#
# cascade_id  — monotonically increasing integer, incremented each time
#               schedule() finishes with >=1 preemption.
# cascade_pos — position within the cascade (0 = first victim this round).
# A "cascading preemption" is a cascade with cascade_pos > 0 (i.e. >=2
# preemptions were needed in a single schedule() call).

import json
import os
import threading

_lock = threading.Lock()
_file = None
_cascade_id: int = 0
_offload_file = None
_lifecycle_file = None


def _get_file():
    global _file
    if _file is None:
        path = os.environ.get("PROBE_PREEMPT_LOG", "")
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            _file = open(path, "a", buffering=1)
    return _file


def _get_offload_file():
    global _offload_file
    if _offload_file is None:
        path = os.environ.get("PROBE_OFFLOAD_LOG", "")
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            _offload_file = open(path, "a", buffering=1)
    return _offload_file


def log_preemption(
    req_id: str,
    num_computed: int,
    num_preemptions_after: int,
    cascade_pos: int,
    timestamp: float,
) -> None:
    """Called once per preempted request, within a schedule() call."""
    f = _get_file()
    if f is None:
        return
    with _lock:
        record = {
            "t": timestamp,
            "req_id": req_id,
            "num_computed": num_computed,
            "num_preemptions": num_preemptions_after,
            "cascade_id": _cascade_id,
            "cascade_pos": cascade_pos,
        }
        f.write(json.dumps(record) + "\n")


def advance_cascade() -> None:
    """Call once per schedule() round after all preemptions in that round."""
    global _cascade_id
    f = _get_file()
    if f is None:
        return
    with _lock:
        _cascade_id += 1


def _get_lifecycle_file():
    """Open the per-request lifecycle log (admit / complete events)."""
    global _lifecycle_file
    if _lifecycle_file is None:
        path = os.environ.get("PROBE_LIFECYCLE_LOG", "")
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            _lifecycle_file = open(path, "a", buffering=1)
    return _lifecycle_file


def log_admit(
    req_id: str,
    prompt_tokens: int,
    timestamp: float,
    is_resume: bool,
    num_computed_at_admit: int,
) -> None:
    """Called when a request transitions from WAITING/PREEMPTED into RUNNING.
    is_resume=False for first admission, True for re-admission after preempt.
    num_computed_at_admit reflects re-start point (0 under DISCARD)."""
    f = _get_lifecycle_file()
    if f is None:
        return
    with _lock:
        record = {
            "t": timestamp,
            "req_id": req_id,
            "event": "resume" if is_resume else "admit",
            "prompt_tokens": prompt_tokens,
            "num_computed_at_admit": num_computed_at_admit,
        }
        f.write(json.dumps(record) + "\n")


def log_complete(
    req_id: str,
    num_output_tokens: int,
    timestamp: float,
    finish_reason: str,
) -> None:
    """Called when a request reaches finish (stop or length cap)."""
    f = _get_lifecycle_file()
    if f is None:
        return
    with _lock:
        record = {
            "t": timestamp,
            "req_id": req_id,
            "event": "complete",
            "num_output_tokens": num_output_tokens,
            "finish_reason": finish_reason,
        }
        f.write(json.dumps(record) + "\n")


def log_offload(
    req_id: str,
    n_store: int,
    n_evict_cpu: int,
    timestamp: float,
) -> None:
    """Called when the OffloadingConnector initiates a GPU→CPU KV block transfer.

    n_store     — new GPU blocks being written to CPU this step.
    n_evict_cpu — CPU pool blocks evicted to make room (CPU-level evictions).
    """
    f = _get_offload_file()
    if f is None:
        return
    with _lock:
        record = {
            "t": timestamp,
            "req_id": req_id,
            "n_store": n_store,
            "n_evict_cpu": n_evict_cpu,
        }
        f.write(json.dumps(record) + "\n")

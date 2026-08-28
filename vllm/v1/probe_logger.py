# SPDX-License-Identifier: Apache-2.0
# PROBE: per-step entropy logger for M1 trace collection.
#
# Writes one JSON line per decode step to PROBE_ENTROPY_LOG (env var).
# Format:
#   {"req_id": "...", "output_len": N, "entropy": H}   -- per-step
# Optionally, one prompt-event line per request to PROBE_PROMPT_LOG:
#   {"req_id": "...", "prompt_len": N, "prompt_token_ids": [...32 ints...]}
# The analysis script reconstructs remaining_growth from the final output_len
# for each req_id across steps.

import json
import os
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request

_lock = threading.Lock()
_file = None
_prompt_file = None
_output_len: dict[str, int] = {}  # req_id -> cumulative output tokens so far
_prompt_seen: set[str] = set()    # req_ids already prompt-logged


def _get_file():
    global _file
    if _file is None:
        path = os.environ.get("PROBE_ENTROPY_LOG", "")
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            _file = open(path, "a", buffering=1)  # line-buffered
    return _file


def _get_prompt_file():
    """Open the prompt-event log file (lazy)."""
    global _prompt_file
    if _prompt_file is None:
        path = os.environ.get("PROBE_PROMPT_LOG", "")
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            _prompt_file = open(path, "a", buffering=1)
    return _prompt_file


_PROMPT_HASH_LEN = 32


def log_prompt(request: "Request") -> None:
    """Called once when a request is added. Logs the first/last prompt tokens
    to enable building an oracle lookup of (prompt_hash → completion length)
    after the run completes."""
    f = _get_prompt_file()
    if f is None:
        return
    rid = request.request_id
    with _lock:
        if rid in _prompt_seen:
            return
        _prompt_seen.add(rid)
        ids = list(request.prompt_token_ids)
        record = {
            "req_id": rid,
            "prompt_len": len(ids),
            "prompt_prefix_ids": ids[:_PROMPT_HASH_LEN],
            "prompt_suffix_ids": ids[-_PROMPT_HASH_LEN:],
        }
        f.write(json.dumps(record) + "\n")


def log_step(model_output: "ModelRunnerOutput") -> None:
    """Called once per engine step. Logs entropy for every request in batch."""
    if model_output.token_entropy is None:
        return
    f = _get_file()
    if f is None:
        return

    req_ids = model_output.req_ids
    entropies = model_output.token_entropy
    sampled = model_output.sampled_token_ids  # list[list[int]]

    with _lock:
        for i, req_id in enumerate(req_ids):
            if i >= len(entropies):
                break
            n_new = len(sampled[i]) if sampled and i < len(sampled) else 1
            _output_len[req_id] = _output_len.get(req_id, 0) + n_new
            record = {
                "req_id": req_id,
                "output_len": _output_len[req_id],
                "entropy": entropies[i],
            }
            f.write(json.dumps(record) + "\n")


def mark_finished(req_id: str) -> None:
    """Remove a request from the running length tracker."""
    with _lock:
        _output_len.pop(req_id, None)
        _prompt_seen.discard(req_id)

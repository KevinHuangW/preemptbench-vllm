# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.v1.kv_offload.worker.worker import TransferSpec

ReqId = str


@dataclass
class OffloadingConnectorMetadata(KVConnectorMetadata):
    reqs_to_load: dict[ReqId, TransferSpec]
    reqs_to_store: dict[ReqId, TransferSpec]
    reqs_to_flush: set[str] | None = None
    # Requests proactively pre-offloaded by PROBE before forced eviction.
    # D2H runs asynchronously (no synchronous wait); worker signals completion
    # via finished_sending so the scheduler can then free GPU blocks.
    reqs_to_proactive_store: dict[ReqId, TransferSpec] | None = None

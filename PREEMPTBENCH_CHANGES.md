# PreemptBench modification notice

This repository is derived from vLLM `v0.19.0` at commit
`2a69949bdadf0e8942b7a1619b229cb475beef20`. It is distributed under the
Apache License 2.0 in [LICENSE](LICENSE).

The following files were added or modified for the retained PreemptBench
implementation. This list is also an explicit notice that these files differ
from the upstream commit:

- `docker/Dockerfile`
- `vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py`
- `vllm/transformers_utils/config.py`
- `vllm/utils/deep_gemm.py`
- `vllm/v1/core/block_pool.py`
- `vllm/v1/core/kv_cache_coordinator.py`
- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/core/sched/output.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/core/single_type_kv_cache_manager.py`
- `vllm/v1/engine/core.py`
- `vllm/v1/outputs.py`
- `vllm/v1/probe_logger.py` (added)
- `vllm/v1/probe_preempt_logger.py` (added)
- `vllm/v1/probe_ring.py` (added)
- `vllm/v1/request.py`
- `vllm/v1/sample/sampler.py`
- `vllm/v1/simple_kv_offload/cuda_mem_ops.py`
- `vllm/v1/simple_kv_offload/manager.py`
- `vllm/v1/worker/gpu_model_runner.py`

Together, the changes add experimental preemption-target selection, predictor
state, event logging, and supporting KV-cache/offload instrumentation. The
whole retained runtime tree is preserved for source equivalence, but only the
seven selectors listed in the top-level README were evaluated for the paper.

Release documentation and the dependency-free contract test were added in a
separate commit and do not alter either retained runtime tree.

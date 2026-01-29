---
date: 2026-01-28T14:00:00-08:00
researcher: Antigravity Agent
git_commit: ec4ceda07d5d03bbb6366136d172214156d691d8
branch: main
repository: SchedRL
topic: "SchedRL Framework Adaptation Research"
tags: [schedrl, nemo-rl, roll, miles, skyrl, weight-sync, async-rollout, vllm, sglang]
status: complete
last_updated: 2026-01-28
last_updated_by: Antigravity Agent
---

# Research: SchedRL Framework Adaptation

**Date**: 2026-01-28
**Researcher**: Antigravity Agent
**Git Commit**: ec4ceda07d5d03bbb6366136d172214156d691d8
**Branch**: main
**Repository**: SchedRL

## Research Question
"we want to implement the scheduler in ./design_doc and adapte each framework for it. focus on the related exsiting component and mechanisms to be reused from each framworks refer ./design_doc for more detaild aspects to research. like we shall research model weight sync between train and inference engine, async multiturn agentic rollout loop per trajectory. how the pipeline coordinator control the train and inference workers/gpus. how does it does the offloading of the model weight, kv cache and optimizer. focus on the megatron as trainer and vllm's v1 engine as inference backend if supported, switch to research on sglang as fallback if vllm is not supported. how the requests dispatcher load balance among all rollout dp workers."

## Summary
This research analyzes four target frameworks (NeMo-RL, ROLL, Miles, SkyRL) to identify reusable components for the SchedRL centralized scheduler integration. Key findings include:

1.  **Weight Synchronization**: All frameworks support NCCL broadcast for weight sync. NeMo-RL and SkyRL also support optimized CUDA IPC for single-node setups.
2.  **Async Rollout**: NeMo-RL and SkyRL have the most mature async rollout loops with replay buffers/staleness control. Miles uses a simpler submit-sync-submit pattern. ROLL uses a dedicated `RolloutScheduler`.
3.  **Lifecycle/Offloading**:
    *   **Inference**: NeMo-RL and Miles have explicit "wake/sleep" or "onload/offload" mechanisms for vLLM/SGLang engines, mapping well to SchedRL's `shrink`/`expand`.
    *   **Training**: ROLL uses manual flat-tensor buffering. Miles uses `torch_memory_saver` hooks. SkyRL uses a coordinated dispatch layer.
4.  **Dispatching**: ROLL and Miles have global routers/schedulers suitable for SchedRL's `migration_policy`. NeMo-RL and SkyRL rely more on worker-group abstractions or client-side routing.

## Detailed Findings

### 1. NeMo-RL (Target: Phase 2 "Structural Pilot")
NeMo-RL provides rigid but clear abstractions for SchedRL's "subset" operations.

*   **Model Weight Sync**:
    *   **Mechanism**: Supports both **IPC/ZMQ** (colocated) and **NCCL Broadcast** (non-colocated).
    *   **Key Files**:
        *   `nemo_rl/algorithms/grpo.py`: `refit_policy_generation` selects strategy (lines 936-976).
        *   `nemo_rl/models/policy/lm_policy.py`: `stream_weights_via_ipc_zmq` (line 760) / `broadcast_weights_for_collective`.
    *   **vLLM Integration**: `VllmGeneration` triggers `collective_rpc` on workers to receive weights (`nemo_rl/models/generation/vllm/vllm_generation.py:770`).

*   **Async Rollout Loop**:
    *   **Mechanism**: Uses `AsyncTrajectoryCollector` with a background `_collection_loop`.
    *   **Multi-turn**: `run_async_multi_turn_rollout` (`nemo_rl/experience/rollouts.py:786`) runs generation/env-step loops concurrently.
    *   **Buffering**: Pushes to `ReplayBuffer` with version tags (`generation_weight_version`, `target_weight_version`).

*   **Lifecycle & Offloading**:
    *   **Inference (vLLM)**: `prepare_for_generation` (wake) and `finish_generation` (sleep).
    *   **Sleep**: Calls `self.llm.sleep(level=1)` and resets prefix cache (`nemo_rl/models/generation/vllm/vllm_worker.py:807`).
    *   **Training Offload**: `policy.offload_before_refit()` moves optimizer states to CPU (`nemo_rl/algorithms/grpo.py:937`).

*   **Dispatcher**:
    *   **Routing**: Round-robin dispatch to DP shards (`self.current_generate_dp_shard_idx` in `vllm_generation.py:549`).
    *   **Reusable**: `RayWorkerGroup` (`nemo_rl/distributed/worker_groups.py`) allows executing on specific workers, enabling SchedRL's subset operations.

### 2. ROLL (Target: Phase 1 "Agentic Pipeline")
ROLL has the most sophisticated scheduler-like components (`GenerateScheduler`, `RequestScheduler`) but needs work to support subset-granular operations.

*   **Model Weight Sync**:
    *   **Mechanism**: **NCCL Broadcast** via `ModelUpdateGroup`.
    *   **Planning**: `make_comm_plan` (`roll/distributed/executor/model_update_group.py:33`) creates static communication groups.
    *   **Gap**: Currently broadcasts to *all* inference workers; SchedRL needs subset-scoped groups.

*   **Pipeline Coordination**:
    *   **Mechanism**: `AgenticPipeline.run` (`roll/pipeline/agentic/agentic_pipeline.py:134`) explicitly sequences offload -> sync -> start_server -> collect -> train.
    *   **Reusable**: The `model_update` call (line 157) and `start_server` (line 162) are perfect hook points for SchedRL.

*   **Dispatcher & Abort**:
    *   **Mechanism**: `GenerateScheduler` tracks load (`load_balance_coordinator`) and dispatch.
    *   **Abort**: `RequestScheduler.abort_request` (`roll/distributed/scheduler/generate_scheduler.py:939`) cancels in-flight requests.
    *   **Reusable**: Global abort primitive is key for SchedRL's `migration_policy=REQUEST_RETRY`.

*   **Training Offload**:
    *   **Mechanism**: **Flat-Tensor CPU Buffer**.
    *   **Implementation**: `MegatronTrainStrategy` moves optimizer states/grads to a contiguous CPU buffer to free GPU (`roll/distributed/strategy/megatron_strategy.py:1107`).

### 3. Miles (Target: Phase 4 "SWE-Agent")
Miles is structurally simple with explicit `onload`/`offload` but uses SGLang (vs. vLLM) and needs router extensions.

*   **Async Training Loop**:
    *   **Mechanism**: "Submit-Sync-Submit" pattern in `train_async.py`.
    *   **Overlap**: Overlaps generation of batch N+1 with training of batch N.
    *   **Reusable**: The `train(args)` loop structure is a good baseline for `update_policy=BATCH`.

*   **Lifecycle & Offloading**:
    *   **SGLang**: `RolloutManager.offload` calls `release_memory_occupation` on engines (`miles/ray/rollout.py:176`).
    *   **Training Offload**: Uses **`torch_memory_saver`** library hooks to "pause" the actor and destroy process groups (`miles/ray/actor_group.py:62`).

*   **Model Weight Sync**:
    *   **Mechanism**: `update_weights_from_distributed` (NCCL) or `update_weights_from_tensor`.
    *   **Integration**: `onload_weights` (`miles/ray/rollout.py:191`) wraps the update call.

*   **Dispatcher**:
    *   **Mechanism**: `MilesRouter` (`miles/router/router.py`) proxies requests to SGLang workers.
    *   **Gap**: Has `/add_worker` but missing `/remove_worker` (needed for SchedRL shrink/admission close).
    *   **Load Balancing**: "Least Active Connections" strategy (`min(worker_request_counts)`).

### 4. SkyRL (Reference)
SkyRL provides valuable reference implementations for async staleness control and efficient weight sync.

*   **Async Trainers**:
    *   **Fully Async**: `FullyAsyncRayPPOTrainer` (`skyrl_train/fully_async_trainer.py`) uses `_AsyncStalenessManager` to track capacity.
    *   **One-Step-Off**: `AsyncRayPPOTrainer` (`examples/async/async_trainer.py`) uses an async queue.

*   **Weight Sync**:
    *   **Mechanisms**: `BroadcastTransferStrategy` (NCCL) and `CudaIpcTransferStrategy` (Shared Memory).
    *   **Reusable**: The `weight_sync/` directory has clean abstractions `WeightTransferSender`/`Receiver`.

*   **Training Offload**:
    *   **Mechanism**: **Coordinated Dispatch**. `WorkerDispatch` (`skyrl_train/workers/worker_dispatch.py`) tracks GPU state and automatically moves models/optimizers to CPU (`offload_to_cpu`) when not in use.

## Architecture Documentation: Cross-Framework Comparison

### Weight Sync Mechanisms
| Framework | Primary Mechanism | Sync Granularity | Code Reference |
| :--- | :--- | :--- | :--- |
| **NeMo-RL** | IPC/ZMQ or NCCL | Subset-ready (via `WorkerGroup`) | `nemo_rl/algorithms/grpo.py:936` |
| **ROLL** | NCCL Broadcast | Static Group (All Workers) | `roll/distributed/executor/model_update_group.py` |
| **Miles** | NCCL or Tensor | Cluster-wide | `miles/ray/actor_group.py` |
| **SkyRL** | NCCL or CUDA IPC | Flexible Strategy | `skyrl_train/weight_sync/` |

### Trainer Offloading
| Framework | Mechanism | Granularity | Code Reference |
| :--- | :--- | :--- | :--- |
| **NeMo-RL** | Explicit CPU Move | Optimizer States | `nemo_rl/algorithms/grpo.py:937` |
| **ROLL** | Flat-Tensor Buffer | Params, Optim, Grads | `megatron_strategy.py:1107` |
| **Miles** | `torch_memory_saver` | Full Actor State | `miles/ray/actor_group.py:62` |
| **SkyRL** | Coordinated Dispatch | Model vs Optimizer | `worker_dispatch.py:24` |

### Dispatching & Routing
| Framework | Router Type | Load Balancing | Abort Support | Code Reference |
| :--- | :--- | :--- | :--- | :--- |
| **NeMo-RL** | Client-side (Round Robin) | DP Shard Round Robin | No global abort (wait for drain) | `vllm_generation.py:549` |
| **ROLL** | Global Scheduler | Least Loaded | Yes (`abort_request`) | `generate_scheduler.py:188` |
| **Miles** | HTTP Proxy Router | Least Active Requests | Per-worker URL | `miles/router/router.py` |
| **SkyRL** | Client-side (Hashing) | Session ID Hashing | Yes (`abort_generation`) | `inference_engine_client.py:97` |

## Open Questions
1.  **ROLL Subset Sync**: How difficult is it to modify `ModelUpdateGroup` to support subset-scoped NCCL groups? (Critical for SchedRL)
2.  **Miles Router**: Can we easily add `/remove_worker` to `MilesRouter` to support safe shrink?
3.  **NeMo-RL Abort**: Can we expose an abort primitive in `VllmGeneration` to support `REQUEST_RETRY` migration, or must we rely solely on "wait for drain"?


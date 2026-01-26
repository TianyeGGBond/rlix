# rLLM (VeRL) Adaptation Plan (Archived)

Status: archived / out of scope.

Reason: the current rLLM+VeRL integration in this repo does not provide the async training modes
needed by the shared SchedRL protocol (e.g., fully-async with staleness control, one-step-off
pipelining, and elastic subset shrink/expand with migration). It is effectively limited to
on-policy / batch-scoped behavior in practice, so we are migrating to SkyRL for async modes.

## 1. Overview
rLLM (backed by VeRL) is the Phase 3 target, focusing on "Logic Hardening". It has the most mature Agentic architecture but requires complex logic changes to support step-level retries essential for preemption handling.
We keep the core training loop intact and introduce a **proxy layer** to intercept framework-specific operations (e.g., rollout/generation controllers). The pipeline coordinator calls the central scheduler **directly**; the proxy is unidirectional (wrapper only) and emits release ACKs.

## 1.1 Protocol Fit (Against `multi-pipeline-adaptation-plan_clean.md`)

This section reality-checks rLLM (VeRL-backed) against the shared protocol in `design_doc/multi-pipeline-adaptation-plan_clean.md`.

**Already present in the codebase**
- **Rollout engine lifecycle**: rLLM has `wake_up()` / `sleep()` on the rollout engine (`rllm/rllm/engine/rollout/verl_engine.py`), which maps to “pause/resume” and can be used for `BATCH`-style boundaries.
- **Batch-style wake/sleep integration**: VeRL agent loop explicitly calls `wake_up()` before generation and `sleep()` after, with notes about ensuring weight sync (`verl/verl/experimental/agent_loop/agent_loop.py`).

**Gaps / required extensions for elastic shrink/expand**
- **Subset lifecycle (indices)**:
  - rLLM `wake_up()` / `sleep()` currently affect *all* rollout replicas; the shared protocol needs subset `indices=...` to support `expand_workers` / `shrink_workers`.
- **Mid-flight shrink migration (`REQUEST_RETRY`)**:
  - rLLM already has retry loops in its higher-level engines (e.g., `AgentWorkflowEngine.process_task_with_retry`, `AgentSdkEngine.process_task_with_retry`) that can restart work from scratch on failure.
  - What’s missing is wiring preemption into those existing retry paths: when a subset is shrunk, in-flight work on that subset must fail/cancel in a way that triggers retry, and queued work must be redistributed onto the remaining active workers.
  - Required minimal extension: add an explicit “preempt signal” that:
    - closes admission to the shrinking subset (do not schedule new tasks to it),
    - cancels in-flight tasks tied to that subset (raising a retryable error / cancellation), and
    - re-enqueues the task back into the existing workflow/task queue so the next retry attempt runs on the remaining active subset.
- **Admission control**:
  - For scheduler-driven time-sharing, we need a clear “do not start new work on preempted subset” mechanism (at least at the task dispatcher level).
- **Version tagging**:
  - The shared protocol requires `generation_checkpoint_version` tagging for any overlap modes; rLLM needs to attach the active checkpoint version to produced trajectories/samples.
- **Progress reporting**:
  - The shared scheduler needs `remaining`, `percent_remaining`, and `oldest_unfinished_creation_ts` heartbeats. rLLM currently has task/trajectory queues, but the standardized heartbeat emission points are not implemented.

**Concrete file refs & immediate actions**
- Files: `rllm/rllm/engine/agent_execution_engine.py`, `verl/verl/experimental/agent_loop/agent_loop.py`, `verl/single_controller/ray/base.py`.
- Actions:
  - Add `wake_up(indices)` / `sleep(indices)` variants in `VerlEngine` and propagate indices through the dispatcher to support subset-level operations.
  - Add heartbeat emission at batch-start in `AgentExecutionEngine.execute_tasks()` and `trajectory_generator()` to report `remaining`, `percent_remaining`, and `oldest_unfinished_creation_ts`.

**Recommended baseline mapping**
- `update_policy = BATCH` for batch-scoped trainers (wake/sleep boundary).
- `update_policy = QUIESCE` for FullyAsync modes (if rollouts and training are decoupled and need a drain boundary).
- `migration_policy = REQUEST_RETRY` (required for time-sharing): cancel tasks on shrinking subset and retry from scratch via existing rLLM retry loops.
- `expand_rebalance_policy = REBALANCE_QUEUED` (enabled by default): queued work is redistributed naturally by the dispatcher once new workers are active.

**Concise actionable items (merged from `design_doc/archive/adaptation_review.md`)**
- Add subset-targeting APIs (`wake_up(indices)`, `sleep(indices)`) and subset-scoped init/offload primitives (e.g., `init_model(indices=...)`, `offload_weights(indices=...)`) in the relevant VeRL/rLLM layers.
- Wire standardized progress heartbeats from `AgentExecutionEngine` task completion / queue updates.

## 2. Existing Code Integration Points (Pre-Adaptation)

### 2.1 Training Entry Point
*   **File**: `verl/trainer/ppo/ray_trainer.py` / `AgentPPOTrainer.train()`
*   **Hook**: Before/after `update_actor()` / `update_critic()`.

### 2.2 Generation Entry Point
*   **File**: `rllm/engine/agent_execution_engine.py`
*   **Method**: `AgentExecutionEngine.execute_tasks()`
*   **Hook**: Initialize rollout workers or sync weights inside the engine loop.

### 2.3 Weight Sync (Pre-Adaptation)
*   **Mechanism**: VeRL worker group broadcast (IPC/NCCL).
*   **Hook**: Sync/broadcast is invoked in the existing loop; it is not sync-on-resume.

## 3. Architecture Mapping

### 3.1 Component Mapping
| Design Doc Concept | rLLM/VeRL Implementation | Key Classes |
|-------------------|--------------------------|-------------|
| **Agent Abstraction** | `BaseAgent` | `rllm/agents/base.py` |
| **Trajectory Execution** | `AgentExecutionEngine` | `agent_execution_engine.py` |
| **Pipeline Wrapper** | `AgentTrainer` / `AgentPPOTrainer` | `agent_trainer.py` |
| **Worker Management** | VeRL `RayWorkerGroup` | `verl/single_controller/ray/base.py` |
| **Training Cluster** | `actor_rollout_wg` | `RayPPOTrainer` attribute |
| **Inference Backend** | vLLM/SGLang/OpenAI | External endpoints |
| **Rollout Buffer** | `Trajectory` / `DataProto` | In-memory collection |

### 3.2 Lifecycle Operations Mapping (Post-Adaptation)

**Terminology Distinction:**
- **Cluster-Level Operations**: Enable or disable the entire cluster (all workers).
- **Subset-Level Operations (Required)**: Activate or deactivate specific DP workers. These require the extensions described in Section 5.2.

| Design Doc Verb | rLLM/VeRL Implementation | Method / Action |
|-----------------|--------------------------|-----------------|
| **expand (cluster)** | VeRL rollout init | Wakes ALL rollout workers |
| **shrink (cluster)** | VeRL rollout shutdown | Sleeps/Restarts ALL workers |
| **expand (subset)** | `VerlEngine.wake_up(indices=...)` (requires extension) | Wakes specific workers |
| **shrink (subset)** | `VerlEngine.sleep(indices=...)` (requires extension) | Sleeps specific workers |
| **offload (DP)** | VeRL worker reset | Clears GPU state |
| **load/backload** | Worker init + Broadcast | Re-initializes model |
| **sync (weights)** | Broadcast for active workers | VeRL collective broadcast |
| **broadcast** | IPC/NCCL | VeRL primitives |

### 3.3 Progress/Heartbeat Mapping
| Metric | rLLM Implementation |
|--------|---------------------|
| **remaining** | `AgentExecutionEngine` task queue |
| **percent_remaining** | Engine task completion % |
| **oldest_unfinished** | Oldest task creation timestamp (needs implementation in engine) |

### 3.4 Preemption & Release Protocol (Post-Adaptation)
| Protocol | rLLM/VeRL Implementation |
|----------|--------------------------|
| **request_gpus (train)** | Coordinator calls central scheduler; training uses existing VeRL worker groups |
| **release_gpus (train)** | Coordinator calls central scheduler; VeRL FSDP offload |
| **request_gpus (gen)** | Coordinator calls central scheduler; scheduler triggers `wake_up(indices)` via proxy |
| **release_gpus (gen)** | Coordinator calls central scheduler; scheduler triggers `sleep(indices)` via proxy |
| **preempt gen** | `sleep(indices)` + cancel pending/in-flight tasks on those indices + re-enqueue tasks for retry (retry the current task/turn on remaining active workers) |
| **resume gen** | Scheduler triggers sync-on-resume then `wake_up(indices)` |

## 4. VeRL Component Reference
Key VeRL classes that the rLLM adapter interacts with (from `verl/` codebase):

| VeRL Component | File Location | SchedRL Mapping | Notes |
|----------------|---------------|-----------------|-------|
| `RayPPOTrainer` | `verl/trainer/ppo/ray_trainer.py` | Pipeline Coordinator | Main training loop via `fit()`, manages all worker groups |
| `ResourcePoolManager` | `verl/trainer/ppo/ray_trainer.py` | GPU pool abstraction | Creates `RayResourcePool` per role, maps roles to resource pools |
| `RayResourcePool` | `verl/single_controller/ray/base.py` | Resource pool | Manages Ray placement groups, GPU bundles |
| `RayWorkerGroup` | `verl/single_controller/ray/base.py` | Cluster controller | Manages distributed workers, supports `spawn()`, `init_model()` |
| `actor_rollout_wg` | `RayPPOTrainer` attribute | Training + rollout workers | Colocated actor/rollout in hybrid engine mode |
| `critic_wg` | `RayPPOTrainer` attribute | Critic workers | Separate worker group for value model |
| `ref_policy_wg` | `RayPPOTrainer` attribute | Reference policy workers | Optional, for KL penalty computation |
| `hybrid_engine` | `RayPPOTrainer` config | GPU sharing mode | When True, training and rollout share workers |
| `DataProto` | `verl/protocol.py` | Rollout data structure | Batched data with tensor + non-tensor fields |

## 5. Required Extensions

### 5.1 Step-Level Retry Logic (Critical)
*   **Status**: Not part of the baseline adaptation plan.
*   **Note**: rLLM has internal retry logic today (`run_agent_trajectory_with_retry` and workflow/sdk retry loops). For SchedRL, we reuse these existing retries and add minimal preemption wiring so mid-flight shrink triggers retry on the remaining active workers (retry the current task/turn; does not imply restarting an entire multi-turn trajectory if context is preserved).

### 5.2 DP-Granular Selective Execution
*   **Status**: Missing.
*   **Analysis**: `VerlEngine.wake_up` targets all replicas.
*   **Required Action**: Modify `wake_up` and `sleep` in `verl_engine.py` to accept `indices` and filter `self.rollout_manager.rollout_replicas`.

### 5.3 Scheduler Progress Hooks
*   **Integration Point**: `AgentExecutionEngine`.
*   **Trigger**: After `self.buffer.extend(...)` (or equivalent result collection) and at batch start.
*   **Action**: Inject `scheduler.report_progress(remaining, percent_remaining, oldest_unfinished_creation_ts)`.
*   **Frequency**: Report at **batch start** and whenever `percent_remaining` crosses a **2% progress band** (event-driven).
*   **Batch-start entrypoint**: At the start of `AgentExecutionEngine.execute_tasks()` (after `total = len(tasks)` and before launching `asyncio.gather(...)`).

### 5.4 Native Request Migration During Stop (Framework-Specific)
*   **rLLM/VeRL Native Pattern**: `executor.shutdown(wait=False, cancel_futures=True)`.
*   **Behavior**:
    1.  `AgentExecutionEngine` uses a `ThreadPoolExecutor` for concurrent trajectory execution.
    2.  On shutdown, `cancel_futures=True` cancels pending asyncio tasks (lines 552, 610 in `agent_execution_engine.py`).
    3.  In-flight async tasks are cancelled; pending tasks are not retried automatically (application-level decision).
*   **SchedRL Integration**: Wire preemption into existing retry loops: on `sleep(indices)` treat cancellations as retryable failures and re-enqueue tasks so they are retried on the remaining active workers (typically retry the current task/turn; does not imply restarting an entire multi-turn trajectory if context is preserved).

### 5.5 Minimal Mid-Flight Shrink/Expand Checklist (Implementation-Ready)

Goal: implement `migration_policy=REQUEST_RETRY` by reusing rLLM’s existing retry loops (workflow/sdk engines) and adding minimal preemption wiring + subset routing.

**Shrink (mid-flight) — required**
- Subset lifecycle: add `wake_up(indices)` / `sleep(indices)` variants in `VerlEngine` and ensure the dispatcher knows which worker indices are active.
- Admission control: when shrinking `P`, stop assigning new tasks/episodes to `P` immediately.
- Cancel in-flight on `P`: cancel futures tied to `P` (or force the underlying rollout call to raise a retryable exception).
- Retry from scratch (reuse existing code): ensure the cancellation path is treated as retryable in:
  - `AgentWorkflowEngine.process_task_with_retry(...)`, and/or
  - `AgentSdkEngine.process_task_with_retry(...)`.
  The retry attempt must be able to pick a different active worker index.

**Expand (default-enabled rebalance) — optional migration**
- After `wake_up(indices=A)` + weight sync, the dispatcher should start assigning new tasks to `A`.
- Optional queued rebalance is naturally handled by the workflow/task queue if it always schedules to “any active worker”; no in-flight cancellation required on expand.
- Optional (aggressive): rebalance in-flight tasks on expand only if explicitly enabled.
  - Preferred: migrate/resume task/trajectory state (future; depends on workflow semantics).
  - Fallback: cancel+retry via existing retry loops (task-level retry; may re-run the current task step).

## 6. Post-Adaptation Integration Overview
This section describes how reused rLLM components and required extensions implement the protocol in `design_doc/multi-pipeline_roll_old_design.md`.

### 6.0 Proxy Layer (Framework Interception)
*   **Purpose**: Wrap rollout controls to emit **release ACKs** without changing core behavior.
*   **Behavior**: The proxy forwards calls by default and only injects release notifications; it does **not** mediate scheduler decisions.
*   **Minimal intrusion**: The pipeline coordinator (`AgentPPOTrainer.fit`) keeps its core logic and calls the central scheduler directly at phase boundaries. Progress reporting must be added (Section 5.3/5.5) so the scheduler can make better time-sharing decisions.

### 6.1 Pipeline Coordinator ↔ Central Scheduler
*   **Who is the pipeline coordinator?** The `AgentPPOTrainer.fit()` loop in `verl/trainer/ppo/ray_trainer.py`.
*   **Request/Release (Training)**: The coordinator calls the **central scheduler API** directly.
*   **Request/Release (Generation)**: The coordinator calls the **central scheduler API** directly; the scheduler triggers `VerlEngine.wake_up(indices)` / `sleep(indices)` via the proxy.
*   **Preempt/Resume**: The scheduler initiates preempt/resume; the proxy executes sleep/wake, and sync-on-resume runs before restarting the subset.
*   **Control split**: The scheduler initiates expand/shrink; the coordinator only requests/releases; the proxy executes expand/shrink on rollout workers.
*   **Sync timing change**: Post-adaptation, sync moves to **sync-on-resume** (right before generation resumes).
*   **Async training constraint**: For async training setups (where rollout overlaps with training), rollout **must be stopped after each training step** before weight sync can proceed. The scheduler coordinates this stop before triggering sync-on-resume.
*   **Versioning**: Keep only the **latest** CPU weight cache (by `global_step`) for sync-on-resume.

### 6.2 Cluster Controller ↔ Scheduler (DP-Granular)
*   **Expand/shrink semantics**: `expand` resumes workers via `wake_up(indices)`, `shrink` preempts via `sleep(indices)`.
*   **Selective sync**: VeRL's `init_model` or broadcast must be scoped to the active subset group; rebuild IPC/NCCL groups on resume and tear down after update.

### 6.3 Rollout Progress ↔ Scheduler Heartbeats
*   **Integration point**: Section 5.3 (2% progress-band).
*   **Tie-break**: `oldest_unfinished` timestamp from the task queue.
*   **Release ACK**: After normal generation release, the proxy (wrapping rollout controls in `AgentExecutionEngine` / `VerlEngine`) notifies the central scheduler (`notify_cluster_released`) before new preemption decisions are applied.

## 7. Implementation Steps (Phase 3)
1.  **Patch `VerlEngine`**: Add subset support to `wake_up`/`sleep` and `init_model`.
2.  **Proxy Layer**: Implement a lightweight proxy to emit release ACKs and execute scheduler-initiated expand/shrink.
3.  **Inject Hooks**: Add batch-start + 2% band progress reporting to `AgentExecutionEngine`.
4.  **Preemption retry wiring**: cancel+re-enqueue tasks on subset shrink; confirm retries land on remaining active subset.
5.  **Verify**: Validate mid-flight shrink/expand and bounded staleness.

## 8. Arbitrary Placement & Selective Sync
Placement is arbitrary at config time via manual Ray placement groups, then fixed for the run; the scheduler only controls active subsets within that fixed placement.
Selective sync rebuilds IPC/NCCL groups on resume for the active subset only.

## 9. Configuration Example
**rLLM** (`rllm_config.yaml`):
```yaml
trainer:
  backend: verl
schedrl:
  enabled: true
  scheduler_name: "CentralizedGPUScheduler"
  pipeline_id: "rllm_pipeline_0"
```

## 10. Framework Maturity
*   **Rank**: Most Mature (Agentic Features).
*   **Features**: Step-level MC estimation, Trajectory rewards.

## 11. Estimated Effort
*   **Complexity**: Medium/High (subset lifecycle + selective sync wiring).
*   **Size**: ~200–400 LOC.

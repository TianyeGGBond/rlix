---
date: 2026-02-04T19:19:15-05:00
researcher: tao
git_commit: 4ce7520e2cb20cc32dbfee71f5abca754a2c99ab
branch: main
repository: SchedRL
topic: "ROLL_multi_pipeline: multi-pipeline GPU time-sharing implementation map"
tags: [research, codebase, roll, multi-pipeline, scheduler, ray, selective-model-update]
status: complete
last_updated: 2026-02-04
last_updated_by: tao
---

# Research: ROLL_multi_pipeline multi-pipeline GPU time-sharing implementation map

**Date**: 2026-02-04T19:19:15-05:00
**Researcher**: tao
**Git Commit**: 4ce7520e2cb20cc32dbfee71f5abca754a2c99ab
**Branch**: main
**Repository**: SchedRL

## Research Question
Focus on the last commit under `third_party/ROLL_multi_pipeline` as an implementation of `design_doc/archive/multi-pipeline_roll_old_design.md`: understand the design and key components, specifically how multi-pipeline GPU time-sharing is designed, including central scheduler, model weight sync service, shrink/expand + selective model update, and distributed coordination between pipeline coordinator and scheduler.

## Summary
`third_party/ROLL_multi_pipeline` implements a hierarchical control plane for time-sharing a shared GPU pool across multiple concurrent RL pipelines using Ray named actors. A detached `CentralizedGPUScheduler` arbitrates GPU ownership across pipelines, while each pipeline’s `ConcurrentAgenticPipeline` (the pipeline coordinator) requests/releases compute for non-generation roles and interacts with a per-pipeline generation control plane (`RolloutScheduler` + `RequestScheduler`) that supports scheduler-driven shrink/expand at DP-worker granularity. Model weight propagation during resume/expand is supported by a per-pipeline `ModelUpdateService` that caches sender weights by `global_step` and applies them selectively to re-activated inference DP ranks.

## Detailed Findings

### 1) Control-plane topology (actors + responsibilities)

**Central scheduler (global, detached)**
- `roll/distributed/scheduler/centralized_gpu_scheduler.py`
  - `CentralizedGPUSchedulerImpl` is the core state machine.
  - Tracks:
    - `pending_requests: Dict[Priority, List[PendingRequest]]`
    - `active_allocations: Dict[str, ClusterAllocation]`
    - `idle_gpus: Set[int]`
    - `pending_completion_requests: Dict[str, PendingCompletionRequest]` (for completion-driven generation suspension)
  - Runs background `_central_scheduling_loop()`; wakeups occur on new request, release, or completion notifications.

**Pipeline coordinator (per pipeline)**
- `roll/pipeline/agentic/concurrent_agentic_pipeline.py`
  - `ConcurrentAgenticPipeline` is a Ray actor that runs the per-pipeline training loop.
  - Calls the scheduler via:
    - `_request_gpu_and_wait(..., gpu_scheduler.request_gpus(...))`
    - `_release_gpu(..., gpu_scheduler.release_gpus(...))`
    - `_release_and_request_gpu_blocking(..., gpu_scheduler.release_and_request_gpus(...))`
  - Owns phase progression and enforces “request → execute → release” for non-generation clusters.

**Generation-side control plane (per pipeline)**
- `roll/distributed/scheduler/rollout_scheduler.py`
  - `RolloutScheduler` Ray actor constructs:
    - `GroupQueueManager` (named `{pipeline_id}_group_queue_manager_{mode}`)
    - `RequestScheduler` (named `{pipeline_id}_request_scheduler_{mode}`)
  - `GroupQueueManager` tracks rollout collection progress and reports it to the centralized scheduler.
- `roll/distributed/scheduler/generate_scheduler.py`
  - `RequestScheduler` is the execution-side controller for rollout workers:
    - Maintains `active_dp_ranks: Set[int]` for partial DP activation.
    - Implements `shrink_workers()` / `expand_workers()` called by the centralized scheduler.
    - Aborts in-flight requests and clears sticky routing mappings during shrink to enable retry/reroute.

**Orchestrator (multi pipeline admission + global singleton actors)**
- `roll/pipeline/agentic/multi_pipeline_orchestrator.py`
  - Creates detached named actors:
    - `ResourceManager` (`RESOURCE_MANAGER_NAME`, `RAY_NAMESPACE`)
    - `CentralizedGPUScheduler` (`CENTRALIZED_GPU_SCHEDULER_NAME`, `RAY_NAMESPACE`)
  - Registers each pipeline’s clusters (`tp_size`, `device_mapping`) with the centralized scheduler.

### 2) Actor discovery and naming (Ray namespace protocol)

**Centralized scheduler discovers per-pipeline progress sources**
- `centralized_gpu_scheduler.py` lazily resolves `GroupQueueManager` by name:
  - `_get_group_queue_manager(pipeline_id, mode="train")` → `ray.get_actor(f"{pipeline_id}_group_queue_manager_{mode}", namespace=registry_entry['namespace'])`

**Per-pipeline rollout components are created with pipeline-scoped names**
- `rollout_scheduler.py`
  - `GroupQueueManager.options(name=f"{pipeline_id}_group_queue_manager_{mode}", namespace=RAY_NAMESPACE).remote(...)`
  - `RequestScheduler.options(name=f"{pipeline_id}_request_scheduler_{mode}", namespace=RAY_NAMESPACE).remote(...)`

**Central scheduler discovery scope**
- `CentralizedGPUScheduler.register_pipeline(..., ray_namespace=RAY_NAMESPACE)` stores the namespace per pipeline.

### 3) GPU allocation model: cluster allocations + DP-worker atomicity

**Allocation state structure**
- `roll/distributed/scheduler/gpu_scheduler_types.py`
  - `ClusterAllocation` includes:
    - `gpu_ids: List[int]` (cluster-level allocation)
    - `active_dp_ranks: Set[int]` (for generation clusters, can be empty when suspended)
    - `dp_rank_to_gpus: Dict[int, List[int]]` (explicit DP-rank → GPU bundle mapping)

**DP-rank bundle mapping inside RequestScheduler**
- `roll/distributed/scheduler/generate_scheduler.py`
  - `_get_gpus_for_dp_rank(dp_rank)` maps `infer_cluster.rank2devices[dp_rank]` → global GPU IDs as:
    - `gpu_id = node_rank * gpu_per_node + gpu_rank`

**Shrink/expand at DP-worker boundaries**
- `RequestScheduler.shrink_workers(target_gpus)` computes `offload_ranks` whose bundles overlap `target_gpus`.
- `RequestScheduler.expand_workers(target_gpus, ...)` computes `load_ranks` whose bundles are fully contained in `target_gpus`.

### 4) Request/release protocol: non-generation vs generation (completion-driven)

**Non-generation clusters: explicit request + explicit release**
- `ConcurrentAgenticPipeline` uses `CentralizedGPUScheduler.request_gpus()` and `release_gpus()` for roles like `actor_train`, `critic`, `reference`.
- `centralized_gpu_scheduler.py: release_gpus(cluster_id, global_step)` validates same-step (non-generation) release semantics using `ClusterAllocation.global_step`.

**Generation cluster (`*_actor_infer`): scheduler-driven shrink/expand and completion-driven suspension**
- Central scheduler supports blocking completion notifications:
  - `CentralizedGPUSchedulerImpl.notify_cluster_released(cluster_id, allocation_id, global_step)`
  - Enqueues a `PendingCompletionRequest`, then blocks on an event that is signaled at Phase 6.
- Pipeline coordinator triggers completion notification via RequestScheduler:
  - `ConcurrentAgenticPipeline.run()` (Phase 1 in the reorganized run loop) calls:
    - `self.request_scheduler.notify_ready_to_release.remote(global_step=global_step-1)`
- `RequestScheduler.notify_ready_to_release(global_step)` calls:
  - `CentralizedGPUScheduler.notify_cluster_released(cluster_id=f"{pipeline_id}_actor_infer", allocation_id=..., global_step=...)`

**Central scheduling loop phases**
- `centralized_gpu_scheduler.py: scheduling_cycle()` performs:
  - Phase 0: `_process_completion_notifications(plan)` → emits `CompletionSuspensionOp` for generation
  - Phase 2: non-generation planning (may shrink generation to satisfy higher priorities)
  - Phase 3: generation planning using gap-ratio (allocates/resumes DP workers)
  - Phase 5: executes shrink/allocation/expansion RPCs
  - Phase 6: updates `idle_gpus` and `active_allocations`, signals pending completion events

Note (correction): The codebase contains an episode‑timestamp query implementation that would call `GroupQueueManager.get_oldest_created_at()` (helpers `_dedup_and_query_timestamps` / `_query_single_timestamp` exist in `centralized_gpu_scheduler.py` at ~lines 1560 and 1616), but the active scheduling loop does not invoke those helpers. In the live `scheduling_cycle()` Phase 1 the scheduler currently calls `_fifo_sorted_pending_and_active_cluster()` (defined ~line 1540) which uses request/allocation arrival timestamps (not GroupQueueManager episode timestamps) and the returned list is not used to drive generation planning. The actual generation planner invoked by the cycle is the gap‑ratio planner `_plan_generation_gap_ratio_alternative(...)` (defined ~line 2723 and called in scheduling_cycle Phase 3 at ~line 1110), which uses progress metrics (`_latest_progress` → `remaining`, `percent_remaining`) to compute target GPU counts and gaps. In short: episode‑age FIFO is implemented in code but is not the active mechanism; gap‑ratio + progress reports drive generation planning. (See `third_party/ROLL_multi_pipeline/roll/distributed/scheduler/centralized_gpu_scheduler.py` for exact helpers and line ranges.)

### Phase → Code mapping (central scheduler)

The following maps each scheduling phase (Phase 0..6) to the concrete helpers in the implementation so readers can jump to the code quickly. Line ranges are approximate anchors within `third_party/ROLL_multi_pipeline/roll/distributed/scheduler/centralized_gpu_scheduler.py`.

- Scheduling entry points
  - `_central_scheduling_loop()` - background loop that triggers cycles
    - file: `third_party/ROLL_multi_pipeline/roll/distributed/scheduler/centralized_gpu_scheduler.py`
    - approx lines: 960–1016
  - `scheduling_cycle(cycle_num: int = -1)` - orchestrates Phases 0..6
    - file: same
    - approx start line: 1032

- Phase 0: CompletionNotification processing
  - helper: `_process_completion_notifications(self, plan: ExecutionPlan)`
  - file: `.../centralized_gpu_scheduler.py`
  - approx lines: 1320–1560
  - short: drains `pending_completion_requests`, creates `CompletionSuspensionOp`, updates `planned_available_gpus`, marks `plan.clusters_to_remove`.

- Phase 1: Dedup & timestamp querying (design vs active path)
  - active helper invoked: `_fifo_sorted_pending_and_active_cluster(self)`
    - file: `.../centralized_gpu_scheduler.py`
    - approx lines: 1540–1560
    - short: builds cluster_id → timestamp using pending request/alloc arrival timestamps (request.timestamp / alloc.timestamp).
  - legacy timestamp-query helpers (implemented but not called by `scheduling_cycle()`):
    - `_dedup_and_query_timestamps(self)` — approx line 1560
    - `_query_single_timestamp(self, cluster_id, max_retries=3)` — approx line 1616 (calls `GroupQueueManager.get_oldest_created_at()`)
    - short: these would query per-pipeline episode timestamps but are not used by the live loop (heartbeat/push + gap-ratio is the active design).

- Phase 2: Non-GENERATION planning (priority 0–5)
  - helper: `_plan_non_generation(self, plan: ExecutionPlan)`
  - file: `.../centralized_gpu_scheduler.py`
  - approx lines: 2180–2280
  - short: full-allocation non-GEN requests, atomic shrink-allocate transactions, updates `planned_available_gpus`.

- Phase 3: GENERATION planning (active: gap‑ratio planner)
  - snapshot helper: `_snapshot_generation_state_after_non_gen(self, plan)`
    - file: `.../centralized_gpu_scheduler.py`
    - approx lines: 2400–2720
    - short: builds `active_dp_workers`, `inactive_dp_workers`, `non_gen_reserved_gpus`, `idle_gpus` for generation planning.
  - active planner: `_plan_generation_gap_ratio_alternative(self, plan, *, active_dp_workers, inactive_dp_workers, non_gen_reserved_gpus, idle_gpus, epsilon=0.0)`
    - file: `.../centralized_gpu_scheduler.py`
    - approx start line: 2723
    - short: gap‑ratio planner using `self._latest_progress` (progress heartbeats: `remaining`, `percent_remaining`) to compute target GPU counts, gaps, and produce `SchedGuidedAllocationOp` / `SchedGuidedShrinkOp`.
  - legacy FIFO planner (present but not used by `scheduling_cycle()`): `_plan_generation_fifo(self, timestamp_list, plan)` — approx line 2538 (would consume Phase 1 episode timestamps).

- Phase 4: Validation
  - call-site: `validate_execution_plan(...)` inside `scheduling_cycle()` after plan construction
    - validator: `third_party/ROLL_multi_pipeline/roll/distributed/scheduler/gpu_scheduler_validation.py::validate_execution_plan`
    - approx lines in validator file: 880–980
    - short: structural + simulated-state validations (double-free, DP overlap, device mapping, etc.)

- Phase 5: Execution (RPCs)
  - helpers and roles:
    - `_batch_shrink_operations(plan)` — Phase 5 Step 1 (batch per cluster)
      - approx line: 3058
    - `async def _execute_shrinks(self, batched_shrinks)` — Phase 5 Step 2: call `request_scheduler.shrink_workers.remote(target_gpus)` (concurrent)
      - approx start line: 3160
    - `async def _execute_allocations(self, plan)` — Phase 5 Step 3: signal pending requests (`pending_req.result` + `pending_req.event`)
      - approx lines: 3240–3300
    - `async def _execute_expansions(self, plan)` — Phase 5 Step 4: call `request_scheduler.expand_workers.remote(...)` (concurrent)
      - approx lines: 3300–3360
  - short: executes RPCs to per-pipeline `RequestScheduler` actors (shrinks, allocation signals, expansions), guarded by timeouts.

- Phase 6: State update (commit)
  - helpers and roles:
    - `_update_idle_gpus(plan, batched_shrinks)` — Phase 6 Step 2: update `self.idle_gpus` (add freed GPUs, remove allocated/expanded GPUs)
      - approx lines: 3680–3880
    - `_signal_pending_completions(plan)` — Phase 6 Step 1.5: set events for completion requests
      - approx same area (~3680+)
    - `_update_active_allocations(plan, batched_shrinks)` — Phase 6 Step 1: update `self.active_allocations` (remove, shrink, expand, add)
      - approx lines: 3360–3720
    - `_remove_processed_pending_requests(plan)` — Phase 6 Step 3: remove signaled pending requests and completion requests
      - approx lines: 3720–3760
  - short: commits plan changes, signals waiting callers, runs final consistency checks (e.g., `_validate_request_scheduler_active_dp_ranks_match`).

Notes / pointer
- All file paths above are rooted at `third_party/ROLL_multi_pipeline/roll/distributed/scheduler/centralized_gpu_scheduler.py` (helpers are in that file). The line numbers are approximate anchors to help navigation — open the file and search for the helper names to get exact positions in your copy.

### 5) Shrink/expand mechanics: preemption, routing, and suspension

**Shrink path**
- `RequestScheduler.shrink_workers(target_gpus)`:
  - Optionally calls `suspend()` when shrinking to zero active ranks.
  - Aborts in-flight requests on shrinking ranks and clears sticky `src_rank → dp_rank` mappings (so work is retried/rerouted).
  - Offloads model states on the target dp ranks via `infer_cluster.offload_states_partial(...)`.

**Expand path**
- `RequestScheduler.expand_workers(target_gpus, skip_load=False, global_step=..., selective_update=...)`:
  - Loads states on `load_ranks` via `infer_cluster.load_states_partial(active_dp_ranks=load_ranks, start_server_thread=True, server_meta_info={...})`.
  - Updates routing by adding to `active_dp_ranks` and optionally calls `resume()` when expanding from zero.

### 6) Progress reporting used by generation planning (heartbeat-style)

**Progress source**
- `rollout_scheduler.py: GroupQueueManager._maybe_emit_progress()` computes:
  - `remaining`, `percent_remaining`, `oldest_unfinished_creation_ts`, etc.
  - Emits in ~2% “progress buckets” (bucket = `floor(percent_remaining * 50)`) or at batch start/completion.
  - Calls `CentralizedGPUScheduler.report_progress.remote(progress)`.

**Progress sink**
- `centralized_gpu_scheduler.py: report_progress(progress)` stores the latest progress per `(pipeline_id, mode)` in `_latest_progress`.

### 7) Model weight sync service: cache + selective update on resume

**Service actor**
- `roll/distributed/executor/model_update_service.py`
  - `ModelUpdateService` is a per-pipeline Ray actor named `{pipeline_id}_model_update_service`.
  - Maintains:
    - `cached_global_step`
    - `_cached_pp_ranks`
    - `_num_buckets`

**Cache creation during init (step 0)**
- `concurrent_agentic_pipeline.py` builds the initial cache (if `ROLL_SELECTIVE_MODEL_UPDATE_CONCURRENT=1`) on `actor_train` dp0 ranks:
  - `actor_train.rank2worker[i].build_model_update_bucket_cache.remote(global_step=0, bucket_size=...)`
  - Then registers cache metadata with `ModelUpdateService.register_sender_cache(global_step=0, pp_ranks=[...], num_buckets=...)`.

**Selective update during expand/resume**
- `generate_scheduler.py: RequestScheduler._get_model_update_service()` resolves `{pipeline_id}_model_update_service`.
- `RequestScheduler.expand_workers(... selective_update=True, global_step=...)` calls:
  - `ModelUpdateService.selective_update(tgt_dp_ranks=load_ranks, requested_global_step=global_step)`
  - Then loads inference states on those ranks.

**Versioning rule**
- `ModelUpdateService.selective_update()` enforces:
  - `cached_global_step` must not be newer than `requested_global_step` (no “future weights” for a given rollout step).

## Code References
- `design_doc/archive/multi-pipeline_roll_old_design.md` — design vocabulary and intended protocol shape.
- `third_party/ROLL_multi_pipeline/roll/pipeline/agentic/multi_pipeline_orchestrator.py` — creates detached scheduler + registers pipelines.
- `third_party/ROLL_multi_pipeline/roll/distributed/scheduler/centralized_gpu_scheduler.py` — central scheduling loop, planning phases, completion-driven suspension.
- `third_party/ROLL_multi_pipeline/roll/distributed/scheduler/rollout_scheduler.py` — RolloutScheduler + GroupQueueManager progress emission; pipeline-scoped actor names.
- `third_party/ROLL_multi_pipeline/roll/distributed/scheduler/generate_scheduler.py` — RequestScheduler shrink/expand, routing, notify_ready_to_release.
- `third_party/ROLL_multi_pipeline/roll/distributed/executor/model_update_service.py` — per-pipeline cache + selective_update API.
- `third_party/ROLL_multi_pipeline/roll/pipeline/agentic/concurrent_agentic_pipeline.py` — pipeline coordinator; phase loop calls notify_ready_to_release; init registers ModelUpdateService.

## Related Research
- `thoughts/shared/research/2026-02-04-roll-megatron-multi-lora.md`

## Open Questions
- Which exact non-generation roles are enabled in the typical concurrent run (e.g., `reference`, `value_compute`, `old_log_probs`) depends on config; the time-sharing control plane supports multiple priorities via `Priority` in `gpu_scheduler_types.py`.

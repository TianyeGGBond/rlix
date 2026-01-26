# SchedRL Adaptation Review — Systematic Findings

Summary: I reviewed `design_doc/*` adaptation plans and code integration points. Key gaps remain across frameworks (NeMo-RL, ROLL, rLLM/VeRL, Miles) that block the multi-pipeline protocol as currently specified. This revision reflects the **direct scheduler calls + unidirectional proxy** model and the latest per-framework plans.

---

## High-level issues (applies to all frameworks)

- Missing central scheduler client package (`schedrl/`) in this repo; plans now assume direct scheduler calls from pipeline coordinators.
  - Severity: blocker for implementation unless scheduler is external.
  - Action: add a minimal scheduler client API (even if the scheduler actor is external).

- Heartbeat/progress hook is not implemented in framework loops. Plans require **batch-start + 2% band** reporting.
  - Severity: high.
  - Action: inject `scheduler.report_progress(remaining, percent_remaining, oldest_unfinished_creation_ts)` at the specified enqueue points.

- DP-granularity (subset-level) operations are required but many APIs are cluster-wide. Need `indices`/`subset` arguments consistently.
  - Severity: high (functional gap).
  - Action: add `*_subset(indices=...)` variants or `indices` parameters on lifecycle methods and selective sync APIs.

- Inconsistencies in docs: section numbering/narrative sometimes references "three frameworks" then later includes Miles as fourth; adapter numbering and file paths are inconsistent in places.
  - Severity: low (documentation hygiene).
  - Action: unify documents (I can fix doc inconsistencies if you want).

---

## Per-framework findings & recommended fixes

### ROLL
- Findings:
  - `ModelUpdateGroup.model_update` runs whole static comm plan; needs filtering by `worker_indices` for selective sync-on-resume.
  - `Cluster.start_server()` / `stop_server()` are cluster-wide only; docs propose `start_server_subset()` / `stop_server_subset()` but there is no implementation.
  - No `scheduler.report_progress` hooks in `GroupQueue.put`/`GroupQueueManager`.
- Severity: high.
- Recommended changes:
  1. Add `start_server_subset(worker_indices)` and `stop_server_subset(worker_indices)` to `Cluster` (vllm strategy), ensuring they only affect the requested indices.
  2. Add filtering to `ModelUpdateGroup.make_comm_plan` and `model_update(worker_indices)`.
  3. Inject progress reporting (`adapter.report_progress`) in `GroupQueue.put` when completed group is enqueued.
- Files touched (doc references): `roll/roll/distributed/executor/model_update_group.py`, `roll/roll/distributed/scheduler/rollout_scheduler.py`, `roll/roll/pipeline/agentic/agentic_pipeline.py`.

### NeMo-RL
- Findings:
  - Weight sync is handled via `refit_policy_generation()` and `VllmGeneration` hooks; need clear DP-rank selection support (dp_ranks or subset arguments).
  - `AsyncTrajectoryCollector` lacks a standard progress report insertion point for scheduler (hook must be placed after batch enqueue).
- Severity: high.
- Recommended changes:
  1. Extend `refit_policy_generation()` / `VllmGeneration` to accept `dp_ranks`/`indices` to do selective sync.
  2. Add progress hook in `AsyncTrajectoryCollector` after `trajectory_buffer.add(...)`.
- Files touched: `nemo-rl/nemo_rl/algorithms/grpo.py`, `nemo-rl/nemo_rl/algorithms/async_utils.py`, `nemo-rl/nemo_rl/distributed/worker_groups.py`.

### rLLM (VeRL)
- Findings:
  - `VerlEngine` and `RayWorkerGroup` need subset-targeting methods (`init_model(worker_indices=...)`, `offload_weights(worker_indices=...)`, `wake_up(indices)` or similar).
  - Preemption handling: stateful trajectories require a step-level retry strategy (docs mention Aborted exception) — current retry approach may restart entire episode.
  - Progress hooks missing in `AgentExecutionEngine` to report completed trajectories or task queue progress.
- Severity: high (correctness & availability of graceful preemption).
- Recommended changes:
  1. Add subset APIs in `verl` worker group code: `init_model(indices=...)`, `offload_weights(indices=...)`, `wake_up(indices)`, `sleep(indices)`.
  2. Implement step-level retry: add explicit `Aborted` exception handling in `VerlEngine.get_model_response()` and make `AgentExecutionEngine` retry only the failed step preserving trajectory state.
  3. Insert `adapter.report_progress(...)` in `AgentExecutionEngine` after task completion or buffer updates.
- Files touched: `rllm/rllm/engine/agent_execution_engine.py`, `rllm/rllm/engine/rollout/verl_engine.py`, `verl/verl/experimental/agent_loop/agent_loop.py`, `verl/verl/workers/rollout/replica.py`.

### Miles
- Findings:
  - `RolloutManager` has onload/offload methods but they lack `indices` filtering needed for DP-granularity.
  - Docs assume `onload_weights()` exists (it does in Miles) and must move to sync-on-resume semantics.
  - `rollout_manager.generate()` does not yet call scheduler progress hook; needs batch-start + 2% band reporting.
- Severity: medium-high.
- Recommended changes:
  1. Add `onload(indices=...)` and `offload(indices=...)` to `RolloutManager` and ensure remote calls accept `indices`.
  2. Add `adapter.report_progress()` calls in `rollout_manager.generate()` at micro-batch completion points (batch-start and when percent remaining crosses 2% bands).
- Files touched: `miles/miles/ray/rollout.py`, `miles/miles/ray/actor_group.py`, `miles/train.py`.

---

## Cross-cutting correctness & safety points

- Tail-end efficiency: Scheduler must not expand worker sets if remaining work is tiny (e.g., <5% or estimated <10s). This is documented but must be enforced in `policies.py` in `schedrl`.
- Hysteresis/min lease time: Scheduler needs minimum lease time (e.g., 60s) to avoid thrashing; add to policy config.
- Weight versioning: Ensure each framework exposes `global_step`/`weight_version` metadata to avoid stale resume; docs reference disparate fields — unify naming (`global_step` preferred).

---

## Documentation inconsistencies observed
- Sometimes documents state "three frameworks" but include Miles later — inconsistent counts.
- Adapter numbering is inconsistent (e.g., `2.5 rLLM Adapter` vs `2.3 rLLM Adapter`) — minor but confusing.
- The docs mention `schedrl/client/adapters/*` but that package doesn't exist in the tree.
- Some TODO annotations exist in `.json` doc artifacts (e.g., "Need to implement init_model(worker_indices=...) in VeRL"). These should be turned into issues or tracked tasks.

---

## Suggested immediate next steps (short-term backlog)
1. Add a minimal scheduler client API or stub (even if the scheduler actor lives elsewhere).
2. Implement subset APIs + selective sync for each framework (per above).
3. Add progress/heartbeat hooks at the identified enqueue points.
4. Add tests or harnesses to validate subset operations and progress reporting cadence.

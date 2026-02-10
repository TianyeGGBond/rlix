# Consolidated SchedRL Plans Review — 2026-01-29

Summary

This review consolidates and prioritizes issues found across the SchedRL adaptation plans in `thoughts/shared/plans`. It reflects annotations of the core dual-mode plan and per-framework adaptation plans for ROLL, NeMo-RL, Miles, and SkyRL-train. The deliverable below lists Critical, High, and Medium issues, concrete fixes, and exact filenames to edit for the first implementation wave.

Source plans referenced

- [`thoughts/shared/plans/2026-01-28-schedrl-dual-mode-final-plan.md`](thoughts/shared/plans/2026-01-28-schedrl-dual-mode-final-plan.md:1)
- [`thoughts/shared/plans/2026-01-28-roll-schedrl-adaptation.md`](thoughts/shared/plans/2026-01-28-roll-schedrl-adaptation.md:1)
- [`thoughts/shared/plans/2026-01-28-nemo-rl-schedrl-adaptation.md`](thoughts/shared/plans/2026-01-28-nemo-rl-schedrl-adaptation.md:1)
- [`thoughts/shared/plans/2026-01-28-miles-schedrl-adaptation.md`](thoughts/shared/plans/2026-01-28-miles-schedrl-adaptation.md:1)
- [`thoughts/shared/plans/2026-01-28-skyrl-train-adaptation-plan.md`](thoughts/shared/plans/2026-01-28-skyrl-train-adaptation-plan.md:1)

Critical Issues (blockers — must resolve before integration)

1) GPU memory release semantics on shrink
   - Problem: Several frameworks expose "sleep" or "pause" but may not fully free GPU memory. The Final Plan requires full GPU release on `shrink_workers(...)`.
   - Files to inspect/test:
     - [`third_party/ROLL/roll/pipeline/agentic/agentic_pipeline.py`](third_party/ROLL/roll/pipeline/agentic/agentic_pipeline.py:138)
     - [`third_party/nemo-rl/nemo_rl/models/generation/vllm/vllm_generation.py`](third_party/nemo-rl/nemo_rl/models/generation/vllm/vllm_generation.py:691)
     - [`third_party/miles/miles/ray/rollout.py`](third_party/miles/miles/ray/rollout.py:127)
   - Suggested fix: define and implement an explicit GPU-free offload path (weights+KV eviction or actor restart). Add a small verification task that runs a worker, calls shrink, then launches a new allocation that uses the freed GPU.

2) Runtime environment / sitecustomize shim reliability
   - Problem: NeMo-RL / SkyRL / Miles adaptation plans rely on `sitecustomize.py` and `PYTHONPATH` propagation into Ray `runtime_env`. This is environment dependent and fragile.
   - Files to inspect:
     - [`thoughts/shared/plans/2026-01-28-schedrl-dual-mode-final-plan.md`](thoughts/shared/plans/2026-01-28-schedrl-dual-mode-final-plan.md:176)
   - Suggested fix: add a runtime_env verification RPC/actor task (marker env var or return value) early in each integration PR.

3) Precise abort ACK definition and enforcement
   - Problem: All frameworks must agree on what constitutes an ACK (e.g., `finish_reason == "abort"`, in-flight-rid=0). Any mismatch may lead to premature offload or deadlocks.
   - Files to align:
     - [`third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py`](third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py:159)
     - [`third_party/nemo-rl/nemo_rl/models/generation/vllm/vllm_worker_async.py`](third_party/nemo-rl/nemo_rl/models/generation/vllm/vllm_worker_async.py:747)
     - [`third_party/miles/miles/router/router.py`](third_party/miles/miles/router/router.py:67)
   - Suggested fix: adopt canonical ACK wording in the `schedrl/protocol/validation.py` (see below) and implement small adapter-level checks that explicitly return `error="Superseded"` when appropriate.

High Priority (important, non-blocking)

1) Deterministic request/request_id format and propagation
   - Canonical: `request_id = "{trajectory_id}:{turn_id}:{attempt}"` (attempt increments only for engine-error retries).
   - Files to change (examples):
     - [`third_party/ROLL/roll/pipeline/agentic/env_manager/traj_env_manager.py`](third_party/ROLL/roll/pipeline/agentic/env_manager/traj_env_manager.py:129)
     - [`third_party/nemo-rl/nemo_rl/experience/rollouts.py`](third_party/nemo-rl/nemo_rl/experience/rollouts.py:175)
     - [`third_party/miles/miles/rollout/sglang_rollout.py`](third_party/miles/miles/rollout/sglang_rollout.py:208)
   - Suggested fix: PRs that plumb request_id through the call chain and log it at the worker boundary.

2) Discovery semantics (create_if_missing) and actor naming
   - Ensure client `connect()` implements get-then-create with create-race handling. Sanity-check across Library vs Service Mode. Reference: [`thoughts/shared/plans/2026-01-28-schedrl-dual-mode-final-plan.md`](thoughts/shared/plans/2026-01-28-schedrl-dual-mode-final-plan.md:55)

Medium Priority

1) Progress heartbeat mapping and 2% bands
   - Implement `report_progress(...)` in each adapter. Files:
     - [`third_party/ROLL/roll/distributed/scheduler/rollout_scheduler.py`](third_party/ROLL/roll/distributed/scheduler/rollout_scheduler.py:216)
     - [`third_party/miles/miles/rollout/sglang_rollout.py`](third_party/miles/miles/rollout/sglang_rollout.py:306)
     - [`third_party/nemo-rl/nemo_rl/algorithms/async_utils.py`](third_party/nemo-rl/nemo_rl/algorithms/async_utils.py:238)

2) Subset-scoped sync-on-resume implementation specifics
   - Implement CPU weight cache + subset sync per ROLL plan: [`third_party/ROLL/roll/distributed/executor/model_update_group.py`](third_party/ROLL/roll/distributed/executor/model_update_group.py:30)

Canonical wording to add to `schedrl/protocol/validation.py` (proposed)

- ActionResponse schema: `{ "success": bool, "error": Optional[str] }` — use `"Superseded"` for superseded intents.
- ACK definition: explicit per-adapter rule, but scheduler-level fallback: "ACK = worker reports no in-flight requests for the targeted request ids or returns `finish_reason==\"abort\"` for those ids".
- create_if_missing: default `False` in Service Mode; `True` allowed in Library Mode. Client must catch create-race and re-get actor.

Suggested first-wave PR checklist (one PR per area)

1) Core package stub: add `schedrl/protocol/{types,actions,validation}.py` and `schedrl/client/{adapter,client}.py` with minimal interfaces. (Reference: [`thoughts/shared/plans/2026-01-28-schedrl-dual-mode-final-plan.md`](thoughts/shared/plans/2026-01-28-schedrl-dual-mode-final-plan.md:81))
2) ROLL PR: deterministic request_id + targeted abort + subset lifecycle (files listed in ROLL plan). Example entry: [`third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py`](third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py:866)
3) NeMo-RL PR: vLLM subset wake/sleep + abort_requests + request_id plumbing (files in NeMo-RL plan). See: [`third_party/nemo-rl/nemo_rl/models/generation/vllm/vllm_worker_async.py`](third_party/nemo-rl/nemo_rl/models/generation/vllm/vllm_worker_async.py:192)
4) Miles PR: MilesRouter admission endpoints + inflight rid accounting + RolloutManager subset lifecycle. See: [`third_party/miles/miles/router/router.py`](third_party/miles/miles/router/router.py:113)
5) SkyRL PR: sitecustomize shim that patches request_id plumbing and abort-by-request on vLLM. See shim targets above.

Next steps I will take after you approve this review

- Create small canonical `schedrl/protocol/validation.py` text and a minimal `schedrl/client/adapter.py` interface stub (draft PR content).
- Optionally, open branch/PR stubs for each framework with the change list above.

If you want line-level comments added back into the source plan files, tell me and I will attach them as inline annotations next.

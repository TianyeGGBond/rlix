# GPU-shrink Verification Plan — 2026-01-29

Summary

This document captures a concrete verification checklist and precise patch diffs to ensure the MUST-FIX (shrink_workers(...) must free GPU memory) is validated and remediated across the three targets identified in the consolidated review.

Targets

- [`third_party/ROLL/roll/pipeline/agentic/agentic_pipeline.py:138`](third_party/ROLL/roll/pipeline/agentic/agentic_pipeline.py:138)
- [`third_party/nemo-rl/nemo_rl/models/generation/vllm/vllm_generation.py:691`](third_party/nemo-rl/nemo_rl/models/generation/vllm/vllm_generation.py:691)
- [`third_party/miles/miles/ray/rollout.py:127`](third_party/miles/miles/ray/rollout.py:127)

Verification checklist (exact steps)

1) Add lightweight instrumentation endpoints / actor methods to report GPU allocation state
   - Add a worker-level RPC: `get_gpu_allocation_report() -> { free_bytes: int, reserved_bytes: int, handles: List[str] }`.
   - Add a cluster/manager RPC that aggregates reports from targeted worker indices.
   - Files to add/modify: add stubs in each adapter module and expose via worker_group.run_all_workers_single_data or equivalent.

2) Baseline measurement
   - Start a single worker on a specific GPU (GPU X). Allocate model + KV.
   - Call `get_gpu_allocation_report()` and record `free_bytes_before`.

3) Call shrink path
   - Invoke the framework's shrink / sleep / offload API that the adapter will call (e.g., `shrink_workers(...)`, `offload_states()`, `sleep()`), matching the plan's adapter call semantics.
   - After shrink returns, immediately call the cluster/manager RPC to collect `free_bytes_after_shrink`.
   - Expected: `free_bytes_after_shrink` > `free_bytes_before` by a significant margin (model weights + KV freed). If not, mark as FAIL for that framework and collect diagnostics.

4) Fallback path test
   - If shrink did not free memory, call the stronger remediation path: actor-level `release_memory_occupation()` RPC (if available) or full actor restart (call actor.exit/terminate then re-create worker).
   - After remediation, collect `free_bytes_after_remediation`. Expected: memory freed.

5) Reallocation test
   - Immediately start a fresh small process (or Ray actor) that allocates a small model or tensor on GPU X. This should succeed without OOM.
   - If OOM occurs, record the exact error and the sequence of RPCs that were executed.

6) Record exact failure modes
   - If memory not freed, capture: exception traces, `nvidia-smi` output (or torch.cuda.mem_get_info values), and worker logs.

Per-target suggested patch diffs

Note: these diffs are proposals for minimal, safe changes. They should be reviewed and adapted to exact API names and types in each framework. Apply as small, self-contained PRs with unit tests described below.

A) ROLL — ensure offload path frees GPU memory

Problem area: [`third_party/ROLL/roll/pipeline/agentic/agentic_pipeline.py:138`](third_party/ROLL/roll/pipeline/agentic/agentic_pipeline.py:138)

Proposed change (high-level): after calling `offload_states()` ensure a `release_memory_occupation()`-equivalent RPC is invoked on the actor cluster. If the actor cluster does not provide a `release_memory_occupation` RPC, add it to the `Cluster` abstraction and implement in the worker to perform cuda free + optional process exit/restart.

Suggested patch (conceptual unified-diff)

--- a/third_party/ROLL/roll/pipeline/agentic/agentic_pipeline.py
+++ b/third_party/ROLL/roll/pipeline/agentic/agentic_pipeline.py
@@
-                    if self.pipeline_config.adv_estimator == "gae":
-                        self.critic.offload_states(blocking=True)
-                    self.actor_train.offload_states(blocking=True)
+                    if self.pipeline_config.adv_estimator == "gae":
+                        self.critic.offload_states(blocking=True)
+                    # Offload and then explicitly request GPU memory release from actor cluster
+                    self.actor_train.offload_states(blocking=True)
+                    # Ensure memory is freed: call explicit release RPC and wait for confirmation
+                    try:
+                        # Cluster should implement release_memory_occupation to free weights/KV/allocations
+                        ray.get(self.actor_train.release_memory_occupation.remote())
+                    except Exception:
+                        # Best-effort fallback: restart actor group members
+                        self.actor_train.restart_workers(blocking=True)
@@
-                        self.actor_infer.start_server(data=DataProto(meta_info={"global_step": global_step, "is_offload_states": False}))
+                        self.actor_infer.start_server(data=DataProto(meta_info={"global_step": global_step, "is_offload_states": False}))

Notes:
- If `Cluster` does not have `release_memory_occupation`/`restart_workers` methods, add them to `roll/distributed/executor/cluster.py` and implement worker-side handlers that call `torch.cuda.empty_cache()`, clear KV caches, and optionally exit the process when `force_restart=True`.
- Add unit test that calls the above path and asserts `torch.cuda.mem_get_info()` increases after release.

B) NeMo-RL / vLLM — ensure sleep path truly frees GPU when requested

Problem area: [`third_party/nemo-rl/nemo_rl/models/generation/vllm/vllm_generation.py:691`](third_party/nemo-rl/nemo_rl/models/generation/vllm/vllm_generation.py:691)

Context: vLLM exposes `sleep()`/`wake_up()` semantics. Some existing tests call `allocator.sleep()` and `wake_up(tags=["weights"])`. The goal is to make `sleep(level=2)` or a new `release_memory()` call actually perform weight/KV eviction and free GPU memory.

Suggested patch (conceptual unified-diff):

--- a/third_party/nemo-rl/nemo_rl/models/generation/vllm/vllm_generation.py
+++ b/third_party/nemo-rl/nemo_rl/models/generation/vllm/vllm_generation.py
@@
     def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
         """Sleep workers and reset prefix cache."""
         try:
@@
-            futures = self.worker_group.run_all_workers_single_data(
-                method_name,
-                run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
-            )
-            results = ray.get(futures)
-            return all(result for result in results if result is not None)
+            futures = self.worker_group.run_all_workers_single_data(
+                method_name,
+                run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
+            )
+            results = ray.get(futures)
+            ok = all(result for result in results if result is not None)
+            if ok:
+                # Request an explicit memory release on vLLM workers to drop weights/KV
+                try:
+                    release_futures = self.worker_group.run_all_workers_single_data(
+                        "release_memory_occupation",
+                        run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
+                    )
+                    ray.get(release_futures)
+                except Exception:
+                    # fallback to worker restart
+                    self.worker_group.run_all_workers_single_data("force_restart", run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"])
+            return ok
@@
     def invalidate_kv_cache(self) -> bool:
@@
             results = ray.get(futures)
             return all(result for result in results if result is not None)
+        except Exception as e:
+            print(f"Error invalidating vLLM caches: {e}")
+            return False
+
+    # Worker-facing RPC expected to be implemented in vllm worker class:
+    # def release_memory_occupation(self) -> bool: perform allocator.sleep(offload_tags=(weights,)) + torch.cuda.empty_cache() and return True

Notes:
- vLLM already has `sleep()` and `reset_prefix_cache` calls. Add `release_memory_occupation` in the worker implementation (look in `third_party/vllm/vllm/v1/worker/gpu_worker.py`) to call allocator.sleep(...) and `torch.cuda.empty_cache()` and verify freed bytes.
- Add tests using `create_new_process_for_each_test()` to measure memory freed after calling `finish_generation()`.

C) Miles — ensure release path is reliable and provide forced-restart fallback

Problem area: [`third_party/miles/miles/ray/rollout.py:127`](third_party/miles/miles/ray/rollout.py:127)

Observation: RolloutManager.offload already calls `engine.release_memory_occupation.remote()` (line ~176). Confirm that `release_memory_occupation()` on `SGLangEngine` performs full eviction and, if not, add a stronger `force_restart()` path.

Suggested patch (conceptual unified-diff):

--- a/third_party/miles/miles/ray/rollout.py
+++ b/third_party/miles/miles/ray/rollout.py
@@
     def offload(self):
         self.health_monitoring_pause()
-        return ray.get(
-            [engine.release_memory_occupation.remote() for engine in self.rollout_engines if engine is not None]
-        )
+        results = ray.get([engine.release_memory_occupation.remote() for engine in self.rollout_engines if engine is not None])
+        # If any engine reported failure to free memory, perform a force restart on that engine
+        for i, engine in enumerate(self.rollout_engines):
+            if engine is None:
+                continue
+            if not results[i]:
+                try:
+                    ray.get(engine.force_restart.remote())
+                except Exception:
+                    # last resort: set engine None and let init_rollout_engines recreate it
+                    self.all_rollout_engines[i] = None
+        return results

Notes:
- Ensure the SGLangEngine exposes `release_memory_occupation()` returning True/False and `force_restart()` performing a quick actor restart.
- Add a small verification script to call `offload()` and then attempt immediate reallocation (torch allocation) on the same GPU.

Unit / integration test templates (what to run locally)

- Test script `tests/integration/test_gpu_shrink_<framework>.py` (pytest-compatible)
  1. Launch a minimal Ray cluster (or use local single-node Ray) with 1 GPU.
  2. Start the relevant manager/cluster that creates one worker pinned to GPU 0.
  3. Allocate a small model on the worker (or have the worker load model). Record `torch.cuda.mem_get_info()` on that worker and on host.
  4. Invoke the shrink/offload/sleep path through the manager adapter.
  5. Immediately measure `torch.cuda.mem_get_info()` again.
  6. If memory not freed, invoke `release_memory_occupation()` then re-measure.
  7. Assert reallocation success: start a new tiny actor requiring GPU memory and import torch and allocate small tensor without OOM.

Expected outputs and pass/fail criteria

- PASS: After shrink + release, free GPU memory increases and the reallocation actor can allocate on the freed GPU.
- FAIL: Memory unchanged or reallocation actor OOMs. Record logs and mark the framework as failing.

Deliverables I will produce next (if you approve)

- Exact PR diffs (actual code edits) for the Cluster/worker APIs referenced above in small, reviewable commits.
- One integration test per framework under `third_party/<framework>/tests/integration/test_gpu_shrink.py` that can be run locally.
- A short pass/fail matrix recorded in `thoughts/shared/reviews/2026-01-29-gpu-shrink-results.md` after running tests.

Status

- This verification plan has been written and saved as [`thoughts/shared/reviews/2026-01-29-gpu-shrink-verification-plan.md`](thoughts/shared/reviews/2026-01-29-gpu-shrink-verification-plan.md:1). If you want, I will now draft the actual code patches for each framework (one PR per framework) or instead produce the integration test scripts first.

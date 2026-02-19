# SchedRL Multi‑LoRA Adapter Extension (ROLL SchedRL Integration) Implementation Plan

**Date**: 2026-02-02 (Updated: 2026-02-18)

## Overview

Port the multi-LoRA feature from `external/ROLL_multi_lora` into `external/ROLL_schedrl` to enable SchedRL-controlled multi-LoRA training. The goal is to make the multi-LoRA pipeline coordinator integrate with the SchedRL scheduler the same way the current concurrent pipeline does.

### Current State
- **external/ROLL_schedrl**: Has `SchedRLConcurrentPipeline` with full SchedRL integration (shrink/expand, `notify_ready_to_release`, selective sync via `ModelUpdateService`)
- **external/ROLL_multi_lora**: Has `AgenticMultiLoraPipeline` with multi-LoRA support (per-tag schedulers, `model_update_lora_subset`, partial GPU mode)
- **Gap**: ROLL_schedrl only supports single LoRA (via `actor_lora_target` check); multi-LoRA patterns exist in ROLL_multi_lora but aren't integrated with SchedRL

### Goal
Create `SchedRLMultiLoraPipeline` in `external/ROLL_schedrl` that:
1. Reuses existing multi-LoRA patterns from ROLL_multi_lora
2. Integrates with SchedRL scheduler like `SchedRLConcurrentPipeline` does
3. Supports per-adapter progress tracking and reporting
4. Handles shrink/expand with adapter-aware routing

---

## OLD PLAN CONTENT (for reference)

Extend the shared multi-pipeline protocol (`design_doc/multi-pipeline-adaptation-plan.md`) so **each RL pipeline** can run in either:

1) **Full fine-tune mode** (single evolving base checkpoint, current protocol), or
2) **Multi‑LoRA mode** where the **base model is fixed** and the pipeline trains **multiple LoRA adapters concurrently**, and rollout supports **S-LoRA-style mixed-adapter batching**: a single inference batch may include prompts targeting different adapters, with adapter selection done per request.

### ROLL-first adapter identity (canonical `adapter_id`)
This extension is ROLL-first: ROLL already has per-domain/per-env labels (`tag` in agentic envs and `domain` in async sampling).

Standardize on a single canonical protocol field name:
- `adapter_id` is the **only** protocol-level key for “which LoRA adapter to apply”.
- ROLL maps `adapter_id := env_config["tag"]` (agentic) and `adapter_id := domain` (async) at the coordinator boundary, and treats `adapter_id` as the source of truth thereafter (request IDs, caching, progress metrics, optimizer state).

Core requirement: LoRA weights are **trained + synchronized at adapter granularity**, while shrink/expand time-sharing remains safe and uses SchedRL’s existing primitives (admission control, abort+ACK, offload, expand, selective sync).

This plan focuses on a **vLLM-first** shape because:
- ROLL already plumbs `lora_request` into vLLM generation (`third_party/ROLL/roll/distributed/strategy/vllm_strategy.py:172`).
- SkyRL-train already has LoRA disk-load hooks using vLLM `add_lora` (`third_party/SkyRL/skyrl-train/skyrl_train/inference_engines/vllm/vllm_engine.py:313`).
NeMo-RL wiring is deferred/archived for now; this plan focuses on ROLL + SkyRL-train.

---

## Current State Analysis

### What the protocol assumes today (single checkpoint axis)
The current protocol models “weights” as a single monotonic `checkpoint_version` chosen by the coordinator, with:
- `active_checkpoint_version` as the rollout target version.
- a trainer-side CPU checkpoint cache service (“bucket list”) as the source of truth for expand/resume.
- shrink/expand orchestration that assumes a single weight version to sync/activate.

This is sufficient for full fine-tune, but insufficient for **multi-LoRA**, because:
- multiple adapters can update at different times (multi-dimensional versioning),
- rollout needs to select **adapter identity** per request/batch,
- expand-from-zero needs to ensure “base + required adapters” are available before opening admission.

### Reference implementation hooks already exist
- **ROLL**: vLLM strategy builds `LoRARequest` and passes `lora_request=...` into generation (`third_party/ROLL/roll/distributed/strategy/vllm_strategy.py:172`, `third_party/ROLL/roll/distributed/strategy/vllm_strategy.py:326`).
- **SkyRL-train**: vLLM engine loads LoRA from disk via `add_lora` (`third_party/SkyRL/skyrl-train/skyrl_train/inference_engines/vllm/vllm_engine.py:313`) and uses `sleep(level=1)` when LoRA is enabled (`third_party/SkyRL/skyrl-train/skyrl_train/inference_engines/vllm/vllm_engine.py:298`).
  - **SchedRL note**: for time-sharing shrink, SchedRL still requires **full GPU release** (weights+KV), so shrink must use deep sleep (`level=2`) even in LoRA mode; “level=1” can remain valid only for internal non-time-sharing pauses.
  - **Mixed-adapter batching note**: vLLM supports passing a per-prompt `lora_request` list (one `LoRARequest` per prompt). This is the mechanism used for S-LoRA-style mixed-adapter batches.
  - **Adapter update note (embedded API)**: in our current embedded integration surfaces (ROLL/SkyRL), we have `add_lora(...)` and `list_loras()`, but no explicit `remove_lora(...)`/`reload_lora(...)` surfaced. Therefore, we should not assume we can safely overwrite/replace adapter X “in place” while other requests are executing unless we validate it in the exact vLLM build used by the framework.

### ROLL reference: `tag` / domain concept maps naturally to adapter identity
ROLL already carries a per-environment/per-sample “domain” concept that is close to “adapter selection”:
- Agentic env managers use `env_config["tag"]` to select templates and per-tag settings (`third_party/ROLL/roll/pipeline/agentic/env_manager/traj_env_manager.py:76`), and also use the tag for rate limiting (`third_party/ROLL/roll/pipeline/agentic/env_manager/traj_env_manager.py:64`).
- Async generation scheduler tracks a per-item `domain` and produces per-domain metrics (`third_party/ROLL/roll/distributed/scheduler/async_generate_scheduler.py:454`, `third_party/ROLL/roll/distributed/scheduler/async_generate_scheduler.py:547`).

**Recommendation (ROLL-first)**:
- Standardize on a single canonical protocol field name: `adapter_id`.
- Define `adapter_id := env_config["tag"]` (agentic) or `adapter_id := domain` (async_generate_scheduler) in `MULTI_LORA`.
- Treat `tag`/`domain` as **source fields** that are mapped/aliased to `adapter_id` at the coordinator boundary; `adapter_id` is the source of truth thereafter.
- Treat “tag/domain” as the *routing key* for both:
  - which LoRA adapter to apply at inference time, and
  - which adapter’s optimizer state/version to update at training time.

This keeps the mental model consistent: “tag/domain” becomes the stable adapter identity across rollout, caching, progress reporting, and training updates.

---

## Desired End State

### Protocol supports both modes without forking the scheduler
The scheduler continues to reason at **pipeline granularity** (one engine group per pipeline; isolation assumption unchanged), but the protocol gains a first-class notion of:
- **Base model** artifact (full weights, slow to move),
- **Adapter** artifacts (small, frequent updates),
- A combined **Active Model Spec** that defines what rollouts should use.

### Correctness and safety invariants (carried over)
For shrink/expand and migration:
- Shrink ordering remains: **Close Admission → Abort(P) → Wait ACK → Offload/Stop(P) → Release GPUs**.
- If abort ACK does not arrive by timeout, **fail fast** (crash pipeline) as in the existing protocol.
- No “resume partial turn” across shrink; shrink uses `REQUEST_RETRY` (abort + re-issue).

### Multi-LoRA-specific requirements
- **One adapter per trajectory**, and rollout may be **mixed-adapter batched** (a batch can contain multiple adapters; each request carries its own `adapter_id`).
- Adapter synchronization happens at **adapter granularity** (update adapter X without requiring a full base sync).
- Expand-from-zero can warm only the adapters that have queued work (avoid preloading all adapters).

---

## Key Design Decisions (Options + Recommendations)

This section “answers the open questions” for multi-LoRA by choosing defaults that are compatible with ROLL/SkyRL/NeMo-RL and keep scheduler complexity low.

### Decision A — Base weights behavior in LoRA mode
**Option A1 (recommended)**: Base model **frozen** for the entire run; only adapters update.
- Reasons: matches common LoRA semantics (NeMo-RL docs), makes cache + versioning tractable, avoids “two axes” (base+adapter) changing concurrently.
- Confirmed requirement for this plan: in `MULTI_LORA`, the pipeline uses a **single shared frozen base** (never updated) and trains multiple adapters concurrently (shared trainer; per-adapter optimizer state).

**Option A2**: Base also updates (full FT + adapters simultaneously).
- Reasons to avoid initially: scheduler would need to coordinate base sync boundaries while adapters are also updating; the “active spec” becomes a true multi-dimensional vector clock and increases failure modes.

### Decision B — Adapter versioning model
**Option B1 (recommended)**: Each adapter has its own monotonic version, `adapter_version[adapter_id]`.
- Base version is fixed in LoRA mode (or changes rarely in future).
- Reasons: enables independent adapter updates and small artifact caching.

**Option B2**: Single global step for all adapters.
- Reasons to avoid: forces lock-step updates and wastes work if adapters progress at different rates.

### Decision C — When does a new adapter version become active for rollouts?
Because rollouts may be mixed-adapter batched, cutovers must be **scoped to the adapter being updated** (not a global stop-the-world boundary).

**Option C1 (recommended default for correctness)**: adapter-scoped `QUIESCE-by-abort` (ROLL-aligned).
- Close admission for adapter X (do not schedule X into mixed batches), abort in-flight requests for adapter X, wait abort ACK, then activate adapter X@v and reopen X admission.
- Other adapters may continue generating (and continue to appear in mixed batches) while X is paused, **if** the embedded inference API is safe to mutate adapter state without a global stop (see fallback below; default is a brief global `QUIESCE-by-abort` during `add_lora(...)` if unvalidated).
- Reasons: aligns with ROLL’s default safety boundary (`QUIESCE-by-abort`) and avoids waiting for natural completion.

**Option C2**: `INFLIGHT` for adapters (finish old trajectories on old adapter; new ones use new adapter).
- Requires tagging samples with `(adapter_id, adapter_version)` (recommended anyway) and accepting mixed-version data.
- Reasons to choose: better throughput if aborting is expensive or too disruptive.

**Option C3**: multi-version residency for the same adapter (keep X@v_old and X@v_new both loaded) and select `(adapter_id, adapter_version)` per request.
- Only valid if the inference engine supports it (or if `adapter_id` is versioned, e.g., `adapter_id = f"{name}@{version}"` mapping to distinct loaded LoRA handles).
- Reasons to choose: avoids pausing adapter X during updates, at the cost of higher memory pressure and more complex GC.

**Implementation fallback rule (embedded API, recommended)**:
- Default to **C1** at the coordinator level (stop scheduling adapter X; abort X in-flight; wait abort ACK).
- The actual “activate X@v_new” step may still require a **brief global control critical section** on each engine (because `add_lora(...)` mutates shared engine state). If the framework’s embedded API is not proven safe to call concurrently with generation, fall back to a short global `QUIESCE-by-abort` of the whole engine group for the duration of the `add_lora(...)` call, then resume mixed-adapter generation immediately.
  - Any requests aborted solely due to this brief global quiesce (including “bystander” adapters not being updated) MUST be treated as **Preemption Retries**, not **Engine Errors** (i.e., they must not count against any “max engine errors” cap).
- This keeps correctness while allowing us to later optimize toward “pure adapter-scoped update” if/when validated.

### Decision D — Retry semantics after shrink-triggered abort
**Option D1 (recommended)**: Abort+retry produces a fresh completion and is attributed to whatever `(adapter_id, active_adapter_version)` is active at retry time.
- Reasons: avoids having to pin old adapter versions just to satisfy retries; aligns with “no mid-turn resume”.

**Option D2**: Strict snapshot retry (retry must use the exact adapter version snapshot).
- Requires pinning old adapter versions until all in-flight/retry windows close; more cache/GC complexity.

---

## Protocol Extensions (What Must Change)

### 1) New concepts in the shared protocol
Add the following protocol-level objects (names illustrative; final naming should match `schedrl/protocol/types.py` once implemented):

- `ModelMode = {FULL_FT, MULTI_LORA}`
- `AdapterId` (stable identifier; recommended string)
- `ActiveModelSpec = {base_version: int, adapters: dict[AdapterId, int]}`
  - `base_version`:
    - in `FULL_FT`: the usual checkpoint version (same meaning as `active_checkpoint_version` / `active_base_version`),
    - in `MULTI_LORA`: `-1` (sentinel) meaning “frozen base for the run” (constant; the base is not updated during adapter training, and its artifact is resolved from static config / cache, not by version lookup).
    - Ordering note: in `MULTI_LORA`, `base_version` is an identifier/sentinel; it must not be used in “newer wins” comparisons (only equality + validation is meaningful).
  - `adapters`: map `adapter_id -> adapter_version` (multi-dimensional “active state”).
  - URIs are resolved by the trainer-side artifact cache / static config, not passed through the scheduler protocol.
  - `ModelMode` is a **registration-time constant per pipeline** (scheduler stores it from `register()`); it is not carried in the active model state/messages.
  - **Validation rule**: `base_version == -1` is only permitted when `model_mode == MULTI_LORA`; in `FULL_FT`, `base_version MUST be >= 0`.

**Compatibility rule**:
- In `FULL_FT`, `ActiveModelSpec.adapters = {}` and the existing single-axis `checkpoint_version` semantics remain.

### 2) Extend the cache contract (“checkpoint cache” → “artifact cache”)
Generalize the trainer-side cache service to manage **artifacts**, not just full checkpoints:
- Base weights cache: same as today (CPU bucket list / staged snapshots).
- Adapter cache: per `(adapter_id, adapter_version)` artifacts (likely file paths or in-memory blobs, depending on framework).

Base-frozen implication (important):
- In `MULTI_LORA`, do **not** repeatedly “sync/update the base” on every adapter update. The base artifact is immutable for the run.
- For time-sharing shrink, rollout workers must fully release GPU memory; this implies the **base weights and all adapters are dropped** on the shrunk subset (and possibly the whole generation cluster if shrinking to zero). On expand/resume, the base is re-loaded from the trainer cache, and adapters are re-loaded as needed.

GC rules (Phase 1, recommended):
- Keep: current active adapter versions.
- Keep: newest `K` versions per adapter (configurable; default small like 2–4).
- Do **not** guarantee strict snapshot retries (per Decision D1), so old versions can GC aggressively.

### 3) Coordinator-driven warmup on expand/resume (scheduler remains workload-agnostic)
Expand-from-zero must avoid opening admission before adapters needed for queued work are available, but the scheduler should not compute adapter-level warmup lists.

Protocol requirement:
- `expand_workers(worker_indices, base_version, action_id, activation_epoch)`
  - Coordinator loads base (per `base_version`) and then warms adapters based on its own local per-adapter queues (e.g., any `adapter_id` with `queued_trajectories[adapter_id] > 0`) before it allows mixed-batch dispatch to those workers.

Coordinator state requirement (MULTI_LORA):
- Track `resident_adapters_by_worker[worker_index] -> dict[adapter_id, adapter_version]` (or equivalent).
- Dispatch MUST only target workers where the requested `adapter_id` is resident at the desired version (or the coordinator must load it first under admission gating).

### 4) Request identity must include adapter identity (for abort + attribution)
Deterministic request IDs should incorporate adapter identity so we can:
- debug mixed adapter workloads,
- target aborts correctly,
- tag samples/trajectories with model spec.

Recommended convention (string):
- `request_id = f\"{trajectory_id}:{turn_id}:{attempt}:{adapter_id}\"`

ROLL mapping (recommended):
- use `adapter_id = env_config["tag"]` (agentic) or `domain` (async_generate_scheduler) and include it in the request id.

### 5) Version tagging for produced data
Tag each completed trajectory with:
- `base_checkpoint_version` (or model hash),
- `adapter_id`,
- `adapter_version`.

This keeps the training side honest under Options C2/D1 (mixed/in-flight updates, retry on latest).

### 6) Progress reporting in multi-LoRA mode (aggregation + per-adapter percent)
SchedRL’s `report_progress(...)` has a single `percent_completed` scalar, but multi-LoRA naturally has “per-adapter readiness”.

**Option 1 (recommended for mixed-adapter batching)**: aggregate queued/inflight, and report per-adapter completion percent via `metrics`.
- Aggregation (required fields):
  - `queued_trajectories = sum_a queued_trajectories[a]`
  - `inflight_trajectories = sum_a inflight_trajectories[a]`
  - `oldest_unfinished_creation_ts = min_a oldest_unfinished_creation_ts[a]` over all unfinished work
- Pipeline-level `percent_completed` (scalar required by the protocol; recommended definition):
  - define `target_trajectories[a]` for the next readiness window (configuration; could be uniform across adapters)
  - define `collected_trajectories[a]` as “complete and ready-to-train for adapter a”
  - Validation (fail fast): require `sum_a target_trajectories[a] > 0` for every readiness window; otherwise crash the pipeline with a clear error (invalid configuration / empty adapter set).
  - `percent_completed = min(1.0, sum_a collected_trajectories[a] / sum_a target_trajectories[a])`
- Per-adapter (extra metrics):
  - `metrics["percent_completed_by_adapter"] = {adapter_id: pct}`
  - (optional) `metrics["queued_by_adapter"]`, `metrics["inflight_by_adapter"]`

ROLL mapping (recommended):
- Use the existing “domain/tag” label as `adapter_id` for the per-adapter metrics so the same key appears in:
  - rollout routing,
  - reward/quality reporting (already emitted as `scheduler/{domain}/...` today),
  - multi-LoRA progress readiness.

**Option 2**: single-target adapter progress (`target_adapter_id`) drives the scalar `percent_completed`.
- Good fit for “one adapter-at-a-time” collection/training, but ambiguous if the pipeline is collecting for many adapters concurrently in mixed batches.

---

## Shrink/Expand + Abort/Resume Semantics (LoRA-aware)

### Shrink (time-sharing preemption)
Unchanged from the shared protocol, with two LoRA-specific clarifications:
1) Shrink must release **all** GPU memory (base + adapters + KV). Any “cheap sleep” that preserves weights is not valid for time-sharing shrink.
2) If requests (for any adapters) are running on a worker in `P`, we abort and retry (no mid-turn resume). If we are mid-update for adapter X, we must not schedule X into mixed batches until X update completes (Decision C1).

### Resume strategy for shrink/expand (Option A only: drop-on-shrink, sync-on-expand)
Use the simplest contract:
- Any workers in the shrunk subset drop everything (base + all adapters + KV).
- On expand/resume, newly activated workers load base (from trainer-side bucket cache) and load/warm needed adapters before opening admission.
- Shrink-to-zero is the “remaining set is empty” special case: all rollout workers drop everything; the next expand loads base+adapters on the newly activated set (which is the full active set).

### Expand (resume or grow)
Expand must guarantee:
- base model weights are present for the active base spec, and
- for LoRA mode: adapters needed for the next scheduled mixed batches are loaded (or loadable before admission opens).

Recommended operational approach:
- Coordinator maintains per-adapter queues and builds mixed batches by drawing from multiple queues.
- On expand/resume, the coordinator warms only adapters with queued work (based on its local queues and `resident_adapters_by_worker`) before dispatching mixed batches to newly activated workers.
- Mixed-batch fairness (Phase 1, recommended):
  - Use a simple no-starvation rule in the mixed-batch builder: each `adapter_id` with non-empty queue gets at least 1 prompt admitted per scheduling tick (up to batch capacity), then fill remaining slots proportional to backlog (or round-robin).
  - If an adapter has consistently low volume, this guarantees eventual service without requiring the central scheduler to be adapter-aware.

### Adapter activation (“activate LoRA”) without resizing
Adapter updates do not require scheduler involvement unless you want scheduling policy to depend on them.

Recommended coordinator behavior (Decision C1):
1) Stop starting new trajectories for adapter X (adapter-scoped admission close).
2) Abort in-flight requests/trajectories for adapter X and wait abort ACK.
3) Load/activate adapter X@v_new on any active workers that may serve adapter X (default: all currently active rollout workers; coordinator may use `resident_adapters_by_worker` to target a smaller set).
4) Reopen adapter X admission **only after** all targeted active workers report “adapter X is ready at v_new” (avoid mixed X@old and X@new serving simultaneously).

If a shrink arrives in the middle of steps 2–3, the same fail-fast rules apply:
- do not proceed if abort ACK cannot be established.
- If shrink/worker failure interrupts step 3, keep adapter X admission closed and retry activation for the remaining active worker set (and any newly expanded workers will load X on resume via warmup).

---

## Framework Mapping (Reference-first: ROLL, then SkyRL-train, then NeMo-RL)

### ROLL (reference target)
Why it’s a good reference:
- already passes `lora_request` into vLLM and has a clear abort path for `REQUEST_RETRY`.

Plan deltas for multi-LoRA:
- Replace “pick first LoRA id from `list_loras()`” logic with “select adapter_id per request in a mixed batch”.
  - Current placeholder behavior exists in `third_party/ROLL/roll/distributed/strategy/vllm_strategy.py:172`.
- Ensure the coordinator can build mixed batches by drawing from multiple per-adapter queues, and provide a per-request `lora_request` list (one entry per prompt) matching each prompt’s adapter_id.
- Ensure adapter activation can pause only adapter X admission while other adapters continue generating (Option C1).
- Ensure shrink-triggered abort targets only the affected requests/workers and that retry re-issues the same `(trajectory_id, turn_id)` with incremented `attempt`.

### SkyRL-train
Why it’s useful:
- it already has a LoRA load path via vLLM `add_lora` (`third_party/SkyRL/skyrl-train/skyrl_train/inference_engines/vllm/vllm_engine.py:313`).

Plan deltas for multi-LoRA:
- make LoRA load deterministic: adapter_id should map to a known LoRA in the engine (instead of generating random int ids).
- ensure time-sharing shrink uses deep sleep (`level=2`) even when LoRA is enabled (SchedRL invariant).
- align request_id construction with SchedRL deterministic IDs for abort+retry.

### NeMo-RL
Why it’s useful:
- it’s a clean reference for LoRA semantics (base frozen) and for a “cheap wake/sleep + selective sync” story.

Plan deltas for multi-LoRA:
- treat adapter artifacts as first-class cached items (parallel to base checkpoint cache).
- if/when RL LoRA rollouts are enabled, mirror the same adapter-batch boundaries for activation.

---

## What We’re NOT Doing (Explicitly Out of Scope)

- Mixed versions of the **same** adapter within a single engine without an explicit multi-version residency mechanism (Option C3).
- Composing multiple LoRAs simultaneously for a single trajectory (adapter stacking).
- Mid-turn suspend/resume (no “resume running turn” across shrink; only abort+retry).
- Sharing a single rollout engine group across multiple pipelines (isolation assumption stands).
- SGLang-first design; this plan is vLLM-first.

---

## Implementation Phases

## Phase 1: Protocol Types + Contracts (mode + model spec + cache)

### Overview
Define the protocol additions so frameworks can implement the same adapter surface for both full FT and multi-LoRA.

### Changes Required
1) Extend protocol schema to add `ModelMode` (pipeline registration) + `ActiveModelSpec` (active base + adapters) and adapter artifact definitions (see “Protocol Extensions”).
2) Extend cache contract from “checkpoint-only” to “base + adapter artifacts” with clear GC rules.
3) Document `expand_workers` warmup mechanism: coordinator-driven warmup based on local per-adapter queues (no scheduler-provided warmup payload).

### Success Criteria
#### Automated Verification
- [ ] N/A (doc/protocol-only phase)

#### Manual Verification
- [ ] Protocol doc has a single, unambiguous definition of “active model” in both modes.
- [ ] Shrink/expand ordering remains identical to the base protocol and is LoRA-safe.

---

## Phase 2: ROLL Reference Wiring (mixed-adapter batching + adapter activation)

### Overview
Implement multi-LoRA semantics in the ROLL reference path first, because it already matches `QUIESCE-by-abort` + `REQUEST_RETRY` patterns.

### Changes Required
1) Maintain per-adapter queues in the coordinator (one queue per adapter_id), plus a **mixed-batch builder** that draws prompts from multiple adapters for each scheduling tick.
2) Ensure vLLM calls use a per-prompt `lora_request` list (one `LoRARequest` per prompt) matching each prompt’s adapter_id.
3) Implement adapter activation (`adapter_id -> adapter_version`) and ensure it happens only at safe boundaries (Decision C1 recommended).
4) Ensure deterministic request IDs include adapter_id, and aborted work retries with incremented attempt (Decision D1).
5) Surface an adapter removal API for GC:
   - Required capability: unload/remove an adapter version from the engine so repeated adapter updates do not accumulate VRAM/LoRA slots indefinitely.
   - Define the engine-facing hook as `remove_lora(adapter_id, adapter_version)` (or equivalent backend API like vLLM “unload LoRA adapter”).
   - If removal is not available in the embedded surface, Phase 2 must fall back to a safe-but-heavier strategy for adapter GC (e.g., brief deep-sleep/restart of the engine group, then warm only the currently-needed adapters before reopening admission).
   - Default eviction policy (Phase 1, recommended):
     - Enforce a per-worker `max_resident_adapters` limit.
     - If a load would exceed the limit, evict the least-recently-used adapter with no in-flight work (LRU by “last used” timestamp updated on dispatch).
     - If no evictable adapter exists (all resident adapters have in-flight work), fail fast: do not attempt the load and return a clear error (avoid OOM by uncontrolled growth).

### Success Criteria
#### Automated Verification
- [ ] ROLL suite passes: `cd third_party/ROLL && make test`

#### Manual Verification
- [ ] Run a multi-adapter rollout where a single inference batch contains prompts from multiple adapters, and produced trajectories are correctly tagged per `(adapter_id, adapter_version)`.
- [ ] Trigger shrink during active mixed-adapter generation; aborted requests retry and complete on remaining workers with no side effects duplicated.

---

## Phase 3: SkyRL-train Wiring (adapter identity + deep-shrink correctness)

### Overview
Align SkyRL’s existing LoRA load hooks with SchedRL’s adapter identity + shrink/expand requirements.

### Changes Required
1) Make adapter IDs stable and map them to vLLM LoRA IDs deterministically (avoid random `time_ns` ids for “the adapter identity”).
2) Ensure SchedRL shrink uses deep release semantics (`level=2`) even if LoRA is enabled (time-sharing invariant).
3) Add deterministic request IDs and adapter-aware retry routing (consistent with Phase 2).

### Success Criteria
#### Automated Verification
- [ ] SkyRL-train test or smoke run command per existing docs (no new tests added in this phase)

#### Manual Verification
- [ ] While mixed-adapter generation continues, update adapter X and verify: requests for X are temporarily not scheduled (or are retried) until X is activated, and then resume using the new adapter version; other adapters continue uninterrupted.
- [ ] Shrink mid-flight fully releases GPU memory and training continues after expand.

---

## Phase 4: NeMo-RL Wiring (adapter artifacts + activation boundaries)

### Overview
Adopt the same adapter artifact + activation semantics for NeMo-RL, reusing its selective/cheap wake/sleep patterns where applicable.

### Changes Required
1) Extend the “artifact cache” notion to include adapter artifacts and GC (parallel to base cache).
2) Implement adapter-scoped activation (Decision C1) in the rollout path when mixed-adapter multi-LoRA rollouts are enabled.

### Success Criteria
#### Automated Verification
- [ ] NeMo-RL tests pass: `cd third_party/nemo-rl && uv run --group test pytest -q`

#### Manual Verification
- [ ] Adapter updates can be staged/activated without requiring full base resync.

---

## Testing Strategy (End-to-End)

### Unit / Component
- ROLL: `cd third_party/ROLL && make test`
- NeMo-RL: `cd third_party/nemo-rl && uv run --group test pytest -q`

### Manual (multi-pipeline safety)
1) Run two pipelines concurrently under SchedRL:
   - one in full-FT mode,
   - one in multi-LoRA mode with multiple adapters.
2) Force shrink/expand cycles during active rollouts.
3) Verify:
   - no admission on inactive workers,
   - abort ACK gating is respected,
   - trajectory tags correctly record `(base_version, adapter_id, adapter_version)`.

---

## Performance Considerations

- Adapter updates should be much cheaper than base sync; avoid turning adapter updates into “full sync events”.
- Don’t preload all adapters on expand; warm only adapters with queued work.
- Keep shrink strict: full GPU release is non-negotiable for time-sharing, even if it drops adapter caches.

---

## Risks & Mitigations (ROLL-first)

### Risk 1: Cache service coupling (checkpoint cache → artifact cache)
**Issue**: existing “trainer CPU bucket list” code may be shaped around monolithic checkpoints. A generic refactor into an “artifact cache” could be larger than expected.

**Mitigation (Phase 1, recommended)**:
- Do not attempt a full “one cache to rule them all” refactor initially.
- Prefer a **wire-format extension** over introducing new “cache managers”:
  - Extend the existing “bucket list” / checkpoint cache RPC payload to carry a list of **artifact entries**, where each entry is either:
    - a base artifact (bucketized weights, same as today), or
    - an adapter artifact (e.g., file path / URI / handle for `(adapter_id, adapter_version)`).
  - Keep the existing trainer-owned cache actor/service as the single source of truth; do not add a second cache service for adapters in Phase 1.
  - If we want code cleanliness, implement thin helpers/wrappers (`get_base(...)`, `get_adapter(...)`) over the same underlying payload, but avoid introducing a new “checkpoint manager” layer.
- Add a small pin/unpin (or refcount) contract to avoid GC races during expand/resume (scheduler dispatch must not observe 404s for the target base/adapters).

### Risk 2: vLLM `add_lora` concurrency (embedded API)
**Issue**: `add_lora(...)` mutates shared engine state and may not be safe to call concurrently with ongoing generation in the exact vLLM build used by ROLL/SkyRL.

**Mitigation (Phase 2, recommended)**:
- Treat adapter updates as “adapter-scoped gating + brief global critical section”:
  1) **Coordinator**: stop scheduling adapter X into new mixed batches (adapter-scoped admission close).
  2) **Coordinator**: abort in-flight X requests and wait abort ACK (targeted by request_id; see Risk 3 mapping).
  3) **Coordinator**: enter a short global control critical section for the engine group (fallback is global `QUIESCE-by-abort`).
  4) **Worker/Engine**: run `add_lora(adapter_id, artifact)` on each active worker that may serve X; return a per-worker “ready” ACK (or fail fast).
  5) **Coordinator**: update `resident_adapters_by_worker` to reflect X@v_new, then exit the critical section and resume mixed-adapter generation.
  6) **Coordinator**: reopen adapter X admission only after all targeted workers report ready at v_new.
- Before optimizing to “pure adapter-scoped update while other adapters continue”, validate behavior on the project’s vLLM build with a minimal reproduction (no new test files required).
- In the same validation pass, confirm whether “overwrite in place” is safe (re-`add_lora` same adapter_id) or whether updates must be “remove then add” (requires a surfaced `remove_lora`/unload API).

### Risk 3: Abort granularity in mixed-adapter batches
**Issue**: “abort adapter X” requires mapping to concrete in-flight request IDs when batches contain multiple adapters. If this mapping is missing, the safest fallback is abort-all on the targeted workers.

**Mitigation (v1-safe default)**:
- For **shrink**: abort is worker-subset scoped; abort all in-flight work on workers in `P` and retry elsewhere (SchedRL core path).
- For **adapter update** (preferred): implement adapter→request mapping so we don’t need “abort-all”:
  - Maintain reverse indexes at the routing boundary (e.g., in ROLL `RequestScheduler`):
    - `request_id -> adapter_id`
    - `adapter_id -> set[request_id]` (active only)
    - (optional) `worker_index/dp_rank -> set[request_id]` for fast subset aborts
  - On submit: insert into the indexes.
  - On completion/abort ACK: remove from the indexes.
  - Adapter update “abort X” enumerates `adapter_id -> request_ids` and aborts exactly those ids (then waits for ACK), while other adapters continue.
  - If the mapping is not yet implemented, fall back to the same short global quiesce around `add_lora(...)` rather than attempting a partial abort that could miss X requests.

---

## NEW IMPLEMENTATION PLAN (Porting Multi-LoRA to ROLL_schedrl)

Based on codebase research, here is the concrete implementation plan:

### Key Files to Port/Modify

#### 1. `roll/pipeline/base_pipeline.py`
**Current State**: ROLL_schedrl has basic `model_update()` without adapter subset support
**Change Required**: Add `model_update_lora_subset()` method (copy from ROLL_multi_lora)
```python
def model_update_lora_subset(self, global_step: int, *, adapters_to_update: set[str] | None = None) -> dict:
    """Adapter-subset model update helper for multi-LoRA pipelines."""
    metrics: dict = {}
    for model_update_group in self.model_update_groups:
        metrics.update(model_update_group.model_update(step=global_step, adapters_to_update=adapters_to_update))
    return metrics
```

#### 2. `roll/distributed/executor/model_update_group.py`
**Current State**: `model_update()` takes only `step` parameter
**Change Required**: Add `adapters_to_update` parameter and pass to workers
```python
def model_update(self, step=None, adapters_to_update: set[str] | None = None):
    if step % self.frequency != 0:
        return {}
    kwargs = {"model_update_name": self.model_update_name}
    if adapters_to_update is not None:
        kwargs["adapters_to_update"] = sorted(adapters_to_update)
    # ... rest of implementation
```

#### 3. Create `roll/schedrl_adapter/multi_lora_pipeline.py`
**New File**: `SchedRLMultiLoraPipeline` class combining:
- SchedRL integration patterns from `SchedRLConcurrentPipeline`:
  - `_notify_ready_to_release_actor_infer()`
  - `_request_actor_infer_gpus()` / `_release_static_cluster()`
  - `resize_infer()` with shrink/expand
- Multi-LoRA patterns from `AgenticMultiLoraPipeline`:
  - Per-tag rollout schedulers (`self.rollout_schedulers: dict[str, Any]`)
  - `lora_step: dict[str, int]` for per-adapter step tracking
  - `dirty_adapters: set[str]` for tracking updates
  - `model_update_lora_subset()` calls

**Key Integration Points**:
1. **Initialization**: Create per-tag `RolloutScheduler` instances like `AgenticMultiLoraPipeline`
2. **Run Loop**: 
   - Request GPUs via `_request_actor_infer_gpus()` before rollout
   - Collect batches from per-tag schedulers
   - Track `dirty_adapters` from batch `lora_name`
   - Call `model_update_lora_subset(global_tick, adapters_to_update=dirty_adapters)`
   - Call `_notify_ready_to_release_actor_infer()` after rollout
3. **Shrink/Expand**: Handle per-tag scheduler shrink/expand in `resize_infer()`

#### 4. `roll/utils/lora_routing.py`
**Action**: Copy from ROLL_multi_lora to ROLL_schedrl (if not present)
- `normalize_domain()` - normalize adapter names
- `get_lora_name_array()` - extract lora_name from batch
- `resolve_microbatch_lora_name()` - validate homogeneous lora_name in batch

#### 5. `roll/schedrl_adapter/adapter.py`
**No Change Required**: Keep `sleep_level=2` requirement for both single-LoRA and multi-LoRA
- Multi-LoRA will broadcast both backbone + all active adapters on selective sync
- This is handled by the existing `ModelUpdateService` pattern with adapter-aware sync

#### 6. `roll/distributed/scheduler/rollout_scheduler.py` (and `GroupQueueManager`)
**Change Required**: Enable passing `adapter_id` through to `GroupQueueManager` for progress reporting.
- Update `GroupQueueManager.__init__` to extract `adapter_id` from `env_manager_config.tags[0]` (since multi-LoRA uses one scheduler per tag).
- Update `GroupQueueManager._maybe_emit_progress` to include `adapter_id` in `metrics`.
- *Note*: No changes needed to `RolloutScheduler` signature if `adapter_id` is derived from `env_manager_config`.

### Implementation Phases

#### Phase 1: Base Pipeline Updates
**Files**:
- `roll/pipeline/base_pipeline.py`: Add `model_update_lora_subset()`
- `roll/distributed/executor/model_update_group.py`: Add `adapters_to_update` parameter
- `roll/distributed/scheduler/rollout_scheduler.py`: Add `adapter_id` extraction to `GroupQueueManager`
- `roll/third_party/megatron/model_update.py`: **CRITICAL** Port adapter-aware logic from `ROLL_multi_lora`.
    - Update `MegatronWeightUpdater.model_update` to accept `adapters_to_update`.
    - Update `gather_all_hf_weights`, `gather_pp_stage_hf_weights` to accept `adapter_name`.
    - Implement `_colocated_model_update` and `_separated_model_update` loop over adapters.

- `roll/distributed/strategy/megatron_strategy.py`: **CRITICAL** Generalize SchedRL's CPU bucket cache for multi-LoRA.
    - **Current**: `_build_latest_bucket_cache` serializes all weights into one bucket list.
    - **New**: Separate base model weights from adapter weights.
        - Identify adapter weights via `adapter_name` or parameter analysis.
        - Store in `_cache_map` with structure supporting multiple components: `cache_key -> { "base": [buckets], "adapters": { "adapter_1": [buckets], ... } }` or distinct keys.
        - **Implementation Detail**:
            - Modify `_build_latest_bucket_cache` to accept optional `adapters_to_cache`.
            - If `adapters_to_cache` is provided, only serialize and cache those specific adapters (and base if needed/changed).
            - Use a composite key or nested dictionary in `self._cache_map` to store base and adapter artifacts separately.
            - Ensure `promote_active_checkpoint` can mark a composite state (base version + specific adapter versions) as active.
    - Update `promote_active_checkpoint` to handle promoting the base and relevant adapters.
    - Update `selective_sync_active_cache`:
        - Accept `adapters_to_sync` list (derived from active configuration).
        - Broadcast base bucket (if needed/missing) AND specific adapter buckets to the target workers.
        - Ensure atomicity or proper sequencing (base first, then adapters).

- `roll/schedrl_adapter/model_update_service.py`: **CRITICAL** Generalize selective sync orchestration.
    - Update `sync_selected_workers`:
        - Accept `adapters_to_sync` (optional list of adapter IDs).
        - Pass this list to `selective_sync_active_cache` on the sender (train worker).
        - **Implementation Detail**:
            - In `sync_selected_workers`, pass `adapters_to_sync` to `_build_comm_plan_for_sender` (if needed for group sizing, though likely not if topology is static).
            - Critical: Pass `adapters_to_sync` to `worker.selective_sync_active_cache.remote(...)`.
            - Verify that `selective_sync_active_cache` on the worker iterates through `adapters_to_sync` and performs the broadcast for each adapter artifact found in the cache.
        - Ensure the `comm_plan` and `setup_collective_group` can handle multiple sequential broadcasts (base then adapters) if they use the same group, or separate groups if needed (reusing the same group is preferred for efficiency).

**Verification Task**:
- [ ] **Critical**: Verify `ActorWorker` (in `roll/pipeline/rlvr/actor_worker.py` / `roll/pipeline/base_worker.py`) supports `selective_sync_active_cache` with multiple adapters.
      - `ModelUpdateService` (used by SchedRL expand) calls `selective_sync_active_cache`.
      - We must ensure that when `adapters_to_update` are active on the trainer, the worker correctly receives and loads both the base and the adapters during sync.
      - **Action**: Add a verification test where we simulate an expand call with `active_adapters=["lora_a", "lora_b"]` and verify the worker has both adapters loaded after sync.
      - **Specific check**: Ensure `ActorWorker.update_parameter_in_bucket` (or the underlying strategy method) correctly handles the received buckets and loads them into the correct LoRA adapter slots in the model.

**Success Criteria**:
- [ ] `make test` passes in `external/ROLL_schedrl`
- [ ] `model_update_lora_subset()` method exists and delegates to ModelUpdateGroup
- [ ] `GroupQueueManager` reports `adapter_id` in metrics
- [ ] `MegatronTrainStrategy` can cache and selectively broadcast specific adapters alongside the base model.
- [ ] `ModelUpdateService` can orchestrate the broadcast of specific adapters.

#### Phase 2: LoRA Routing Utilities
**Files**:
- `roll/utils/lora_routing.py`: Copy from ROLL_multi_lora

**Success Criteria**:
- [ ] `normalize_domain()`, `get_lora_name_array()`, `resolve_microbatch_lora_name()` available

#### Phase 2.5: Adapter GC / Eviction
**Overview**:
Implement adapter eviction to prevent OOM as new adapters are loaded.

**Changes Required**:
1. **Surface Removal API**: Implement `remove_lora(adapter_id)` (or equivalent) in the engine/strategy layer.
2. **Implement Eviction Logic**:
   - Track resident adapters per worker.
   - Enforce a `max_resident_adapters` limit.
   - When loading a new adapter exceeds the limit:
     - Identify the least-recently-used (LRU) adapter with no in-flight work.
     - Evict it.
     - If no adapter can be evicted (all have in-flight work), fail fast (skip load/schedule).

**Success Criteria**:
- [ ] Adapter eviction logic handles OOM scenarios by unloading unused adapters.
- [ ] Validated that vLLM (or target engine) supports unloading/removing adapters without restarting.

#### Phase 3: Multi-LoRA Pipeline Coordinator
**Files**:
- `roll/schedrl_adapter/multi_lora_pipeline.py`: New file

**Key Implementation Details**:
```python
class SchedRLMultiLoraPipeline(BasePipeline):
    """SchedRL-controlled multi-LoRA pipeline.

    Combines:
    - SchedRL GPU allocation/release patterns from SchedRLConcurrentPipeline
    - Multi-LoRA per-tag scheduling from AgenticMultiLoraPipeline
    """

    def __init__(self, *, pipeline_id: str, pipeline_config: Any):
        # ... initialize like AgenticMultiLoraPipeline ...
        # ... but also setup SchedRL scheduler connection ...

        # IMPORTANT: When porting from AgenticMultiLoraPipeline, remove the
        # `sleep_level=1` runtime check. SchedRL requires `sleep_level=2`.
        # Ensure proper validation of sleep_level=2 with LoRA reload.

    def run(self):
        # Similar to AgenticMultiLoraPipeline.run() but with SchedRL integration:
        # 1. Request GPUs: self._request_actor_infer_gpus(global_step=global_tick)
        # 2. Run rollout loop with per-tag schedulers
        # 3. Release GPUs: self._notify_ready_to_release_actor_infer(global_step=global_tick)
        #
        # IMPORTANT: Do NOT copy the manual `shrink_sampler` / partial GPU offloading logic
        # from `AgenticMultiLoraPipeline.run`. SchedRL handles resource arbitration via
        # the request/release cycle (Phase 4 of the step).
        # Use `_release_and_request_static_cluster` pattern from `SchedRLConcurrentPipeline`.

    def resize_infer(self, *, dp_ranks_to_remove: List[int], dp_ranks_to_add: List[int]):
        """Reuse SchedRLConcurrentPipeline.resize_infer() pattern for per-tag schedulers.

        Instead of operating on train_rollout_scheduler + val_rollout_scheduler,
        operate on all per-tag schedulers in self.rollout_schedulers.
        """
        # ... validation from SchedRLConcurrentPipeline ...

        schedulers = list(self.rollout_schedulers.values())

        if dp_ranks_to_remove:
            # Shrink all per-tag schedulers
            for sched in schedulers:
                self._shrink_scheduler(sched, dp_ranks_to_remove)
        else:
            # Expand: Coordinator-driven Warmup
            # 1. Identify "Active Model Spec" (base + all currently active/queued adapters).
            #    (Or query local queues to find needed adapters)
            # 2. Warm up these adapters on the new workers using `expand_workers`.
            #    - This ensures workers have necessary state before admission opens.

            # Expand all per-tag schedulers
            for sched in schedulers:
                self._expand_scheduler(sched, dp_ranks_to_add)

        return ActionResponse(success=True)
```

**Key Design Decision**: Reuse `_shrink_workers()` / `_expand_workers()` helper methods from `SchedRLConcurrentPipeline` (or refactor into shared helpers) to avoid code duplication.

**Multi-LoRA Sync on Expand**: When expanding inference workers with sleep_level=2:
1. Backbone/base model is broadcast from training workers
2. All active LoRA adapters are broadcast alongside the backbone
3. This ensures inference workers have complete model state (backbone + adapters)

**Progress Reporting**: Each per-tag `GroupQueueManager` reports progress independently:
- `ProgressReport.pipeline_id`: Identifies which pipeline the report belongs to
- `ProgressReport.metrics["adapter_id"]`: Identifies which specific adapter within the pipeline
- **Aggregation Location**: SchedRL Scheduler (e.g., `RoundRobinGlobalScheduler`) aggregates per-adapter reports (queued, inflight) to track total pipeline load and make admission decisions. No aggregation needed in the pipeline itself.

**Success Criteria**:
- [ ] Pipeline initializes with per-tag rollout schedulers
- [ ] Requests/releases GPUs via SchedRL scheduler
- [ ] Handles shrink/expand via `resize_infer()` with same API as `SchedRLConcurrentPipeline`
- [ ] Selective sync broadcasts both backbone + all active adapters on expand
- [ ] Each adapter reports progress independently with `adapter_id` in metrics
- [ ] `sleep_level=2` validated with LoRA reload (no weight loss).

#### Phase 4: Adapter Registration
**Files**:
- `roll/schedrl_adapter/adapter.py`: Add support for creating `SchedRLMultiLoraPipeline`

**Changes**:
```python
def create_coordinator(self, *, pipeline_config: Any, multi_lora: bool = False):
    if multi_lora:
        from roll.schedrl_adapter.multi_lora_pipeline import SchedRLMultiLoraPipeline
        Coordinator = ray.remote(SchedRLMultiLoraPipeline)
    else:
        from roll.schedrl_adapter.concurrent_pipeline import SchedRLConcurrentPipeline
        Coordinator = ray.remote(SchedRLConcurrentPipeline)
    # ... rest of method
```

**Success Criteria**:
- [ ] Adapter can create both single-LoRA and multi-LoRA coordinators

#### Phase 5: Testing
**Test Scenarios**:
1. Single adapter training (regression test)
2. Multi-adapter training with 2+ adapters
3. Shrink/expand during multi-adapter rollout
4. Adapter subset updates (only dirty adapters synced)

**Commands**:
```bash
cd external/ROLL_schedrl
make test
```

### Code Reuse Strategy

| Component | ROLL_multi_lora Source | Reuse Approach |
|-----------|----------------------|----------------|
| `model_update_lora_subset()` | `base_pipeline.py:85` | Copy method |
| `ModelUpdateGroup.model_update()` | `model_update_group.py:28` | Add parameter |
| LoRA routing utilities | `utils/lora_routing.py` | Copy file |
| Per-tag scheduler setup | `agentic_multi_lora_pipeline.py:136-180` | Adapt to SchedRL |
| Run loop structure | `agentic_multi_lora_pipeline.py:500-1028` | Adapt with SchedRL calls |
| `resize_infer()` API | `concurrent_pipeline.py:985-1040` | **Reuse directly** - same API for per-tag schedulers |
| `_shrink_workers()` / `_expand_workers()` | `concurrent_pipeline.py` | Refactor to shared helpers or inherit |

### Implementation Notes

#### `resize_infer()` Reuse Strategy
The `resize_infer()` API from `SchedRLConcurrentPipeline` is reused directly:
- **Same signature**: `resize_infer(*, dp_ranks_to_remove: List[int], dp_ranks_to_add: List[int])`
- **Same behavior**: Shrink or expand inference workers by DP rank
- **Different scope**: Applied to all per-tag schedulers instead of just train/val schedulers

This allows the SchedRL scheduler to control multi-LoRA pipelines the same way it controls single-LoRA pipelines.

#### Helper Method Reuse
Consider refactoring `_shrink_workers()` and `_expand_workers()` from `SchedRLConcurrentPipeline` into shared helper methods or a common base class to avoid code duplication between `SchedRLConcurrentPipeline` and `SchedRLMultiLoraPipeline`.

### Design Decisions

1. **Sleep Level**: Use `sleep_level=2` for both full fine-tune and multi-LoRA (SchedRL time-sharing requirement).
   - For multi-LoRA: On selective model update, broadcast **both** the backbone and all active LoRA adapters to newly scheduled actor_infer engines
   - The `ModelUpdateService` needs to handle multi-adapter sync

2. **Progress Reporting**: Each adapter reports its own progress independently; SchedRL aggregates.
   - Each `GroupQueueManager` (per adapter) calls `report_progress()` independently.
   - **Metrics**: Each report includes `metrics["adapter_id"]` and `metrics["adapter_progress"]` (unclamped `completed / target`).
   - **Aggregation**: SchedRL aggregates `queued` and `inflight` counts from all adapter reports to track total pipeline load.
   - `ProgressReport.pipeline_id` identifies the pipeline.

3. **Code Sharing**: Should we create a shared base class (e.g., `SchedRLPipelineBase`) containing common methods like `_shrink_workers()`, `_expand_workers()`, `_notify_ready_to_release_actor_infer()`?

---

## References

- Shared protocol: `design_doc/multi-pipeline-adaptation-plan.md`
- Dual-mode scheduler plan: `thoughts/shared/plans/2026-01-28-schedrl-dual-mode-final-plan.md`
- ROLL adaptation plan: `thoughts/shared/plans/2026-01-28-roll-schedrl-adaptation.md`
- SkyRL-train adaptation plan: `thoughts/shared/plans/2026-01-28-skyrl-train-adaptation-plan.md`
- NeMo-RL adaptation plan: `thoughts/shared/plans/2026-01-28-nemo-rl-schedrl-adaptation.md` (deferred; archived)
- ROLL_multi_lora: `external/ROLL_multi_lora/roll/pipeline/agentic/agentic_multi_lora_pipeline.py`
- ROLL_schedrl: `external/ROLL_schedrl/roll/schedrl_adapter/concurrent_pipeline.py`

# Multi-LoRA Plan Refinement: Simplifications

**Date**: 2026-02-04
**Original Plan**: `thoughts/shared/plans/2026-02-02-schedrl-multi-lora-adapter-extension.md`

## 1. Unnecessary Class Complexity Detected

The original plan introduces a complex `ActiveModelSpec` hierarchy:

```python
# Original Proposal
class BaseSpec:
    base_version: int
    base_uri: Optional[str]

class AdapterSpec:
    adapter_id: str
    adapter_version: int
    adapter_uri: Optional[str]

class ActiveModelSpec:
    mode: ModelMode  # FULL_FT vs MULTI_LORA
    base: BaseSpec
    adapters: Dict[str, AdapterSpec]
```

### Problem: Over-engineering "BaseSpec" for Multi-LoRA
In `MULTI_LORA` mode, the base model is **frozen** and constant for the entire run (Decision A1).
- The `base_version` never changes (it's effectively a static config).
- The `base_uri` never changes.

In `FULL_FT` mode, there are no adapters, and `base_version` is the *only* thing that changes.

The current protocol treats "weights" as a single monotonic integer `checkpoint_version`. We can preserve this simplicity for `FULL_FT` without wrapping it in a `BaseSpec` object, and treat `MULTI_LORA` as "base is static config, only adapters have versions".

## 2. Simplification Proposal

### A. Flatten the Protocol State
Instead of a nested `ActiveModelSpec`, use a flat model where `base_version` is always present and `adapters` is an (often-empty) `dict`.

**Proposed simplified state:**
- **FULL_FT Mode**:
  - `base_version: int` (maps to today's `checkpoint_version` / `active_checkpoint_version`)
  - `adapters: Dict[str, int] = {}` (empty)

- **MULTI_LORA Mode**:
  - `base_version: int` (constant for the run; use `-1` as the frozen-base sentinel)
  - `adapters: Dict[str, int]` (map `adapter_id -> adapter_version`)
  - Validation rule: `base_version == -1` is only permitted when the pipeline’s registered `model_mode == MULTI_LORA`; in `FULL_FT`, `base_version MUST be >= 0`.
  - Ordering note: `base_version == -1` must never be used in “newer version wins” comparisons (only equality + validation is meaningful).

**Benefit**:
- Removes `BaseSpec` and `AdapterSpec` classes entirely from the hot path.
- "Version" remains a simple integer for the 90% case (FULL_FT).
- "Adapters" is just a dictionary of `{id: version}`.
- URIs should be resolved by the cache layer/static config, not passed around in every heartbeat/action.
- `ModelMode` should be a **registration-time constant per pipeline**, not carried in the active model state/messages.

### B. Remove "AdapterSpec" Object
We don't need an `AdapterSpec` object to track `adapter_id`, `adapter_version`, and `adapter_uri`.
- `adapter_id` is the key.
- `adapter_version` is the value.
- `adapter_uri` is strictly a function of `(adapter_id, adapter_version)` handled by the artifact cache, not the scheduler protocol.

### C. Simplify "Expand" Signature
Original Plan:
```python
expand_workers(
    worker_indices,
    base_version,  # Explicit argument
    action_id,
    activation_epoch
)
```

**Refinement**:
- `base_version` is redundant in `MULTI_LORA` (it's static).
- In `FULL_FT`, `base_version` is the active checkpoint version.
- To keep the scheduler workload-agnostic, **do not pass adapter state in `expand_workers(...)`** in Phase 1:
  - the coordinator already owns per-adapter queues and can warm adapters on the newly activated workers before opening admission,
  - the target adapter set is derived from coordinator-local backlog, not a scheduler input.
- **Decision**: keep `expand_workers(worker_indices, base_version, action_id, activation_epoch)` and eliminate any `BaseSpec`/`AdapterSpec` object usage.

## 3. Revised Action Plan

1.  **Modify `schedrl/protocol/types.py` (Planned)**:
    - Do **NOT** create `BaseSpec` or `AdapterSpec` classes.
    - Define `ActiveModelSpec` as a dataclass with:
      - `base_version: int` (meaningful in FULL_FT; constant in MULTI_LORA)
      - `adapters: Dict[str, int]` (adapter_id -> version; empty in FULL_FT)
    - Treat `ModelMode` as a **pipeline registration field** (constant for the run).

2.  **Update `ActiveModelSpec` definition in 2026-02-02 plan**:
    - "Flatten the spec: `ActiveModelSpec` contains `base_version: int` and `adapters: dict[str, int]`. URIs are resolved by cache, not protocol."

## 4. Why this matters
- **Less Serialization**: Passing lightweight `dict[str, int]` is cheaper than serializing list of objects.
- **Easier Integration**: Frameworks (ROLL/SkyRL) just need to track a version dict, not map internal objects to SchedRL protocol classes.
- **Forward Compat**: If we add `base_uri` dynamic changing later, we can add it then. For now, YAGNI (You Ain't Gonna Need It) applies strongly to "Base Spec" objects in a frozen-base LoRA regime.

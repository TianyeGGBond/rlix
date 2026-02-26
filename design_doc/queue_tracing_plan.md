# Queue Visualization for GPU Tracing

**Date**: 2024-02-25
**Status**: IMPLEMENTATION-READY (Revised after review)

---

## Overview

Add queue request visualization to GPU timeline traces. This shows:
- **Counter tracks**: Queue depth per priority over time
- **Slice tracks**: Individual request wait time in queue (per-cluster, not per-priority)

---

## Critical Design Decision: Per-Cluster Slice Tracks

**Problem**: `NormalTrack.close()` is LIFO - it closes "the last 'open' call" ([`_tgen.py#L23`](../external/tg4perfetto/src/tg4perfetto/_tgen.py#L23)). If multiple requests share one track and complete out of order, wrong slices get closed.

**Solution**: Use **per-cluster slice tracks**, not per-priority. Store the track handle in pending state.

```python
# WRONG: Per-priority tracks (LIFO breaks out-of-order completion)
_pending_request_start_ns: Dict[str, Tuple[int, Priority]]  # cluster_id -> (start_ns, priority)
_queue_slice_tracks: Dict[str, NormalTrack]  # priority_key -> track

# CORRECT: Per-cluster tracks, store handle
_pending_queue_trace_state: Dict[str, Tuple["NormalTrack", int, Priority]]  # cluster_id -> (track, start_ns, priority)
```

---

## Visualization Example

```
Timeline View in Perfetto UI
═══════════════════════════════════════════════════════════════════════════

GPU 0                    ████████████████████░░░░░░░░████████████
                         └─ actor_train ──────┘    └─ actor_infer ─┘

GPU 1                    ████████████████████░░░░░░░░████████████
                         └─ actor_train ──────┘    └─ actor_infer ─┘

───────────────────────────────────────────────────────────────────────────
QDEPTH_TRN (counter)         ▲2        ▲1
                              ████      ██
                              ████      ██
                              ████      ██
                         ▲3   ████  ▼1  ██  ▼0
                         ████ ████ ████ ██ ████
                         ████ ████ ████ ██ ████
───────────────────────────────────────────────────────────────────────────
QUEUE_TRN_p1_actor_train ├─────wait─────┤
QUEUE_TRN_p2_actor_train ├─────wait─────────────┤
QUEUE_TRN_p3_actor_train ├──wait──┤
                         
                         ↑     ↑    ↑    ↑     ↑
                        t0    t1   t2   t3    t4
                         
Events:
  t0: req1, req2, req3 enqueued  → counter jumps to 3
  t1: req1 fulfilled             → counter drops to 2
  t2: req3 fulfilled             → counter drops to 1  
  t3: req2 fulfilled             → counter drops to 0
  t4: new request                → counter jumps to 1
```

---

## Implementation Plan

**File:** `schedrl/scheduler/scheduler.py`

### 1. TYPE_CHECKING Import (line ~49)

Add `CounterTrack` to type-only imports:

```python
if TYPE_CHECKING:
    from tg4perfetto import CounterTrack, Group, NormalTrack
```

**Note:** `Tuple` should already be in the existing `from typing import ...` line. If not, add it.

### 2. State Fields (lines ~191-193)

Replace existing placeholder fields:

```python
# Queue tracing state
_pending_queue_trace_state: Dict[str, Tuple["NormalTrack", int, Priority]] = field(
    init=False, default_factory=dict
)  # cluster_id -> (track, start_ns, priority)
_queue_counter_tracks: Dict[str, "CounterTrack"] = field(
    init=False, default_factory=dict
)  # priority_key -> counter track
_queue_depth: Dict[str, int] = field(
    init=False, default_factory=dict
)  # priority_key -> depth (debug-only, may diverge from trace)
_queue_track_count: int = field(
    init=False, default=0
)  # Total slice tracks created (for cap enforcement)
```

**Constants:**
```python
_QUEUE_TRACK_CAP: int = 1000  # Max slice tracks before disabling queue slice creation
```

**Key Changes:**
- `_pending_queue_trace_state` stores the track handle, not just timestamps
- No more `_queue_slice_tracks` dict - each request has its own track
- `_queue_depth` is debug-only; real depth comes from bucket length
- `_queue_track_count` enforces cap to prevent unbounded memory growth

### 3. Helper Methods (after `_get_or_create_gpu_track`, ~line 294)

#### 3.1 `_get_or_create_queue_counter_track(priority)`

```python
def _get_or_create_queue_counter_track(self, priority: Priority) -> Optional["CounterTrack"]:
    """Get or create counter track for queue depth. Returns None on failure."""
    key = _PRIORITY_SHORT.get(priority, priority.name[:3])
    if key in self._queue_counter_tracks:
        return self._queue_counter_tracks[key]
    
    track = self._safe_trace_get(
        self._scheduler_group.create_counter_track,
        f"QDEPTH_{key}",
    )
    if track is not None:
        self._queue_counter_tracks[key] = track
    return track
```

#### 3.2 `_create_queue_slice_track(cluster_id, priority)` - NEW

**Creates a per-cluster track with cap enforcement. Returns None if cap exceeded.**

```python
# Module-level constant
_QUEUE_TRACK_CAP: int = 1000  # Max slice tracks before disabling queue slice creation

def _create_queue_slice_track(self, cluster_id: str, priority: Priority) -> Optional["NormalTrack"]:
    """Create a per-cluster slice track for queue visualization.
    
    Returns None on failure OR if track cap exceeded.
    When cap is exceeded, counter tracks still work but slice tracks are skipped.
    """
    # Check cap BEFORE creating
    if self._queue_track_count >= _QUEUE_TRACK_CAP:
        # Log once when cap first hit
        if self._queue_track_count == _QUEUE_TRACK_CAP:
            logging.getLogger(__name__).warning(
                f"Queue slice track cap ({_QUEUE_TRACK_CAP}) reached. "
                "Subsequent queue slices will not be traced (counters continue)."
            )
        self._queue_track_count += 1  # Increment to avoid repeated warnings
        return None
    
    key = _PRIORITY_SHORT.get(priority, priority.name[:3])
    track = self._safe_trace_get(
        self._scheduler_group.create_track,
        f"QUEUE_{key}_{cluster_id[:16]}",  # Truncate cluster_id to avoid long names
    )
    if track is not None:
        self._queue_track_count += 1
    return track
```

#### 3.3 `_trace_queue_enqueue(cluster_id, priority, lora_name)`

**Called AFTER `pending_bucket().append()`. Stores track handle AFTER successful open.**

```python
def _trace_queue_enqueue(self, cluster_id: str, priority: Priority, lora_name: Optional[str] = None) -> None:
    """Start queue slice and increment counter when request is enqueued.
    
    CRITICAL: Call AFTER pending_bucket().append() so len() reflects correct depth.
    CRITICAL: Track handle stored AFTER successful open to avoid orphan state.
    """
    if not self._enable_gpu_tracing or self._scheduler_group is None:
        return
    
    now_ns = time.time_ns()
    key = _PRIORITY_SHORT.get(priority, priority.name[:3])
    
    # Create per-cluster track
    slice_track = self._create_queue_slice_track(cluster_id, priority)
    
    # Start slice FIRST
    if slice_track:
        label = f"[{key}] {cluster_id}"
        if lora_name:
            safe_lora = lora_name.replace("|", "_").replace(" ", "_")[:32]
            label += f" | lora:{safe_lora}"
        ok = self._safe_trace(slice_track.open, now_ns, label)
        # CRITICAL: Only store state AFTER successful open
        if ok:
            self._pending_queue_trace_state[cluster_id] = (slice_track, now_ns, priority)
    
    # Counter: depth = current bucket size (AFTER append, so correct)
    counter_track = self._get_or_create_queue_counter_track(priority)
    if counter_track:
        depth = len(self._state.pending_bucket(priority))
        self._queue_depth[key] = depth  # Debug-only cache
        self._safe_trace(counter_track.count, now_ns, depth)
```

#### 3.4 `_trace_queue_slice_close(cluster_id)` - REVISED

**Always pops pending state, then direct close with error handling. No track creation.**

```python
def _trace_queue_slice_close(self, cluster_id: str) -> None:
    """Close queue slice when request is fulfilled.
    
    CRITICAL: 
    - Call BEFORE bucket.pop()
    - ALWAYS pops pending state (prevents leaks)
    - Uses stored track handle (no track creation on close)
    - Uses DIRECT close (not _safe_trace) to work even when tracing disabled
    
    Note: Direct close is safe here because we have a valid track handle from
    successful open. On unexpected errors, disables tracing (consistent with _safe_trace_call).
    """
    # ALWAYS pop entry first to prevent state leaks
    entry = self._pending_queue_trace_state.pop(cluster_id, None)
    if entry is None:
        return  # No pending trace state for this cluster
    
    stored_track, _, stored_priority = entry
    
    # Direct close - NOT via _safe_trace, so works even if tracing disabled
    now_ns = time.time_ns()
    if stored_track is not None:
        try:
            stored_track.close(now_ns)
        except (IOError, OSError):
            # I/O errors are expected - ignore
            pass
        except Exception as e:
            # Unexpected error - log and disable tracing (consistent with _safe_trace_call)
            logging.getLogger(__name__).warning(f"Queue trace close error, disabling tracing: {e}")
            self._enable_gpu_tracing = False
```

#### 3.5 `_trace_queue_counter_update(priority, depth)`

**Called AFTER `bucket.pop()` with explicit depth.**

```python
def _trace_queue_counter_update(self, priority: Priority, depth: int) -> None:
    """Update queue depth counter with explicit depth value.
    
    CRITICAL: 
    - Call AFTER bucket.pop() with len(bucket) as depth
    - Depth comes from real bucket length, not cache
    - Separating this from slice_close allows correct ordering
    """
    if not self._enable_gpu_tracing or self._scheduler_group is None:
        return
    
    now_ns = time.time_ns()
    key = _PRIORITY_SHORT.get(priority, priority.name[:3])
    
    counter_track = self._get_or_create_queue_counter_track(priority)
    if counter_track:
        self._queue_depth[key] = depth  # Debug-only cache
        self._safe_trace(counter_track.count, now_ns, depth)
```

#### 3.6 `_shutdown_close_queue_slices()` - NEW

**Close all queue slices WITHOUT gating on `_enable_gpu_tracing`.**

```python
def _shutdown_close_queue_slices(self) -> None:
    """Close all open queue slices during shutdown.
    
    CRITICAL: 
    - Does NOT gate on _enable_gpu_tracing (called after it's False)
    - Uses stored track handles directly
    - Called BEFORE _safe_final_flush()
    """
    if not self._pending_queue_trace_state:
        return
    
    now_ns = time.time_ns()
    for cluster_id, (track, _, _) in list(self._pending_queue_trace_state.items()):
        if track is not None:
            # Direct call, no gating - we're in shutdown
            try:
                track.close(now_ns)
            except Exception:
                pass  # Ignore errors during shutdown
    self._pending_queue_trace_state.clear()
```

### 4. Integration Points

#### 4.1 `request_gpus()` (~line 820)

**Order: append THEN trace** (counter depth is correct):

```python
self._state.pending_bucket(priority).append(pending)
self._trace_queue_enqueue(cluster_id, priority, lora_name)  # AFTER append
self._wakeup_event.set()
```

#### 4.2 `release_and_request_gpus()` (~line 903)

Same order:

```python
self._state.pending_bucket(request_priority).append(pending)
self._trace_queue_enqueue(request_cluster_id, request_priority, request_lora_name)  # AFTER append
self._wakeup_event.set()
```

#### 4.3 `_signal_pending_request()` (~line 1804) - CRITICAL FIX

**Order: close slice → pop → update counter**

```python
def _signal_pending_request(self, *, cluster_id: str, priority: Priority, result: Optional[List[int]] = None) -> None:
    bucket = self._state.pending_bucket(priority)
    for idx, pending in enumerate(bucket):
        if pending.request.cluster_id != cluster_id:
            continue
        # 1. Close slice BEFORE pop (track from stored state)
        self._trace_queue_slice_close(cluster_id)
        # 2. Pop from bucket
        bucket.pop(idx)
        # 3. Update counter AFTER pop with correct depth
        self._trace_queue_counter_update(priority, len(bucket))
        pending.result = list(result or [])
        pending.event.set()
        return
    raise RuntimeError(f"No pending request found for cluster_id={cluster_id!r} priority={priority!r}")
```

#### 4.4 `_signal_all_waiters_with_error()` (~line 916) - CRITICAL FIX

**Close slices, then single counter emit per priority:**

```python
def _signal_all_waiters_with_error(self, *, error: str) -> None:
    # Close all queue slices (track from stored state)
    for priority in Priority:
        for pending in list(self._state.pending_bucket(priority)):
            self._trace_queue_slice_close(pending.request.cluster_id)
            pending.error = error
            pending.event.set()
        self._state.pending_bucket(priority).clear()
        # Single counter update to 0 after clear
        self._trace_queue_counter_update(priority, 0)
    # ... rest unchanged (pending_planned_release_requests handling)
```

#### 4.5 `unregister_pipeline()` (~line 537) - CRITICAL FIXES

**Fix both queue tracing AND pipeline matching bug.**

**Bug:** Current code uses `startswith(f"{pipeline_id}_")` which can match wrong pipeline if pipeline_id contains underscores (e.g., "p_1" matches "p_1_x" AND "p_1_2_x").

**Fix:** Use exact `pipeline_id` match via `parse_cluster_id()`. **Two-phase atomic design:**
1. **Validate phase**: Parse ALL cluster_ids first, raise immediately if any malformed
2. **Mutation phase**: Perform all removals (non-throwing)

This prevents partial state corruption on malformed IDs.

**Note:** `parse_cluster_id(cluster_id)` returns `tuple[str, str]` `(pipeline_id, cluster_name)`, not an object.

```python
async def unregister_pipeline(self, *, pipeline_id: str) -> None:
    validate_pipeline_id(pipeline_id)
    async with self._lock:
        # ============================================================
        # PHASE 1: VALIDATE - Parse all cluster_ids, fail-fast if any malformed
        # ============================================================
        allocations_to_remove: List[str] = []
        for cluster_id in self._state.active_allocations:
            try:
                parsed_pipeline_id, _ = parse_cluster_id(cluster_id)
            except ValueError as e:
                # CRITICAL: Malformed cluster_id indicates system corruption
                # Signal all waiters and trigger global fail-fast shutdown
                error_msg = f"Malformed cluster_id in active_allocations: {cluster_id!r}"
                self._signal_all_waiters_with_error(error=error_msg)
                await self._fail_fast_shutdown(reason=f"unregister_pipeline_invalid_cluster_id: {cluster_id!r}")
                # RAISE after shutdown attempt - caller must know this failed
                # (shutdown is best-effort, may have failed silently)
                raise RuntimeError(error_msg) from e
            if parsed_pipeline_id == pipeline_id:
                allocations_to_remove.append(cluster_id)
        
        pending_to_remove: Dict[Priority, List[PendingRequest]] = {}
        for priority in Priority:
            pending_to_remove[priority] = []
            for pending in self._state.pending_bucket(priority):
                try:
                    parsed_pipeline_id, _ = parse_cluster_id(pending.request.cluster_id)
                except ValueError as e:
                    error_msg = f"Malformed cluster_id in pending bucket: {pending.request.cluster_id!r}"
                    self._signal_all_waiters_with_error(error=error_msg)
                    await self._fail_fast_shutdown(reason=f"unregister_pipeline_invalid_cluster_id: {pending.request.cluster_id!r}")
                    raise RuntimeError(error_msg) from e
                if parsed_pipeline_id == pipeline_id:
                    pending_to_remove[priority].append(pending)
        
        planned_releases_to_remove: List[str] = []
        for cluster_id in self._state.pending_planned_release_requests:
            try:
                parsed_pipeline_id, _ = parse_cluster_id(cluster_id)
            except ValueError as e:
                error_msg = f"Malformed cluster_id in pending_planned_release_requests: {cluster_id!r}"
                self._signal_all_waiters_with_error(error=error_msg)
                await self._fail_fast_shutdown(reason=f"unregister_pipeline_invalid_cluster_id: {cluster_id!r}")
                raise RuntimeError(error_msg) from e
            if parsed_pipeline_id == pipeline_id:
                planned_releases_to_remove.append(cluster_id)
        
        # ============================================================
        # PHASE 2: MUTATE - Non-throwing operations only
        # ============================================================
        self._state.pipeline_registry.pop(pipeline_id, None)
        self._state.latest_progress_by_pipeline.pop(pipeline_id, None)
        self._adapter_handle_cache.pop(pipeline_id, None)
        
        # Remove allocations
        for cluster_id in allocations_to_remove:
            alloc = self._state.active_allocations.pop(cluster_id, None)
            if alloc is not None:
                self._end_traces_for_gpu_ids(alloc.gpu_ids)
                self._state.idle_gpus |= set(alloc.gpu_ids)
        
        # Remove pending requests and close queue slices
        affected_priorities: set[Priority] = set()
        for priority, pendings in pending_to_remove.items():
            bucket = self._state.pending_bucket(priority)
            for pending in pendings:
                self._trace_queue_slice_close(pending.request.cluster_id)
                pending.error = f"Pipeline {pipeline_id!r} unregistered"
                pending.event.set()
                if pending in bucket:
                    bucket.remove(pending)
                affected_priorities.add(priority)
            # Update counter for affected priorities
            if priority in affected_priorities:
                self._trace_queue_counter_update(priority, len(bucket))
        
        # Remove planned releases
        for cluster_id in planned_releases_to_remove:
            req = self._state.pending_planned_release_requests.pop(cluster_id, None)
            if req is not None:
                req.error = f"Pipeline {pipeline_id!r} unregistered"
                req.event.set()
        
        self._wakeup_event.set()
```

**Note:** This fixes the pipeline matching bug in existing code, not just for queue tracing. Fail-fast errors now trigger global shutdown (consistent with scheduler loop error handling).

### 5. Cleanup in `_shutdown_tracing()` (~line 390) - CRITICAL FIX

**Order: Close slices BEFORE disabling tracing:**

```python
def _shutdown_tracing(self) -> None:
    if self._trace_shutdown_started:
        return
    self._trace_shutdown_started = True
    
    if self._trace_gen is None:
        return
    
    # Step 1: Close all open queue slices FIRST (before disabling tracing)
    # This uses stored track handles, works even if we proceed to disable
    self._shutdown_close_queue_slices()
    
    # Step 2: Disable tracing to stop new calls
    self._enable_gpu_tracing = False
    
    # Step 3: Clear all trace state
    self._gpu_tracks.clear()
    self._gpu_contexts.clear()
    self._queue_counter_tracks.clear()
    self._queue_depth.clear()
    self._scheduler_group = None
    
    # Step 4: Final flush
    self._safe_final_flush()
    
    self._trace_gen = None
    try:
        atexit.unregister(self._shutdown_tracing)
    except Exception:
        pass
```

---

## Track Naming Convention

| Priority | Counter Track | Slice Track (per-cluster) |
|----------|---------------|---------------------------|
| INITIALIZATION | `QDEPTH_INIT` | `QUEUE_INIT_{cluster_id}` |
| ACTOR_TRAINING | `QDEPTH_TRN` | `QUEUE_TRN_{cluster_id}` |
| CRITIC_TRAINING | `QDEPTH_CRT` | `QUEUE_CRT_{cluster_id}` |
| OLD_LOG_PROBS | `QDEPTH_OLD` | `QUEUE_OLD_{cluster_id}` |
| REF_LOG_PROBS | `QDEPTH_REF` | `QUEUE_REF_{cluster_id}` |
| VALUE_COMPUTE | `QDEPTH_VAL` | `QUEUE_VAL_{cluster_id}` |
| GENERATION | `QDEPTH_GEN` | `QUEUE_GEN_{cluster_id}` |

---

## Existing Placeholder Fields

The scheduler already has placeholder fields that need updating:

```python
# Current (line ~191):
_queue_tracks: Dict[str, "NormalTrack"] = field(init=False, default_factory=dict)
_pending_request_start_ns: Dict[str, int] = field(init=False, default_factory=dict)

# Should become:
_pending_queue_trace_state: Dict[str, Tuple["NormalTrack", int, Priority]] = field(
    init=False, default_factory=dict
)  # cluster_id -> (track, start_ns, priority)
_queue_counter_tracks: Dict[str, "CounterTrack"] = field(
    init=False, default_factory=dict
)  # priority_key -> counter track
_queue_depth: Dict[str, int] = field(
    init=False, default_factory=dict
)  # priority_key -> depth (debug-only)
```

---

## Critical Ordering Summary

| Operation | Method | Timing | Reason |
|-----------|--------|--------|--------|
| Enqueue slice open | `_trace_queue_enqueue()` | AFTER `append()` | Counter needs correct depth |
| Enqueue state store | `_trace_queue_enqueue()` | AFTER successful `open()` | Avoid orphan state |
| Fulfill slice close | `_trace_queue_slice_close()` | BEFORE `pop()` | Slice must close while request exists |
| Fulfill state pop | `_trace_queue_slice_close()` | ALWAYS (even if tracing disabled) | Prevent state leaks |
| Fulfill counter update | `_trace_queue_counter_update()` | AFTER `pop()` | `len(bucket)` is new depth |
| Error path counter | `_trace_queue_counter_update(priority, 0)` | AFTER `clear()` | Single emit, not per-request |
| Unregister pipeline | `_trace_queue_slice_close()` + `_trace_queue_counter_update()` | Before/after clear | Avoid orphaned slices |
| Shutdown slices | `_shutdown_close_queue_slices()` | BEFORE `_enable_gpu_tracing = False` | Close must work |

---

## Import Verification

`CounterTrack` exists in `tg4perfetto/_tgen.py`:
```python
class CounterTrack:
    def count(self, ts, value):
        """Add a count value on the track."""
        self._parent._track_count(self._uuid, ts, value)
```

`NormalTrack.close()` is LIFO:
```python
def close(self, ts, flow = []):
    """ Close a track.  The last 'open' call is closed """
    self._parent._track_close(self._uuid, ts, flow)
```

---

## Bug Fixes Summary

### P0 Critical (Fixed in Plan)

| Issue | Fix |
|-------|-----|
| LIFO track model | Per-cluster slice tracks + store track handle |
| Shutdown close gating | Close slices before `_enable_gpu_tracing = False` |

### P1 Serious (Fixed in Plan)

| Issue | Fix |
|-------|-----|
| Pending map leak | Always pop entry first in `_trace_queue_slice_close` |
| State stored before successful open | Store after `open()` returns success |
| Queue depth cache diverges | Treat `_queue_depth` as debug-only, emit from real bucket |
| Close creates phantom tracks | Use stored track handle, never create on close |
| Latency risk (trace in lock) | Keep tracing append-only; tg4perfetto flush is threshold-based, not per-call |

### P1 Code Bug (Documented for Execution)

| Issue | Fix |
|-------|-----|
| Pipeline matching prefix-based | Use `parse_cluster_id()` for exact match |

---

## Verification

### Manual Test

1. Run scheduler with tracing enabled:
   ```bash
   SCHEDRL_ENABLE_GPU_TRACING=1 python your_script.py
   ```

2. Trigger enqueue/fulfill sequence with out-of-order completion

3. Open trace in [Perfetto UI](https://ui.perfetto.dev/)

4. Verify:
   - Counter shows: `1 → 2 → 3 → 2 → 1 → 0`
   - Each slice has correct duration
   - Out-of-order completion doesn't corrupt slices

### Unit Test Scenarios

**Note:** 
- Tests must manually create pending requests instead of calling `request_gpus()` which blocks on `event.wait()`.
- `initialize()` requires a Ray actor-style resource manager with `.remote()` methods that return awaitables.
- `initialize()` starts a background `_central_scheduling_loop` task that must be cancelled in teardown.
- `parse_cluster_id` requires known suffixes: `_actor_train`, `_actor_infer`, `_critic`, `_reference`.
- Tracing may be disabled if `tg4perfetto` is unavailable; tests should assert or skip.

```python
import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock
import pytest


def make_mock_resource_manager(num_gpus: int = 8, gpus_per_node: int = 8) -> MagicMock:
    """Create a mock resource manager that works with initialize().
    
    initialize() calls: await self._resource_manager.get_required_gpus_per_node.remote()
                       await self._resource_manager.get_num_gpus.remote()
    """
    rm = MagicMock()
    # .remote() must return an awaitable
    rm.get_required_gpus_per_node.remote = AsyncMock(return_value=gpus_per_node)
    rm.get_num_gpus.remote = AsyncMock(return_value=num_gpus)
    return rm


@asynccontextmanager
async def scheduler_context(enable_tracing: bool = True):
    """Context manager for scheduler lifecycle in tests.
    
    Ensures proper teardown (cancel loop, shutdown) even on skip or exception.
    """
    scheduler = SchedulerImpl()
    scheduler._resource_manager = make_mock_resource_manager()
    await scheduler.initialize(enable_gpu_tracing=enable_tracing)
    try:
        yield scheduler
    finally:
        # Cancel background loop
        if scheduler._loop_task is not None:
            scheduler._loop_task.cancel()
            try:
                await scheduler._loop_task
            except asyncio.CancelledError:
                pass
        # Full shutdown for proper atexit cleanup
        try:
            await scheduler.shutdown()
        except Exception:
            pass


async def test_queue_slice_track_per_cluster():
    """Verify each request gets its own slice track (not per-priority)."""
    async with scheduler_context(enable_tracing=True) as scheduler:
        # Skip if tracing not available (tg4perfetto not installed)
        if not scheduler._enable_gpu_tracing:
            pytest.skip("tg4perfetto not available")
        
        # Manually create pending requests with valid cluster_id suffixes
        async with scheduler._lock:
            for i, cid in enumerate(["p1_actor_train", "p2_actor_train", "p3_actor_train"]):
                event = asyncio.Event()
                pending = PendingRequest(
                    request=Request(cluster_id=cid, priority=Priority.ACTOR_TRAINING, timestamp=float(i)),
                    event=event,
                )
                scheduler._state.pending_bucket(Priority.ACTOR_TRAINING).append(pending)
                scheduler._trace_queue_enqueue(cid, Priority.ACTOR_TRAINING)
        
        # Verify each cluster has its own track
        assert len(scheduler._pending_queue_trace_state) == 3
        tracks = [entry[0] for entry in scheduler._pending_queue_trace_state.values()]
        assert len(set(id(t) for t in tracks)) == 3  # All different track objects


async def test_out_of_order_close():
    """Verify out-of-order close doesn't corrupt slices."""
    async with scheduler_context(enable_tracing=True) as scheduler:
        if not scheduler._enable_gpu_tracing:
            pytest.skip("tg4perfetto not available")
        
        # Create requests with valid cluster_id suffixes
        async with scheduler._lock:
            for i, cid in enumerate(["p1_actor_train", "p2_actor_train", "p3_actor_train"]):
                pending = PendingRequest(
                    request=Request(cluster_id=cid, priority=Priority.ACTOR_TRAINING, timestamp=float(i)),
                    event=asyncio.Event(),
                )
                scheduler._state.pending_bucket(Priority.ACTOR_TRAINING).append(pending)
                scheduler._trace_queue_enqueue(cid, Priority.ACTOR_TRAINING)
        
        # Close in different order: p2 first, then p1, then p3
        async with scheduler._lock:
            bucket = scheduler._state.pending_bucket(Priority.ACTOR_TRAINING)
            
            # Close p2_actor_train (middle one) - MUTATE actual bucket via slice assignment
            scheduler._trace_queue_slice_close("p2_actor_train")
            bucket[:] = [p for p in bucket if p.request.cluster_id != "p2_actor_train"]
            scheduler._trace_queue_counter_update(Priority.ACTOR_TRAINING, len(bucket))
            
            # Close p1_actor_train (first one)
            scheduler._trace_queue_slice_close("p1_actor_train")
            bucket[:] = [p for p in bucket if p.request.cluster_id != "p1_actor_train"]
            scheduler._trace_queue_counter_update(Priority.ACTOR_TRAINING, len(bucket))
            
            # Close p3_actor_train (last one)
            scheduler._trace_queue_slice_close("p3_actor_train")
            bucket[:] = [p for p in bucket if p.request.cluster_id != "p3_actor_train"]
            scheduler._trace_queue_counter_update(Priority.ACTOR_TRAINING, len(bucket))
        
        # Verify all state cleaned up
        assert len(scheduler._pending_queue_trace_state) == 0
        assert scheduler._queue_depth.get("train") == 0


async def test_state_cleanup_when_tracing_disabled():
    """Verify state is cleaned up even if tracing is disabled after enqueue."""
    async with scheduler_context(enable_tracing=True) as scheduler:
        if not scheduler._enable_gpu_tracing:
            pytest.skip("tg4perfetto not available")
        
        # Create request with valid cluster_id suffix
        async with scheduler._lock:
            pending = PendingRequest(
                request=Request(cluster_id="p1_actor_train", priority=Priority.ACTOR_TRAINING, timestamp=1.0),
                event=asyncio.Event(),
            )
            scheduler._state.pending_bucket(Priority.ACTOR_TRAINING).append(pending)
            scheduler._trace_queue_enqueue("p1_actor_train", Priority.ACTOR_TRAINING)
        
        # Disable tracing
        scheduler._enable_gpu_tracing = False
        
        # Close should still clean up state (direct close, not gated)
        scheduler._trace_queue_slice_close("p1_actor_train")
        
        # Verify state was popped even though tracing was disabled
        assert "p1_actor_train" not in scheduler._pending_queue_trace_state
```

---

## Considerations

1. **Single pending per cluster**: Current code prevents duplicate pending requests per cluster_id, so slice keying by cluster_id is safe

2. **Priority-level aggregation**: Counter tracks are per-priority (not per-cluster) to show overall queue pressure

3. **Reuse existing helpers**: `_safe_trace`, `_safe_trace_get` pattern works for queue tracing

4. **Graceful degradation**: Same fail-safe pattern as GPU tracing (disable on unexpected error, don't crash scheduler)

5. **Track count scalability (IMPLEMENTED)**: 
   - Per-cluster slice tracks means each enqueued request creates a new `NormalTrack`.
   - In tg4perfetto, each track writes a descriptor packet (`_tid_packet`) that is **not reclaimed** during trace generation.
   - Track count grows with **total queued requests per run**, not just current pending.
   - **Solution implemented**: Hard cap (`_QUEUE_TRACK_CAP = 1000`) on slice tracks. When exceeded:
     - Counter tracks continue working (queue depth visualization preserved)
     - Slice tracks are skipped (individual wait time not traced for excess requests)
     - Single warning logged when cap first reached
     - **NOT falling back to per-priority slices** (that would reintroduce LIFO corruption)

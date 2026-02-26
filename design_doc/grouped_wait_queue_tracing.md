# Queue Tracing Sub-Groups Implementation Plan

## Context

Queue tracing is fully implemented with per-cluster slice tracks, counters, and cap enforcement.
This change reorganizes the Perfetto UI layout by grouping queue tracks per-priority into collapsible
sub-groups, making it easier to navigate when many pipelines are running.

**Current flat structure:**
```
SCHEDULER (Group)
├── QDEPTH_TRN (CounterTrack)
├── QDEPTH_GEN (CounterTrack)
├── QUEUE_TRN_p1_actor_train (NormalTrack)
└── QUEUE_GEN_p1_actor_infer (NormalTrack)
```

**Target sub-group structure:**
```
SCHEDULER (Group)
├── Queue_TRN (sub-group)
│   ├── depth (CounterTrack)
│   └── p1_actor_train (NormalTrack)
└── Queue_GEN (sub-group)
    ├── depth (CounterTrack)
    └── p1_actor_infer (NormalTrack)
```

## Design Decision: Wrapper Instead of Modifying tg4perfetto

`tg4perfetto.GroupTrack` (returned by `Group.create_group()`) only has `create_track(self)` with no
name param and no `create_counter_track`. We need named creation.

We do NOT modify tg4perfetto. Instead we add `_QueueSubGroup` to `scheduler.py` — a thin wrapper
that holds `GroupTrack`'s internal handles and calls `_create_track` directly (the same private
method that `Group.create_track()` uses internally).

This eliminates the capability-check problem entirely: we control all method bodies so there is no
AttributeError risk from a missing method on `GroupTrack`.

---

## Change 1: Add _QueueSubGroup dataclass to scheduler.py

**File**: `schedrl/scheduler/scheduler.py`

**Location**: After `_GPUAllocTraceContext` (~line 152), following same `@dataclass(slots=True)` pattern.

```python
@dataclass(slots=True)
class _QueueSubGroup:
    """Named-track factory wrapping a tg4perfetto GroupTrack sub-group.

    GroupTrack.create_track() creates tracks named after the group, not a caller-supplied name.
    This wrapper holds GroupTrack's internal uuid and parent handles and calls _create_track
    directly (same private method Group.create_track uses) to pass an explicit name.

    Source pattern: _GPUAllocTraceContext in scheduler.py
    """

    # GroupTrack internal handles — accessed via gt._uuid and gt._parent after create_group()
    _uuid: int
    _parent: Any  # tg4perfetto.TraceGenerator at runtime; Any to avoid runtime import

    def create_track(self, track_name: str) -> "NormalTrack":
        """Create a named slice track under this sub-group."""
        return self._parent._create_track(self._uuid, track_name, 0)

    def create_counter_track(self, track_name: str) -> "CounterTrack":
        """Create a named counter track under this sub-group."""
        return self._parent._create_track(self._uuid, track_name, 1)

    @classmethod
    def from_group_track(cls, gt: "GroupTrack") -> "_QueueSubGroup":
        """Extract handles from a freshly created GroupTrack."""
        return cls(gt._uuid, gt._parent)
```

---

## Change 2: TYPE_CHECKING import for GroupTrack

**File**: `schedrl/scheduler/scheduler.py` (~line 49-51)

`GroupTrack` is not exported from `tg4perfetto.__init__`, so import from the internal module.
Both imports must stay inside the `if TYPE_CHECKING:` block:

```python
# Before:
if TYPE_CHECKING:
    from tg4perfetto import CounterTrack, Group, NormalTrack

# After:
if TYPE_CHECKING:
    from tg4perfetto import CounterTrack, Group, NormalTrack
    from tg4perfetto._tgen import GroupTrack  # GroupTrack not re-exported in __init__
```

---

## Change 3: Add _queue_groups field

**File**: `schedrl/scheduler/scheduler.py` (~line 207, after existing queue state fields)

```python
# Maps priority short-key (e.g. "TRN") to its Perfetto queue sub-group wrapper
_queue_groups: Dict[str, _QueueSubGroup] = field(init=False, default_factory=dict)
```

Note: no string-quotes needed — `_QueueSubGroup` is defined in the same file above the dataclass.

---

## Change 4: Add _get_or_create_queue_group() helper

**File**: `schedrl/scheduler/scheduler.py` (~line 310, after `_get_or_create_gpu_track`)

Copy-then-revise from `_get_or_create_gpu_track` (lines 277-293):

```python
def _get_or_create_queue_group(self, priority: Priority) -> Optional[_QueueSubGroup]:
    """Get or create the Queue_<KEY> sub-group wrapper for a priority tier.

    Returns None if tracing is disabled or scheduler group is not initialized.
    """
    if not self._enable_gpu_tracing or self._scheduler_group is None:
        return None
    key = _PRIORITY_SHORT.get(priority, priority.name[:3])
    if key in self._queue_groups:
        return self._queue_groups[key]
    # _safe_trace_get wraps the attribute lookup + call together here because
    # create_group is a method on Group (guaranteed to exist).
    raw_group = self._safe_trace_get(
        self._scheduler_group.create_group,
        f"Queue_{key}",
    )
    if raw_group is None:
        logging.getLogger(__name__).debug(
            f"Failed to create queue sub-group for priority {key}"
        )
        return None
    sub_group = _QueueSubGroup.from_group_track(raw_group)
    self._queue_groups[key] = sub_group
    return sub_group
```

---

## Change 5: Update _get_or_create_queue_counter_track()

**File**: `schedrl/scheduler/scheduler.py` (~line 298)

Create counter inside the sub-group, named `"depth"` instead of `f"QDEPTH_{key}"`:

```python
# Before:
if key in self._queue_counter_tracks:
    return self._queue_counter_tracks[key]
track = self._safe_trace_get(
    self._scheduler_group.create_counter_track,
    f"QDEPTH_{key}",
)

# After:
if key in self._queue_counter_tracks:
    return self._queue_counter_tracks[key]
queue_group = self._get_or_create_queue_group(priority)
if queue_group is None:
    return None
# create_counter_track is a method on _QueueSubGroup (our wrapper) — always exists
track = self._safe_trace_get(
    queue_group.create_counter_track,
    "depth",
)
```

---

## Change 6: Update _create_queue_slice_track()

**File**: `schedrl/scheduler/scheduler.py` (~line 312)

Create slice inside the sub-group, named `cluster_id[:16]` instead of `f"QUEUE_{key}_{cluster_id[:16]}"`:

```python
# Before:
track = self._safe_trace_get(
    self._scheduler_group.create_track,
    f"QUEUE_{key}_{cluster_id[:16]}",
)

# After:
queue_group = self._get_or_create_queue_group(priority)
if queue_group is None:
    return None
# create_track is a method on _QueueSubGroup (our wrapper) — always exists
track = self._safe_trace_get(
    queue_group.create_track,
    cluster_id[:16],
)
```

---

## Change 7: Update _shutdown_tracing()

**File**: `schedrl/scheduler/scheduler.py` (~line 575)

Add `_queue_groups.clear()` before `_scheduler_group = None`:

```python
self._gpu_tracks.clear()
self._gpu_contexts.clear()
self._queue_counter_tracks.clear()
self._queue_depth.clear()
self._queue_groups.clear()   # ADD: wrapper refs hold _parent (TraceGenerator) — must clear
self._scheduler_group = None
```

---

## Change 8: Add schedrl/requirements.txt

**File**: `schedrl/requirements.txt` (new file, user-approved)

```
# Runtime dependencies for the schedrl package
# tg4perfetto: GPU/scheduler timeline tracing via Perfetto UI
# Requires >=0.0.6 for Group.create_group() support (used by queue sub-group tracing)
tg4perfetto>=0.0.6
```

---

## Change 9: Add duplicate-enqueue assertion in _trace_queue_enqueue

**File**: `schedrl/scheduler/scheduler.py` (~line 425)

Before storing pending state, assert cluster_id is not already tracked (Fail Fast — detects scheduler bug):

```python
# Before:
if ok:
    self._pending_queue_trace_state[cluster_id] = (slice_track, now_ns, priority)

# After:
if ok:
    # Fail fast: duplicate cluster_id means scheduler violated one-request-per-cluster invariant
    assert cluster_id not in self._pending_queue_trace_state, (
        f"Duplicate queue trace enqueue for cluster_id={cluster_id!r}"
    )
    self._pending_queue_trace_state[cluster_id] = (slice_track, now_ns, priority)
```

---

## Critical Files

- `schedrl/requirements.txt` (new file)
- `schedrl/scheduler/scheduler.py` only — tg4perfetto is NOT modified:
  - TYPE_CHECKING imports (~line 50)
  - `_QueueSubGroup` dataclass (~line 155, after `_GPUAllocTraceContext`)
  - Field declarations (~line 207)
  - `_get_or_create_queue_group()` new method (~line 310)
  - `_get_or_create_queue_counter_track()` (~line 298)
  - `_create_queue_slice_track()` (~line 312)
  - `_shutdown_tracing()` (~line 575)

## Reuse Patterns

- `_QueueSubGroup` dataclass: copies `_GPUAllocTraceContext` style (`@dataclass(slots=True)`)
- `_get_or_create_queue_group()`: copies `_get_or_create_gpu_track()` structure (lines 277-293):
  guard check → key lookup in cache → `_safe_trace_get` → cache on success → return

---

## Prerequisites

**Verify `Group.create_group()` is available** (new API dependency — scheduler doesn't currently use it):
```bash
python -c "from tg4perfetto._tgen import Group; assert hasattr(Group, 'create_group'), 'Group.create_group missing — use submodule version'"
```
If this fails, install from the submodule: `pip install -e external/tg4perfetto`

---

## Verification

**Type check** (must pass clean):
```bash
cd schedrl && mypy --strict scheduler/scheduler.py
```

**Runtime smoke** — inline enqueue→dequeue→shutdown cycle (no new test file):
```bash
cd schedrl && python -c "
import tempfile, os, asyncio
from schedrl.scheduler.scheduler import Scheduler
from schedrl.protocol.types import Priority

with tempfile.NamedTemporaryFile(suffix='.pftrace', delete=False) as f:
    path = f.name

async def smoke():
    # instantiate with minimal required args — adapt as needed
    s = Scheduler(...)
    s.enable_tracing(path)
    s._trace_queue_enqueue('test-cluster', Priority.GENERATION)
    s._trace_queue_slice_close('test-cluster')
    s._shutdown_tracing()
    assert not s._pending_queue_trace_state, 'pending state leaked on shutdown'
    assert os.path.exists(path), 'trace file not written'
    os.unlink(path)
    print('smoke: OK')

asyncio.run(smoke())
"
```

**Perfetto UI check** — run with tracing enabled, open `.pftrace` in ui.perfetto.dev:
- `Queue_TRN`, `Queue_GEN`, etc. appear as collapsible sub-groups under `SCHEDULER`
- `depth` counter is nested inside each sub-group (NOT at top level as `QDEPTH_TRN`)
- Per-cluster slices show as `p1_actor_train` (NOT prefixed `QUEUE_TRN_p1_actor_train`)
- Old flat `QDEPTH_*` and `QUEUE_*` tracks do NOT appear

**Commit message must document track name changes**:
Track names change in this commit: `QDEPTH_TRN` → `depth` (inside `Queue_TRN`), `QUEUE_TRN_p1_actor_train` → `p1_actor_train` (inside `Queue_TRN`). Note this in the commit body for anyone who has saved trace bookmarks by name.

**Note on sub-group rendering**: `GroupTrack` (ttype=2) produces the same Perfetto wire format as
`NormalTrack` (ttype=0) — both are plain `track_descriptor` packets with `parent_uuid`. Collapsibility
in Perfetto UI comes from the parent-child uuid relationship: any track whose `uuid` is referenced as
`parent_uuid` by other tracks is rendered as a collapsible group. README.md:162-169 demonstrates this
pattern with a screenshot. No special Perfetto protobuf field is needed.

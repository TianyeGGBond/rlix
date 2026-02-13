# ENG-123 Phase 3 Code Review Round 2: Fresh Attack Angles

**Date**: 2026-02-13
**Reviewer**: Architect Mode
**Scope**: Phase 3 implementation review from angles NOT covered in previous review (2026-02-12-ENG-123-phase3-code-review.md)

## Executive Summary

This review attacks the codebase from **10 fresh angles** that were NOT covered in the previous 87-bug review:

1. **Queue/Backpressure behavior** - LoadBalancer.Lease `__del__` crash, ReplayBuffer edge cases
2. **Worker initialization ordering** - Startup/shutdown sequence dependencies
3. **Checkpoint/recovery paths** - Weight sync failures, corruption handling
4. **Distributed coordination** - Multi-node sync, node failure scenarios
5. **Memory pressure handling** - OOM scenarios, fragmentation during offload/load
6. **Timeout cascades** - Nested timeouts, propagation across components
7. **Event loop blocking** - Sync operations in async context
8. **Ray actor lifecycle** - Death handling, restart semantics
9. **Placement group lifecycle** - Creation, destruction, leak detection
10. **Data plane correctness** - Trajectory integrity, buffer consistency

**Total New Findings**: **15 P0 (Critical) + 8 P1 (High) + 3 P2 (Medium) = 26 new bugs**

---

## P0 Bugs (Critical)

### P0-R2-01: LoadBalancer.Lease `__del__` Assertion Crashes GC Thread

**Severity**: CRITICAL - Process crash
**Location**: [`generate_scheduler.py:104-106`](third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py:104)

**Status**: ✅ **RESOLVED** (2026-02-13) — replaced `assert` in `Lease.__del__` with a loud stderr error message (no exception raised from `__del__`).

**Problem**:
```python
class Lease:
    def __del__(self):
        # User must call clear or consume all lease to give back credit explicitly.
        assert self.lease == 0  # CRASHES GC IF NOT CLEARED!
```

**Impact**:
- If a `Lease` object is garbage collected without calling `clear()` or consuming all credit, the `__del__` assertion fails
- This crashes the Python garbage collector thread
- Can happen if:
  - Exception occurs before `clear()` is called
  - Context manager not properly exited
  - Object deleted while still holding credit
  - Async task cancelled while holding lease

**Root Cause**: Using `assert` in `__del__` is dangerous because:
1. Assertions can be disabled with `-O` flag
2. Exceptions in `__del__` are printed to stderr but don't propagate
3. Failed assertions in `__del__` can corrupt GC state

**Fix Required**:
```python
def __del__(self):
    if self.lease != 0:
        import warnings
        warnings.warn(
            f"Lease garbage collected with {self.lease} credits still held. "
            f"Call clear() or use context manager properly.",
            ResourceWarning
        )
        # Force release to prevent credit leak
        self.load_balancer._release(self._dp_rank, credit=self.lease)
```

**Extraction Plan Reference**: Not explicitly covered, but violates fail-fast principle.

---

### P0-R2-02: LoadBalancer.acquire() Has No Timeout, Can Block Forever

**Severity**: CRITICAL - Indefinite hang
**Location**: [`generate_scheduler.py:150-172`](third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py:150)

**Status**: ❌ **INVALID / Over-scoped for Phase 3** (2026-02-13) — Phase 3 assumes happy-path execution and uses blocking waits as backpressure. Adding timeouts/retries is out of scope; the system should fail-fast on real errors (not time out healthy waits).

**Problem**:
```python
async def acquire(self, credit: int) -> Lease:
    while True:  # NO TIMEOUT!
        while self._suspend:
            self.suspend_event.clear()
            await self.suspend_event.wait()  # NO TIMEOUT!

        target = -1
        for dp_rank, running_requests in self.workers.items():
            if running_requests >= self.max_running_requests:
                continue
            if target == -1 or running_requests < self.workers[target]:
                target = dp_rank
        if target != -1:
            self.workers[target] += credit
            self.running_request += credit
            return self.Lease(self, lease=credit, dp_rank=target)
        self.acquire_event.clear()
        await self.acquire_event.wait()  # NO TIMEOUT!
```

**Impact**:
- If all workers are at `max_running_requests`, the loop blocks forever
- If `_suspend` is True and never resumed, blocks forever
- No way to detect or recover from stuck state
- Blocks entire training pipeline

**Fix Required**:
```python
async def acquire(self, credit: int, timeout: Optional[float] = None) -> Lease:
    start_time = time.time() if timeout else None
    
    while True:
        if timeout:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise asyncio.TimeoutError(f"Could not acquire lease within {timeout}s")
            remaining = timeout - elapsed
        else:
            remaining = None
        
        # ... existing logic ...
        
        if target != -1:
            return self.Lease(...)
        
        try:
            await asyncio.wait_for(self.acquire_event.wait(), timeout=remaining)
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(f"Could not acquire lease within {timeout}s")
```

---

### P0-R2-03: LoadBalancer.acquire() Race Condition in Worker Selection

**Severity**: CRITICAL - Credit accounting corruption
**Location**: [`generate_scheduler.py:160-170`](third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py:160)

**Status**: ❌ **INVALID** (2026-02-13) — this selection+increment path has no `await` between choosing `target` and updating counters, so within the single-threaded event loop it is atomic. The `FIXME` is about logical oversubscription vs `max_running_requests` (a separate policy choice), not an async race.

**Problem**:
```python
# Line 160-165: Find least-loaded worker
target = -1
for dp_rank, running_requests in self.workers.items():
    if running_requests >= self.max_running_requests:
        continue
    if target == -1 or running_requests < self.workers[target]:
        target = dp_rank

# Lines 166-170: Update state - BUT ANOTHER COROUTINE MAY HAVE TAKEN THIS SLOT!
if target != -1:
    # FIXME may send more than max_running_requests (i.e. workers[target] + credit > max_running_requests)
    self.workers[target] += credit  # RACE: Another coroutine may have already incremented!
    self.running_request += credit
    return self.Lease(self, lease=credit, dp_rank=target)
```

**Impact**:
- The `FIXME` comment acknowledges the bug: can exceed `max_running_requests`
- Two coroutines can both select the same `target` and both increment
- Results in `workers[target] > max_running_requests`
- Worker overload, request queueing, potential OOM

**Fix Required**: Use atomic compare-and-swap or lock:
```python
async def acquire(self, credit: int) -> Lease:
    async with self._acquire_lock:  # Add lock
        # ... existing selection logic ...
        if target != -1:
            if self.workers[target] + credit > self.max_running_requests:
                continue  # Re-select if race occurred
            self.workers[target] += credit
            self.running_request += credit
            return self.Lease(self, lease=credit, dp_rank=target)
```

---

### P0-R2-04: ReplayBuffer.poll() Has No Timeout, Can Block Forever

**Severity**: CRITICAL - Indefinite hang
**Location**: [`generate_scheduler.py:415-427`](third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py:415)

**Status**: ❌ **INVALID / Over-scoped for Phase 3** (2026-02-13) — `poll()` is designed to block as backpressure until capacity is available or shutdown is triggered.

**Problem**:
```python
async def poll(self) -> int:
    prompt_id = self._next_pid()
    while True:  # NO TIMEOUT!
        if self._shutdown:
            raise asyncio.CancelledError
        elif self._check_send_new_request():
            self.prompt_id_to_start_step[prompt_id] = None
            return prompt_id
        self.event.clear()
        await self.event.wait()  # NO TIMEOUT!
```

**Impact**:
- If `_check_send_new_request()` always returns False and `_shutdown` never set, blocks forever
- No way to detect stuck state
- Blocks entire training pipeline

**Fix Required**: Add timeout parameter and fail-fast.

---

### P0-R2-05: ReplayBuffer.get_batch() Potential Infinite Loop

**Severity**: CRITICAL - Infinite loop
**Location**: [`generate_scheduler.py:541-551`](third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py:541)

**Status**: ❌ **INVALID / Overthinking** (2026-02-13) — in the happy path, `group.get_batch()` makes progress because running prompts complete; adding timeouts/stuck detection is out of Phase 3 scope.

**Problem**:
```python
# step == self.current_step, wait for scheduler to send enough new prompts
while collected_samples < expected_samples:  # NO TERMINATION GUARANTEE!
    # There may be no running prompt at this time,
    # yield control to schedule process_new_prompt.
    await asyncio.sleep(0)
    finished_prompts = await group.get_batch(expected_samples=expected_samples-collected_samples)
    amount = sum(len(response) for response in finished_prompts)
    collected_samples += amount
    # ...
```

**Impact**:
- If `group.get_batch()` returns empty list repeatedly (e.g., all prompts aborted), loop never terminates
- No check for "no running prompts" condition
- Blocks training indefinitely

**Fix Required**:
```python
while collected_samples < expected_samples:
    await asyncio.sleep(0)
    finished_prompts = await group.get_batch(expected_samples=expected_samples-collected_samples)
    amount = sum(len(response) for response in finished_prompts)
    
    if amount == 0 and len(group.running_prompts) == 0:
        raise RuntimeError(
            f"Cannot collect {expected_samples} samples: no running prompts and "
            f"only {collected_samples} samples collected"
        )
    
    collected_samples += amount
```

---

### P0-R2-06: DynamicSamplingScheduler.shutdown() Race Condition

**Severity**: CRITICAL - State corruption
**Location**: [`generate_scheduler.py:899-905`](third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py:899)

**Status**: ❌ **INVALID** (2026-02-13) — shutdown sets `ReplayBuffer._shutdown` first, which prevents new prompt issuance (`poll()` raises `CancelledError`). `load_balancer.resume()` is used to unblock waiters for teardown; it does not create new prompts.

**Problem**:
```python
async def shutdown(self):
    self.replay_buffer.shutdown()
    self.load_balancer.resume()  # ALLOWS NEW REQUESTS TO START!
    self.gc()
    await self.abort_running_requests()  # NEW REQUESTS MAY HAVE STARTED!
    await self.load_balancer.wait_complete()
    await self.async_sending_task
```

**Impact**:
- `load_balancer.resume()` is called BEFORE `abort_running_requests()`
- This allows new requests to start between resume and abort
- Those new requests may not be properly cleaned up
- Resource leaks, stranded requests

**Fix Required**:
```python
async def shutdown(self):
    self.replay_buffer.shutdown()
    # Do NOT resume - keep suspended to prevent new requests
    self.gc()
    await self.abort_running_requests()
    await self.load_balancer.wait_complete()
    await self.async_sending_task
    # Now safe to resume (or just skip resume since we're shutting down)
```

---

### P0-R2-07: _rebalance_on_shrink() Incomplete Rollback on Failure

**Severity**: CRITICAL - State inconsistency
**Location**: [`generate_scheduler.py:1563-1568`](third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py:1563)

**Status**: ❌ **INVALID / Over-scoped for Phase 3** (2026-02-13) — Phase 3 is fail-fast: if shrink fails mid-way, the error propagates and the job is expected to stop. Restoring mappings for continued execution is unnecessary.

**Problem**:
```python
except Exception as e:
    self.active_dp_ranks = old_active_ranks  # Rollback 1
    self.need_suspend = old_need_suspend      # Rollback 2
    if not self.need_suspend:
        self.suspend_notifier.set()
    raise RuntimeError(f"Shrink failed: {e}") from e
    # MISSING: Rollback of src_rank2_dp_rank mappings cleared at line 1554!
```

**Impact**:
- Line 1554 clears `src_rank2_dp_rank` mappings: `self._clear_src_rank_mappings(src_ranks_to_remap)`
- On failure, `active_dp_ranks` is restored but mappings are NOT restored
- System in inconsistent state: mappings point to wrong workers
- Future requests may route to wrong workers

**Fix Required**:
```python
async def _rebalance_on_shrink(self, shrink_dp_ranks: List[int]) -> Dict[str, int]:
    old_active_ranks = self.active_dp_ranks.copy()
    old_need_suspend = self.need_suspend
    old_src_rank2_dp_rank = self.src_rank2_dp_rank.copy()  # Save mappings
    
    # ... existing logic ...
    
    except Exception as e:
        self.active_dp_ranks = old_active_ranks
        self.need_suspend = old_need_suspend
        self.src_rank2_dp_rank = old_src_rank2_dp_rank  # Restore mappings
        if not self.need_suspend:
            self.suspend_notifier.set()
        raise RuntimeError(f"Shrink failed: {e}") from e
```

---

### P0-R2-08: generate_one_request() No Timeout on Worker RPC

**Severity**: CRITICAL - Indefinite hang
**Location**: [`generate_scheduler.py:1335`](third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py:1335)

**Status**: ❌ **INVALID / Over-scoped for Phase 3** (2026-02-13) — request RPCs are expected to complete in the happy path. Timeouts are a fault-tolerance policy and are intentionally not added in ENG-123 Phase 3.

**Problem**:
```python
try:
    response_data = await self.infer_cluster.workers[dp_rank].generate_request.remote(data=data)
    # NO TIMEOUT!
```

**Impact**:
- If worker hangs (OOM, deadlock, network issue), request hangs forever
- Blocks the entire request pipeline
- No way to detect or recover

**Fix Required**:
```python
try:
    response_data = await asyncio.wait_for(
        self.infer_cluster.workers[dp_rank].generate_request.remote(data=data),
        timeout=self.request_timeout  # Add configurable timeout
    )
except asyncio.TimeoutError:
    logger.error(f"Worker {dp_rank} generate_request timed out after {self.request_timeout}s")
    raise
```

---

### P0-R2-09: abort_request() No Error Handling for Dead Workers

**Severity**: CRITICAL - Silent failure
**Location**: [`generate_scheduler.py:1374-1379`](third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py:1374)

**Status**: ❌ **INVALID** (2026-02-13) — a dead worker causing abort to raise is acceptable under fail-fast semantics (the pipeline should crash loudly).

**Problem**:
```python
async def abort_request(self):
    await asyncio.gather(*(
        self.infer_cluster.workers[dp_rank].abort_requests.remote(list(self.running_requests[dp_rank]))
        for dp_rank in range(self.infer_cluster.world_size)
        if self.running_requests[dp_rank]
    ))
    # NO ERROR HANDLING!
```

**Impact**:
- If worker is dead (`RayActorError`), abort fails silently
- `running_requests` not cleaned up
- Subsequent operations may hang waiting for dead worker
- No timeout on abort RPC

**Fix Required**:
```python
async def abort_request(self):
    abort_tasks = []
    for dp_rank in range(self.infer_cluster.world_size):
        if self.running_requests[dp_rank]:
            abort_tasks.append(
                self._abort_with_timeout(dp_rank, list(self.running_requests[dp_rank]))
            )
    
    results = await asyncio.gather(*abort_tasks, return_exceptions=True)
    for dp_rank, result in zip(range(self.infer_cluster.world_size), results):
        if isinstance(result, Exception):
            logger.warning(f"Abort failed for dp_rank {dp_rank}: {result}")
            # Clear running requests anyway to prevent hang
            self.running_requests[dp_rank].clear()

async def _abort_with_timeout(self, dp_rank: int, request_ids: List[str]):
    try:
        await asyncio.wait_for(
            self.infer_cluster.workers[dp_rank].abort_requests.remote(request_ids),
            timeout=30.0
        )
    except (asyncio.TimeoutError, RayActorError) as e:
        logger.warning(f"Abort RPC failed for dp_rank {dp_rank}: {e}")
        raise
```

---

### P0-R2-10: ItemsGroup.get_batch() Potential Infinite Wait

**Severity**: CRITICAL - Indefinite hang
**Location**: [`generate_scheduler.py:280-305`](third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py:280)

**Status**: ❌ **INVALID / Over-scoped for Phase 3** (2026-02-13) — this is expected blocking behavior; adding timeouts/stuck detection is out of scope.

**Problem**:
```python
async def get_batch(self, expected_samples) -> List[List[ExperienceItem]]:
    assert expected_samples >= 0
    while self.num_samples < expected_samples and not len(self.running_prompts) == 0:
        self.event.clear()
        await self.event.wait()  # NO TIMEOUT!
```

**Impact**:
- If `running_prompts` is not empty but never makes progress (all stuck), hangs forever
- No timeout, no detection of stuck state
- Blocks training pipeline

**Fix Required**: Add timeout and stuck detection.

---

### P0-R2-11: sending_request() Bare Except Catches Everything

**Severity**: CRITICAL - Debugging impossibility
**Location**: [`generate_scheduler.py:1072-1076`](third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py:1072)

**Status**: ✅ **RESOLVED** (2026-02-13) — `sending_request()` now catches `asyncio.CancelledError` only (shutdown) and does not swallow other exceptions.

**Problem**:
```python
while True:
    try:
        prompt_id = await self.replay_buffer.poll()
    except:  # BARE EXCEPT - CATCHES EVERYTHING!
        logger.info(f"stop sending_request coroutine")
        break
```

**Impact**:
- Catches `KeyboardInterrupt`, `SystemExit`, `MemoryError`
- Makes debugging production issues nearly impossible
- Silently swallows critical errors

**Note**: This was partially covered in previous review (P0-A4) but the fix was incomplete - the code still uses bare `except:`.

**Fix Required**:
```python
except asyncio.CancelledError:
    logger.info("stop sending_request coroutine (cancelled)")
    break
except Exception as e:
    logger.exception(f"Unexpected error in sending_request: {e}")
    raise
```

---

### P0-R2-12: Request ID Not Pipeline-Scoped (Duplicate of P0-A1)

**Status**: ✅ **RESOLVED** (2026-02-13) — Phase 3 uses `meta_info["schedrl_request_id"]` as the SchedRL-canonical ID; `meta_info["request_id"]` remains ROLL-internal for backend compatibility.

---

### P0-R2-13: request_id_2_dp_rank Memory Leak (Duplicate of P0-A2)

**Status**: ✅ **RESOLVED** (2026-02-13) — `RequestScheduler.generate_one_request()` now pops `request_id_2_dp_rank[request_id]` in its `finally:` cleanup path.

---

### P0-R2-14: Expand Rebalance Infinite Loop (Duplicate of Bug #3)

**Status**: ✅ **RESOLVED** (2026-02-13) — expand rebalance selection now terminates when all per-rank lists are empty and caps work by `available_to_abort`.

---

### P0-R2-15: VLLM Strategy _collect_metrics_snapshot No Cancellation Handling

**Severity**: CRITICAL - Background task leak
**Location**: [`vllm_strategy.py:370-389`](third_party/ROLL/roll/distributed/strategy/vllm_strategy.py:370)

**Status**: ❌ **INVALID / Over-scoped for Phase 3** (2026-02-13) — cancellation/shutdown plumbing for background metrics tasks is non-critical to Phase 3 integration and adds complexity beyond fail-fast requirements.

**Problem**:
```python
async def _collect_metrics_snapshot(self):
    """Collect metrics snapshots periodically in a background thread."""
    from vllm.v1.metrics.reader import get_metrics_snapshot
    while True:  # NO CANCELLATION HANDLING!
        raw_metrics = get_metrics_snapshot()
        # ... process metrics ...
        await asyncio.sleep(self._metrics_snapshot_interval)
```

**Impact**:
- Background task runs forever with no way to stop it
- On strategy shutdown, task continues running
- Memory leak, resource leak
- May access dead strategy state

**Fix Required**:
```python
async def _collect_metrics_snapshot(self):
    """Collect metrics snapshots periodically in a background thread."""
    from vllm.v1.metrics.reader import get_metrics_snapshot
    try:
        while True:
            raw_metrics = get_metrics_snapshot()
            # ... process metrics ...
            await asyncio.sleep(self._metrics_snapshot_interval)
    except asyncio.CancelledError:
        logger.info("Metrics collection task cancelled")
        raise

async def shutdown(self):  # Add shutdown method
    if self._metrics_task:
        self._metrics_task.cancel()
        try:
            await self._metrics_task
        except asyncio.CancelledError:
            pass
```

---

## P1 Bugs (High Priority)

### P1-R2-01: LoadBalancer._release() Negative Credit Not Prevented

**Severity**: HIGH - Credit accounting corruption
**Location**: [`generate_scheduler.py:191-199`](third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py:191)

**Status**: ❌ **INVALID / Overthinking** (2026-02-13) — credit underflow indicates a logic bug and should crash loudly. Tightening assertions into explicit raises is not required for Phase 3.

**Problem**:
```python
def _release(self, dp_rank: int, credit: int = 1):
    assert credit >= 0
    self.workers[dp_rank] -= credit
    self.running_request -= credit
    assert self.workers[dp_rank] >= 0  # Assertion, not proper error handling
    assert self.running_request >= 0
```

**Impact**:
- Assertions can be disabled with `-O` flag
- If credit accounting is corrupted, assertions don't catch it in production
- Should use explicit validation with proper error messages

**Fix Required**:
```python
def _release(self, dp_rank: int, credit: int = 1):
    if credit < 0:
        raise ValueError(f"credit must be non-negative, got {credit}")
    if dp_rank not in self.workers:
        raise KeyError(f"Unknown dp_rank: {dp_rank}")
    
    self.workers[dp_rank] -= credit
    self.running_request -= credit
    
    if self.workers[dp_rank] < 0:
        raise RuntimeError(
            f"Credit underflow for dp_rank {dp_rank}: "
            f"workers[{dp_rank}]={self.workers[dp_rank]}"
        )
    if self.running_request < 0:
        raise RuntimeError(
            f"Global credit underflow: running_request={self.running_request}"
        )
```

---

### P1-R2-02: ReplayBuffer._check_send_new_request() Integer Overflow Not Handled

**Severity**: HIGH - State corruption
**Location**: [`generate_scheduler.py:404-413`](third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py:404)

**Status**: ❌ **INVALID / Overthinking** (2026-02-13) — Python integers do not overflow; this is theoretical and not a Phase 3 issue.

**Problem**:
```python
def _check_send_new_request(self) -> bool:
    if self.running_prompts + self.completed_prompts < self.batch_size:
        self.running_prompts += 1
        return True
    # ...
```

**Impact**:
- No upper bound check on `running_prompts`
- If called more times than `batch_size`, can overflow
- Integer overflow in Python becomes long, but logic breaks

**Fix Required**: Add explicit bounds checking.

---

### P1-R2-03: ItemsGroup.commit_prompt() No Validation

**Severity**: HIGH - State corruption
**Location**: [`generate_scheduler.py:268-273`](third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py:268)

**Status**: ❌ **INVALID** (2026-02-13) — incorrect inputs should raise immediately; more validation is not required for Phase 3 integration.

**Problem**:
```python
def commit_prompt(self, prompt_id: int, result: List[ExperienceItem]):
    self.running_prompts.remove(prompt_id)  # Can raise KeyError if not in set
    assert prompt_id not in self.finished_prompts  # Assertion, not proper check
    self.finished_prompts.append(result)
    self.num_samples += len(result)
    self.event.set()
```

**Impact**:
- `remove()` can raise `KeyError` if prompt_id not in set
- Assertion can be disabled
- No validation of `result` contents

**Fix Required**: Add proper error handling.

---

### P1-R2-04: RequestScheduler._get_gpus_for_dp_rank() No Error Handling

**Severity**: HIGH - Crash on invalid input
**Location**: [`generate_scheduler.py:1405-1430`](third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py:1405)

**Status**: ❌ **INVALID** (2026-02-13) — invalid `dp_rank` is a programming error; crashing is acceptable under fail-fast semantics.

**Problem**:
```python
def _get_gpus_for_dp_rank(self, dp_rank: int) -> List[int]:
    devices_info = self.infer_cluster.rank2devices[dp_rank]  # Can raise KeyError
    # ...
```

**Impact**:
- No validation that `dp_rank` exists in `rank2devices`
- No handling for missing or malformed device info
- Crash on invalid input

**Fix Required**: Add input validation.

---

### P1-R2-05: RequestScheduler.resume() Called Without Checking State

**Severity**: HIGH - State machine violation
**Location**: [`generate_scheduler.py:1399-1403`](third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py:1399)

**Status**: ❌ **INVALID** (2026-02-13) — `resume()` being a no-op when not suspended is intentional; adding warnings is optional.

**Problem**:
```python
def resume(self):
    if not self.need_suspend:
        return  # Silent no-op
    self.need_suspend = False
    self.suspend_notifier.set()
```

**Impact**:
- `resume()` is a no-op if not suspended
- No logging, no error
- Makes debugging difficult
- May mask logic errors where resume is called incorrectly

**Fix Required**:
```python
def resume(self):
    if not self.need_suspend:
        logger.warning("resume() called but not suspended")
        return
    logger.info("Resuming request scheduler")
    self.need_suspend = False
    self.suspend_notifier.set()
```

---

### P1-R2-06: RolloutContext.do_generate_and_reward() Lease Not Cleared on All Exception Paths

**Severity**: HIGH - Resource leak
**Location**: [`generate_scheduler.py:1200-1231`](third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py:1200)

**Status**: ✅ **RESOLVED** (2026-02-13) — `do_generate_and_reward()` now clears the lease in `finally:` (always releases remaining credit), and also clears on `BaseException` during `begin`/`yield` setup.

**Problem**:
```python
@asynccontextmanager
async def do_generate_and_reward(self, max_concurrency):
    # ...
    self._lease = await self._scheduler.load_balancer.acquire(credit=max_concurrency)

    try:
        sampling_start_step = await self._scheduler.replay_buffer.begin(prompt_id=self.prompt_id)
    except:
        self._lease.clear()  # Clear on exception
        raise
    # ...
    try:
        yield
    except:
        self._lease.clear()  # Clear on exception
        raise
    finally:
        # ... cleanup ...
```

**Impact**:
- If `yield` raises an exception AND `_lease.clear()` also raises, lease is leaked
- Nested exception handling is fragile

**Fix Required**: Use more robust cleanup pattern.

---

### P1-R2-07: DynamicSamplingScheduler.get_batch() No Validation of finished_items

**Severity**: HIGH - Silent data corruption
**Location**: [`generate_scheduler.py:998-1003`](third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py:998)

**Status**: ❌ **INVALID / Overthinking** (2026-02-13) — this is internal consistency checking; leaving as asserts is acceptable for Phase 3.

**Problem**:
```python
if self.pipeline_config.is_use_additional_prompts:
    # Keep the first batch_size*num_return_sequences ExperienceItem now.
    assert len(finished_items) >= batch_size * num_return_sequences  # Assertion only
    finished_items = finished_items[:batch_size * num_return_sequences]
assert len(finished_items) == batch_size * num_return_sequences  # Assertion only
```

**Impact**:
- Assertions can be disabled
- No proper validation of data integrity
- Silent truncation may lose data

**Fix Required**: Use explicit validation with proper errors.

---

### P1-R2-08: VLLM Strategy offload_states() No Memory Validation

**Severity**: HIGH - Silent memory leak
**Location**: [`vllm_strategy.py:342-349`](third_party/ROLL/roll/distributed/strategy/vllm_strategy.py:342)

**Status**: ❌ **INVALID / Over-scoped for Phase 3** (2026-02-13) — Phase 3 avoids adding extra memory verification loops; failures surface as OOM and should fail-fast.

**Problem**:
```python
async def offload_states(self, include=None, non_blocking=False):
    await self.model.reset_prefix_cache()
    if include is None or OffloadStateType.model_params in include:
        if self.is_model_in_gpu:
            await self.model.offload_states(self.sleep_level)
            self.is_model_in_gpu = False  # Assumes success!
    gc.collect()
    current_platform.empty_cache()
    # NO VALIDATION THAT MEMORY WAS ACTUALLY RELEASED!
```

**Impact**:
- No validation that offload actually freed GPU memory
- `is_model_in_gpu = False` set regardless of actual state
- vLLM may not release memory immediately
- Subsequent operations may OOM

**Fix Required**: Add memory validation (covered in extraction plan lines 1803-1808).

---

## P2 Bugs (Medium Priority)

### P2-R2-01: LoadBalancer.full() Not Used Consistently

**Severity**: MEDIUM - Dead code
**Location**: [`generate_scheduler.py:204-205`](third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py:204)

**Problem**:
```python
def full(self) -> bool:
    return all(running_requests >= self.max_running_requests for running_requests in self.workers.values())
```

**Impact**:
- Method exists but is never called
- Could be useful for backpressure detection
- Dead code increases maintenance burden

---

### P2-R2-02: ReplayBuffer.gc() Complex Invariants Not Documented

**Severity**: MEDIUM - Maintainability
**Location**: [`generate_scheduler.py:559-592`](third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py:559)

**Problem**:
The `gc()` method has complex invariants that are not clearly documented:
- Must be called after `get_batch(step=current_step)` and before `advance_step(step=current_step + 1)`
- Assumes atomic operations (no yield)
- Complex interaction with `async_generation_ratio`

**Impact**:
- Easy to misuse
- Hard to maintain
- Bugs may be introduced during refactoring

---

### P2-R2-03: RequestScheduler.worker_iter Not Used

**Severity**: MEDIUM - Dead code
**Location**: [`generate_scheduler.py:1297`](third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py:1297)

**Problem**:
```python
self.worker_iter = itertools.cycle(range(self.infer_cluster.world_size))
```

**Impact**:
- `worker_iter` is created but never used
- Dead code
- May indicate incomplete implementation

---

## Additional Findings: SchedRL Scheduler (`schedrl/scheduler/scheduler.py`)

### P0-R2-16: request_gpus() No Timeout on Event Wait

**Severity**: CRITICAL - Indefinite hang
**Location**: [`scheduler.py:307`](schedrl/scheduler/scheduler.py:307)

**Status**: ❌ **INVALID / Over-scoped for Phase 3** (2026-02-13) — this wait is normal admission/allocation blocking; adding default timeouts is a policy choice and not required for correctness.

**Problem**:
```python
async def request_gpus(...) -> List[int]:
    # ...
    async with self._lock:
        # ... create pending request ...
        self._wakeup_event.set()
    await event.wait()  # NO TIMEOUT!
```

**Impact**:
- If scheduler never processes the request (e.g., stuck in earlier phase), caller hangs forever
- No way to detect or recover from stuck state
- Blocks entire training pipeline

**Fix Required**: Add configurable timeout with fail-fast.

---

### P0-R2-17: release_and_request_gpus() No Timeout on Event Wait

**Severity**: CRITICAL - Indefinite hang
**Location**: [`scheduler.py:351`](schedrl/scheduler/scheduler.py:351)

**Status**: ❌ **INVALID / Over-scoped for Phase 3** (2026-02-13) — same rationale as P0-R2-16.

**Problem**: Same as P0-R2-16 - `await event.wait()` with no timeout.

---

### P0-R2-18: Central Scheduling Loop No Timeout on Wakeup

**Severity**: CRITICAL - Indefinite hang
**Location**: [`scheduler.py:411`](schedrl/scheduler/scheduler.py:411)

**Status**: ❌ **INVALID** (2026-02-13) — loop is event-driven by design; periodic wakeups are a health-check feature, not a Phase 3 requirement.

**Problem**:
```python
async def _central_scheduling_loop(self) -> None:
    while True:
        await self._wakeup_event.wait()  # NO TIMEOUT!
        self._wakeup_event.clear()
        try:
            await self.scheduling_cycle()
```

**Impact**:
- If wakeup_event is never set, loop hangs forever
- No heartbeat / health check mechanism
- No way to detect dead scheduler

**Fix Required**: Add periodic wakeup for health checks:
```python
async def _central_scheduling_loop(self) -> None:
    while True:
        try:
            await asyncio.wait_for(self._wakeup_event.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            pass  # Periodic wakeup for health check
        self._wakeup_event.clear()
        # ...
```

---

### P0-R2-19: Adapter RPC No Timeout in _execute_shrink_ops

**Severity**: CRITICAL - Indefinite hang
**Location**: [`scheduler.py:674`](schedrl/scheduler/scheduler.py:674)

**Status**: ✅ **RESOLVED (partial)** (2026-02-13) — shrink RPC execution moved outside the scheduler lock to prevent deadlocks; timeouts are intentionally not added in Phase 3.

**Problem**:
```python
async def _execute_shrink_ops(self, plan: ExecutionPlan) -> None:
    # ...
    for pipeline_id, dp_ranks in sorted(pipeline_to_dp_ranks.items()):
        if not dp_ranks:
            continue
        adapter = self._get_or_lookup_adapter_handle_locked(pipeline_id=pipeline_id)
        await adapter.shrink_workers.remote(sorted(dp_ranks))  # NO TIMEOUT!
```

**Impact**:
- If adapter hangs (dead worker, network issue), scheduler hangs forever
- Blocks all subsequent scheduling cycles
- No fail-fast on adapter timeout

**Fix Required**:
```python
try:
    await asyncio.wait_for(
        adapter.shrink_workers.remote(sorted(dp_ranks)),
        timeout=30.0  # Configurable
    )
except asyncio.TimeoutError:
    await self._fail_fast_shutdown(reason=f"adapter_shrink_timeout: pipeline_id={pipeline_id}")
    raise
```

---

### P0-R2-20: KeyError on Missing Pipeline in Registry

**Severity**: CRITICAL - Crash on invalid state
**Location**: [`scheduler.py:451`](schedrl/scheduler/scheduler.py:451)

**Status**: ❌ **INVALID** (2026-02-13) — the scheduling cycle holds `_lock`, so pipelines cannot be unregistered concurrently during planning; direct indexing is safe under this lock discipline.

**Problem**:
```python
# Line 450-451:
pipeline_id, _ = parse_cluster_id(cluster_id)
infer_cfg = self._state.pipeline_registry[pipeline_id]["cluster_configs"]["actor_infer"]
# KeyError if pipeline_id not in registry!
```

**Impact**:
- If pipeline was unregistered between planning phases, crashes scheduler
- No graceful handling of race condition
- Same issue at lines 484, 536, 1034, 1091, 1120, 1198

**Fix Required**:
```python
pipeline_id, _ = parse_cluster_id(cluster_id)
pipeline_info = self._state.pipeline_registry.get(pipeline_id)
if pipeline_info is None:
    raise RuntimeError(f"pipeline_id {pipeline_id!r} was unregistered during scheduling cycle")
infer_cfg = pipeline_info["cluster_configs"]["actor_infer"]
```

---

### P1-R2-09: Assertions Can Be Disabled

**Severity**: HIGH - Silent correctness failure
**Location**: [`scheduler.py:754, 836`](schedrl/scheduler/scheduler.py:754)

**Status**: ❌ **INVALID / Overthinking** (2026-02-13) — these jobs do not run with `python -O`; converting asserts to raises everywhere is not a Phase 3 blocker.

**Problem**:
```python
assert idle_gpus.isdisjoint(non_gen_reserved_gpus), "idle_gpus must exclude non-GEN reserved GPUs"
```

**Impact**:
- Assertions can be disabled with `-O` flag
- Critical invariant check may be skipped in production
- Should use explicit validation

**Fix Required**:
```python
if not idle_gpus.isdisjoint(non_gen_reserved_gpus):
    raise RuntimeError("idle_gpus must exclude non-GEN reserved GPUs")
```

---

### P1-R2-10: Gap-Ratio Iteration Limits May Be Insufficient

**Severity**: HIGH - Premature failure
**Location**: [`scheduler.py:991-992`](schedrl/scheduler/scheduler.py:991)

**Status**: ❌ **INVALID / Overthinking** (2026-02-13) — guardrails prevent pathological loops; Phase 3 does not tune these without evidence.

**Problem**:
```python
if iterations > 10_000 or activations > 1_000:
    raise RuntimeError("gap_ratio_generation_planning_exceeded_limits")
```

**Impact**:
- Arbitrary limits may be insufficient for large clusters
- 1000 activations may be too few for 100+ GPU clusters
- No configuration mechanism

**Fix Required**: Make limits configurable and scale with cluster size:
```python
max_iterations = max(10_000, len(self._state.pipeline_registry) * 100)
max_activations = max(1_000, self._num_gpus * 10)
if iterations > max_iterations or activations > max_activations:
    raise RuntimeError(f"gap_ratio_generation_planning_exceeded_limits: iterations={iterations}, activations={activations}")
```

---

### P1-R2-11: notify_ready_to_release() No Timeout When timeout_s=None

**Severity**: HIGH - Indefinite hang
**Location**: [`scheduler.py:1238`](schedrl/scheduler/scheduler.py:1238)

**Status**: ❌ **INVALID / Over-scoped for Phase 3** (2026-02-13) — caller can pass `timeout_s` explicitly; setting defaults changes semantics.

**Problem**:
```python
if timeout_s is None:
    await event.wait()  # NO TIMEOUT!
else:
    await asyncio.wait_for(event.wait(), timeout=float(timeout_s))
```

**Impact**:
- If scheduler never commits the shrink, caller hangs forever
- Should have a default timeout even when timeout_s is None

**Fix Required**: Add default timeout:
```python
DEFAULT_RELEASE_TIMEOUT = 300.0  # 5 minutes

if timeout_s is None:
    timeout_s = DEFAULT_RELEASE_TIMEOUT
await asyncio.wait_for(event.wait(), timeout=float(timeout_s))
```

---

### P2-R2-04: Inconsistent Exception Types

**Severity**: MEDIUM - API inconsistency
**Location**: [`scheduler.py:807, 810`](schedrl/scheduler/scheduler.py:807)

**Problem**:
```python
raise KeyError(f"pipeline_id={pipeline_id!r} missing actor_infer cluster config")
# ...
raise ValueError(f"pipeline_id={pipeline_id!r} has invalid actor_infer tp_size={tp_size}")
```

**Impact**:
- Mixed exception types make error handling inconsistent
- Should use RuntimeError for all scheduler-internal errors
- KeyError implies dict access failure, not validation failure

---

## Additional Findings: SchedRL Orchestrator (`schedrl/orchestrator/orchestrator.py`)

### P0-R2-21: allocate_pipeline_id() Potential Infinite Loop

**Severity**: CRITICAL - Infinite loop
**Location**: [`orchestrator.py:151-155`](schedrl/orchestrator/orchestrator.py:151)

**Status**: ❌ **INVALID** (2026-02-13) — UUID collisions are practically impossible; adding retry limits is not required for Phase 3.

**Problem**:
```python
def allocate_pipeline_id(self) -> str:
    while True:  # NO TERMINATION GUARANTEE!
        pipeline_id = f"p_{uuid.uuid4().hex}"
        validate_pipeline_id(pipeline_id)
        if pipeline_id not in self._pipelines:
            return pipeline_id
```

**Impact**:
- UUID collision is astronomically unlikely, but the loop has no termination condition
- In theory, could loop forever if UUIDs collide repeatedly
- No maximum retry count

**Fix Required**: Add maximum retry count:
```python
def allocate_pipeline_id(self) -> str:
    for _ in range(100):  # Max retries
        pipeline_id = f"p_{uuid.uuid4().hex}"
        validate_pipeline_id(pipeline_id)
        if pipeline_id not in self._pipelines:
            return pipeline_id
    raise RuntimeError("Failed to allocate unique pipeline_id after 100 attempts")
```

---

### P0-R2-22: kill_pipeline() Uses Internal Ray APIs

**Severity**: CRITICAL - API stability risk
**Location**: [`orchestrator.py:289-307`](schedrl/orchestrator/orchestrator.py:289)

**Status**: ❌ **INVALID / Known tradeoff** (2026-02-13) — this is a best-effort fallback only for unnamed actors; Phase 3 already logs loudly when it happens and recommends naming actors.

**Problem**:
```python
from ray._raylet import ActorID  # type: ignore  # INTERNAL API!
# ...
handle = ray.worker.global_worker.core_worker.get_actor_handle(actor_id_obj)  # INTERNAL API!
```

**Impact**:
- Uses Ray internal APIs that may change without notice
- Will break on Ray version upgrades
- No deprecation warning or fallback

**Fix Required**: Document this as a known risk and add version check:
```python
import ray
RAY_VERSION = tuple(int(x) for x in ray.__version__.split(".")[:2])
if RAY_VERSION < (2, 0) or RAY_VERSION >= (3, 0):
    raise RuntimeError(
        f"kill_pipeline internal API compatibility not verified for Ray {ray.__version__}. "
        "Please ensure all actors are named to avoid relying on internal APIs."
    )
```

---

### P0-R2-23: kill_pipeline() No Timeout on Scheduler RPCs

**Severity**: CRITICAL - Indefinite hang
**Location**: [`orchestrator.py:230, 236`](schedrl/orchestrator/orchestrator.py:230)

**Status**: ❌ **INVALID / Over-scoped for Phase 3** (2026-02-13) — adding timeouts to control-plane `ray.get(...)` calls is a policy choice and not required for Phase 3 correctness.

**Problem**:
```python
ray_namespace = ray.get(self._scheduler.get_pipeline_namespace.remote(pipeline_id=pipeline_id))  # NO TIMEOUT!
# ...
ray.get(self._scheduler.unregister_pipeline.remote(pipeline_id=pipeline_id))  # NO TIMEOUT!
```

**Impact**:
- If scheduler is dead/unresponsive, kill_pipeline hangs forever
- Cannot kill a pipeline if scheduler is stuck
- Blocks cleanup and recovery

**Fix Required**: Add timeouts:
```python
try:
    ray_namespace = ray.get(
        self._scheduler.get_pipeline_namespace.remote(pipeline_id=pipeline_id),
        timeout=10.0
    )
except asyncio.TimeoutError:
    raise RuntimeError(f"Scheduler unresponsive for pipeline_id {pipeline_id!r}")
```

---

### P1-R2-12: _force_stop_cluster_workers_first() Hardcoded Sleep

**Severity**: HIGH - Unnecessary delay
**Location**: [`orchestrator.py:126`](schedrl/orchestrator/orchestrator.py:126)

**Problem**:
```python
time.sleep(0.2)  # Hardcoded sleep
```

**Impact**:
- Magic number with no explanation
- May be insufficient on slow systems
- May be excessive on fast systems

**Fix Required**: Make configurable or remove if unnecessary.

---

### P1-R2-13: kill_pipeline() Silent Exception Swallowing

**Severity**: HIGH - Debugging difficulty
**Location**: [`orchestrator.py:266-273`](schedrl/orchestrator/orchestrator.py:266)

**Problem**:
```python
try:
    handle = ray.get_actor(name, namespace=ray_namespace)
except Exception:
    continue  # Silent swallow
try:
    ray.kill(handle, no_restart=True)
except Exception:
    continue  # Silent swallow
```

**Impact**:
- All exceptions silently swallowed
- No logging of failures
- Makes debugging nearly impossible

**Fix Required**: Add logging:
```python
try:
    handle = ray.get_actor(name, namespace=ray_namespace)
except Exception as e:
    logger.debug(f"Could not get actor {name}: {e}")
    continue
try:
    ray.kill(handle, no_restart=True)
except Exception as e:
    logger.warning(f"Failed to kill actor {name}: {e}")
    continue
```

---

### P1-R2-14: shutdown() Not Idempotent on Concurrent Calls

**Severity**: HIGH - Race condition
**Location**: [`orchestrator.py:327-333`](schedrl/orchestrator/orchestrator.py:327)

**Problem**:
```python
def shutdown(self, force: bool = True, reason: Optional[str] = None, source: Optional[str] = None) -> None:
    if self._shutdown_started:
        return
    self._shutdown_started = True  # Race condition!
    if not force:
        raise RuntimeError("shutdown(force=False) is not supported in ENG-123 Phase 1")
    _force_stop_cluster_workers_first()
```

**Impact**:
- Two concurrent calls can both pass the `if self._shutdown_started` check
- Both proceed to set `_shutdown_started = True`
- Both call `_force_stop_cluster_workers_first()`
- Double shutdown attempt

**Fix Required**: Use asyncio.Lock or threading.Lock:
```python
def __init__(self, ...):
    # ...
    self._shutdown_lock = threading.Lock()

def shutdown(self, ...):
    with self._shutdown_lock:
        if self._shutdown_started:
            return
        self._shutdown_started = True
    _force_stop_cluster_workers_first()
```

---

### P2-R2-05: _ensure_scheduler_singleton() Race Condition

**Severity**: MEDIUM - Race condition
**Location**: [`orchestrator.py:63-82`](schedrl/orchestrator/orchestrator.py:63)

**Problem**:
```python
try:
    return ray.get_actor(SCHEDULER_ACTOR_NAME, namespace=SCHEDRL_NAMESPACE)
except ValueError:
    pass  # Not found, create it

# ... create scheduler ...
try:
    scheduler = SchedulerActor.options(...).remote()
except Exception:
    scheduler = ray.get_actor(...)  # Another process may have created it
```

**Impact**:
- Two processes can both fail to find the actor
- Both try to create it
- One fails and falls back to get_actor
- Works but is racy and may cause issues

**Fix Required**: Use Ray's `get_or_create` pattern if available, or add retry logic.

---

## Summary

### Bug Count by Angle

| Angle | P0 | P1 | P2 | Total |
|-------|----|----|----|-------|
| Queue/Backpressure (ROLL) | 5 | 2 | 1 | 8 |
| Worker Init/Shutdown (ROLL) | 1 | 0 | 0 | 1 |
| Timeout Cascades (ROLL) | 2 | 0 | 0 | 2 |
| Event Loop Blocking (ROLL) | 1 | 0 | 0 | 1 |
| Ray Actor Lifecycle (ROLL) | 1 | 1 | 0 | 2 |
| Memory Pressure (ROLL) | 0 | 1 | 0 | 1 |
| Data Plane Correctness (ROLL) | 1 | 2 | 0 | 3 |
| Code Quality (ROLL) | 0 | 2 | 2 | 4 |
| **Duplicates (ROLL)** | 3 | 0 | 0 | 3 |
| Timeout Cascades (SchedRL) | 4 | 1 | 0 | 5 |
| Registry Consistency (SchedRL) | 1 | 0 | 0 | 1 |
| Assertions (SchedRL) | 0 | 1 | 0 | 1 |
| Algorithm Limits (SchedRL) | 0 | 1 | 0 | 1 |
| Exception Types (SchedRL) | 0 | 0 | 1 | 1 |
| Pipeline ID Allocation (Orchestrator) | 1 | 0 | 0 | 1 |
| Ray Internal APIs (Orchestrator) | 1 | 0 | 0 | 1 |
| Scheduler RPC Timeouts (Orchestrator) | 1 | 0 | 0 | 1 |
| Shutdown Race Conditions (Orchestrator) | 0 | 2 | 1 | 3 |
| Silent Exception Swallowing (Orchestrator) | 0 | 1 | 0 | 1 |
| **TOTAL NEW** | **18** | **11** | **5** | **34** |

### Files Requiring Changes

| File | P0 Bugs | P1 Bugs | P2 Bugs | Est. Lines |
|------|---------|---------|---------|------------|
| `generate_scheduler.py` | 9 | 6 | 3 | ~150 lines |
| `vllm_strategy.py` | 1 | 1 | 0 | ~20 lines |
| `scheduler.py` | 5 | 3 | 1 | ~80 lines |
| `orchestrator.py` | 3 | 2 | 1 | ~50 lines |

### Combined Total (All Reviews)

| Review Round | P0 Bugs | P1 Bugs | P2 Bugs | Total |
|--------------|---------|---------|---------|-------|
| Previous Review (2026-02-12) | 54 | 32 | 1 | 87 |
| **This Review (Round 2)** | **18** | **11** | **5** | **34** |
| **GRAND TOTAL** | **72** | **43** | **6** | **121** |

Note: 3 P0 bugs in this review are duplicates of previous findings (P0-A1, P0-A2, Bug #3) that remain unfixed.

---

## Recommended Fix Priority

### Phase 0: Queue/Backpressure (Blocks All Operations)
1. **P0-R2-01**: LoadBalancer.Lease `__del__` crash
2. **P0-R2-02**: LoadBalancer.acquire() no timeout
3. **P0-R2-03**: LoadBalancer.acquire() race condition
4. **P0-R2-04**: ReplayBuffer.poll() no timeout
5. **P0-R2-05**: ReplayBuffer.get_batch() infinite loop

### Phase 1: Error Handling (Blocks Debugging)
6. **P0-R2-11**: sending_request() bare except
7. **P0-R2-09**: abort_request() no error handling
8. **P0-R2-07**: _rebalance_on_shrink() incomplete rollback

### Phase 2: Timeout Handling (Blocks Reliability)
9. **P0-R2-08**: generate_one_request() no timeout
10. **P0-R2-10**: ItemsGroup.get_batch() infinite wait
11. **P0-R2-16**: request_gpus() no timeout
12. **P0-R2-17**: release_and_request_gpus() no timeout
13. **P0-R2-18**: Central scheduling loop no timeout
14. **P0-R2-19**: Adapter RPC no timeout
15. **P0-R2-23**: kill_pipeline() no timeout on scheduler RPCs

### Phase 3: Resource Management (Blocks Stability)
11. **P0-R2-06**: shutdown() race condition
12. **P0-R2-15**: VLLM metrics task no cancellation
13. **P0-R2-20**: KeyError on missing pipeline in registry
14. **P0-R2-21**: allocate_pipeline_id() potential infinite loop
15. **P0-R2-22**: kill_pipeline() uses internal Ray APIs

### Phase 4: P1 Bugs (Polish)
- All P1 bugs in order of appearance

---

## Validation Plan

After fixes are applied, validate:

1. **Lease lifecycle test**:
   - Create Lease, let it be GC'd without clear()
   - Verify warning logged, no crash
   - Verify credits returned

2. **Timeout test**:
   - Set all workers to max capacity
   - Call acquire() with timeout
   - Verify TimeoutError raised

3. **Race condition test**:
   - Call acquire() from multiple coroutines simultaneously
   - Verify no worker exceeds max_running_requests

4. **Shutdown test**:
   - Start requests, call shutdown()
   - Verify no new requests start after shutdown begins
   - Verify all resources cleaned up

5. **Rollback test**:
   - Trigger failure during shrink
   - Verify all state restored to pre-shrink values

"""Behavioral tests for the extracted gap-ratio planning module.

Test 1: single-pipeline idle GPU activation (free-GPU path).
Test 2: two-pipeline donor shrink (donor-search path).
"""
from __future__ import annotations

import asyncio
import importlib
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RLIX_ROOT = REPO_ROOT / "rlix"


def _install_import_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same stub pattern as test_scheduler_apply_plan_invariants.py."""
    for module_name in list(sys.modules):
        if module_name == "ray" or module_name.startswith("rlix"):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    ray_stub = types.ModuleType("ray")

    def _remote(*args, **kwargs):
        def _decorate(obj):
            return obj
        return _decorate

    ray_stub.remote = _remote
    ray_stub.get_actor = lambda *args, **kwargs: None
    ray_stub.get = lambda value: value
    monkeypatch.setitem(sys.modules, "ray", ray_stub)

    package_roots = {
        "rlix": RLIX_ROOT,
        "rlix.protocol": RLIX_ROOT / "protocol",
        "rlix.scheduler": RLIX_ROOT / "scheduler",
        "rlix.utils": RLIX_ROOT / "utils",
    }
    for module_name, module_path in package_roots.items():
        package_module = types.ModuleType(module_name)
        package_module.__path__ = [str(module_path)]
        monkeypatch.setitem(sys.modules, module_name, package_module)


def _load_gap_ratio_modules(monkeypatch: pytest.MonkeyPatch):
    _install_import_stubs(monkeypatch)
    gap_ratio_mod = importlib.import_module("rlix.scheduler.planner")
    scheduler_types = importlib.import_module("rlix.scheduler.types")
    protocol_types = importlib.import_module("rlix.protocol.types")
    return gap_ratio_mod, scheduler_types, protocol_types


def test_single_pipeline_idle_gpus_activated(monkeypatch: pytest.MonkeyPatch) -> None:
    """One pipeline with 2 idle generation GPUs -> gap-ratio activates both dp workers."""
    gap_ratio_mod, scheduler_types, protocol_types = _load_gap_ratio_modules(monkeypatch)

    ExecutionPlan = scheduler_types.ExecutionPlan
    Priority = protocol_types.Priority
    Request = scheduler_types.Request
    PendingRequest = scheduler_types.PendingRequest

    plan = ExecutionPlan()
    pipeline_id = "ft_000000000000"
    cluster_id = f"{pipeline_id}_actor_infer"

    # Construct inputs: 0 active dp workers, 2 inactive, 2 idle GPUs
    _GapRatioDPWorker = gap_ratio_mod._GapRatioDPWorker
    active_dp_workers = {pipeline_id: []}
    inactive_dp_workers = {
        pipeline_id: [
            _GapRatioDPWorker(pipeline_id=pipeline_id, dp_rank=0, gpu_ids=[2]),
            _GapRatioDPWorker(pipeline_id=pipeline_id, dp_rank=1, gpu_ids=[3]),
        ]
    }

    pipeline_registry = {
        pipeline_id: {
            "cluster_configs": {
                "actor_infer": {
                    "tp_size": 1,
                    "is_generation": True,
                    "device_mapping": [2, 3],
                    "max_dp_workers": 2,
                },
            },
            "admitted": True,
        }
    }

    pending_bucket_gen = [
        PendingRequest(
            request=Request(cluster_id=cluster_id, priority=Priority.GENERATION, timestamp=0.0),
            event=asyncio.Event(),
        )
    ]

    # 50% remaining -> nonzero demand weight
    def progress_totals_fn(*, pipeline_id):
        return (50.0, 100.0)

    remaining_idle = gap_ratio_mod.plan_generation_gap_ratio(
        plan,
        active_dp_workers=active_dp_workers,
        inactive_dp_workers=inactive_dp_workers,
        non_gen_reserved_gpus=set(),
        idle_gpus={2, 3},
        pipeline_registry=pipeline_registry,
        active_allocations={},
        pending_bucket_gen=pending_bucket_gen,
        progress_totals_fn=progress_totals_fn,
    )

    # Assert: both dp workers activated, all idle GPUs consumed
    assert len(plan.sched_guided_allocation_ops) == 1
    op = plan.sched_guided_allocation_ops[0]
    assert op.cluster_id == cluster_id
    assert {gpu_id for gpus in op.dp_rank_to_gpus_to_add.values() for gpu_id in gpus} == {2, 3}
    assert remaining_idle == set()
    # No shrink ops needed (GPUs were free, no donors)
    assert len(plan.sched_guided_shrink_ops) == 0


def test_pending_request_uses_step_target_estimate_without_progress_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pending GENERATION request can bootstrap demand from its explicit estimate."""
    gap_ratio_mod, scheduler_types, protocol_types = _load_gap_ratio_modules(monkeypatch)

    ExecutionPlan = scheduler_types.ExecutionPlan
    Priority = protocol_types.Priority
    Request = scheduler_types.Request
    PendingRequest = scheduler_types.PendingRequest

    plan = ExecutionPlan()
    pipeline_id = "ft_222222222222"
    cluster_id = f"{pipeline_id}_actor_infer"

    _GapRatioDPWorker = gap_ratio_mod._GapRatioDPWorker
    active_dp_workers = {pipeline_id: []}
    inactive_dp_workers = {
        pipeline_id: [
            _GapRatioDPWorker(pipeline_id=pipeline_id, dp_rank=0, gpu_ids=[0]),
            _GapRatioDPWorker(pipeline_id=pipeline_id, dp_rank=1, gpu_ids=[1]),
        ]
    }
    pipeline_registry = {
        pipeline_id: {
            "cluster_configs": {
                "actor_infer": {
                    "tp_size": 1,
                    "is_generation": True,
                    "device_mapping": [0, 1],
                    "max_dp_workers": 2,
                },
            },
            "admitted": True,
        }
    }
    pending_bucket_gen = [
        PendingRequest(
            request=Request(cluster_id=cluster_id, priority=Priority.GENERATION, timestamp=0.0),
            event=asyncio.Event(),
            step_target_estimate=4,
        )
    ]

    def progress_totals_fn(*, pipeline_id):
        return (0.0, 0.0)

    remaining_idle = gap_ratio_mod.plan_generation_gap_ratio(
        plan,
        active_dp_workers=active_dp_workers,
        inactive_dp_workers=inactive_dp_workers,
        non_gen_reserved_gpus=set(),
        idle_gpus={0, 1},
        pipeline_registry=pipeline_registry,
        active_allocations={},
        pending_bucket_gen=pending_bucket_gen,
        progress_totals_fn=progress_totals_fn,
    )

    assert len(plan.sched_guided_allocation_ops) == 1
    op = plan.sched_guided_allocation_ops[0]
    assert op.cluster_id == cluster_id
    assert set(op.gpus_to_allocate)
    assert set(op.gpus_to_allocate).issubset({0, 1})
    assert set(op.dp_ranks_to_add)
    assert remaining_idle != {0, 1}


def test_pending_request_without_progress_or_estimate_does_not_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    """No progress snapshot and no estimate means no synthetic demand is invented."""
    gap_ratio_mod, scheduler_types, protocol_types = _load_gap_ratio_modules(monkeypatch)

    ExecutionPlan = scheduler_types.ExecutionPlan
    Priority = protocol_types.Priority
    Request = scheduler_types.Request
    PendingRequest = scheduler_types.PendingRequest

    plan = ExecutionPlan()
    pipeline_id = "ft_333333333333"
    cluster_id = f"{pipeline_id}_actor_infer"

    _GapRatioDPWorker = gap_ratio_mod._GapRatioDPWorker
    active_dp_workers = {pipeline_id: []}
    inactive_dp_workers = {
        pipeline_id: [
            _GapRatioDPWorker(pipeline_id=pipeline_id, dp_rank=0, gpu_ids=[0]),
            _GapRatioDPWorker(pipeline_id=pipeline_id, dp_rank=1, gpu_ids=[1]),
        ]
    }
    pipeline_registry = {
        pipeline_id: {
            "cluster_configs": {
                "actor_infer": {
                    "tp_size": 1,
                    "is_generation": True,
                    "device_mapping": [0, 1],
                    "max_dp_workers": 2,
                },
            },
            "admitted": True,
        }
    }
    pending_bucket_gen = [
        PendingRequest(
            request=Request(cluster_id=cluster_id, priority=Priority.GENERATION, timestamp=0.0),
            event=asyncio.Event(),
        )
    ]

    def progress_totals_fn(*, pipeline_id):
        return (0.0, 0.0)

    remaining_idle = gap_ratio_mod.plan_generation_gap_ratio(
        plan,
        active_dp_workers=active_dp_workers,
        inactive_dp_workers=inactive_dp_workers,
        non_gen_reserved_gpus=set(),
        idle_gpus={0, 1},
        pipeline_registry=pipeline_registry,
        active_allocations={},
        pending_bucket_gen=pending_bucket_gen,
        progress_totals_fn=progress_totals_fn,
    )

    assert plan.sched_guided_allocation_ops == []
    assert remaining_idle == {0, 1}


def test_real_progress_overrides_pending_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reported progress, when present, takes precedence over request-carried estimates."""
    gap_ratio_mod, scheduler_types, protocol_types = _load_gap_ratio_modules(monkeypatch)

    ExecutionPlan = scheduler_types.ExecutionPlan
    Priority = protocol_types.Priority
    Request = scheduler_types.Request
    PendingRequest = scheduler_types.PendingRequest

    plan = ExecutionPlan()
    pipeline_id = "ft_444444444444"
    cluster_id = f"{pipeline_id}_actor_infer"

    _GapRatioDPWorker = gap_ratio_mod._GapRatioDPWorker
    active_dp_workers = {pipeline_id: []}
    inactive_dp_workers = {
        pipeline_id: [
            _GapRatioDPWorker(pipeline_id=pipeline_id, dp_rank=0, gpu_ids=[0]),
            _GapRatioDPWorker(pipeline_id=pipeline_id, dp_rank=1, gpu_ids=[1]),
        ]
    }
    pipeline_registry = {
        pipeline_id: {
            "cluster_configs": {
                "actor_infer": {
                    "tp_size": 1,
                    "is_generation": True,
                    "device_mapping": [0, 1],
                    "max_dp_workers": 2,
                },
            },
            "admitted": True,
        }
    }
    pending_bucket_gen = [
        PendingRequest(
            request=Request(cluster_id=cluster_id, priority=Priority.GENERATION, timestamp=0.0),
            event=asyncio.Event(),
            step_target_estimate=1000,
        )
    ]

    def progress_totals_fn(*, pipeline_id):
        return (5.0, 10.0)

    gap_ratio_mod.plan_generation_gap_ratio(
        plan,
        active_dp_workers=active_dp_workers,
        inactive_dp_workers=inactive_dp_workers,
        non_gen_reserved_gpus=set(),
        idle_gpus={0, 1},
        pipeline_registry=pipeline_registry,
        active_allocations={},
        pending_bucket_gen=pending_bucket_gen,
        progress_totals_fn=progress_totals_fn,
    )

    assert len(plan.sched_guided_allocation_ops) == 1
    op = plan.sched_guided_allocation_ops[0]
    assert set(op.gpus_to_allocate) == {0, 1}


def test_two_pipelines_donor_shrink(monkeypatch: pytest.MonkeyPatch) -> None:
    """Over-provisioned pipeline donates GPUs to under-provisioned pipeline."""
    gap_ratio_mod, scheduler_types, protocol_types = _load_gap_ratio_modules(monkeypatch)

    ExecutionPlan = scheduler_types.ExecutionPlan
    Priority = protocol_types.Priority
    Request = scheduler_types.Request
    PendingRequest = scheduler_types.PendingRequest
    ClusterAllocation = scheduler_types.ClusterAllocation

    plan = ExecutionPlan()
    pipeline_a = "ft_000000000000"  # Nearly complete -> low demand -> over-provisioned
    pipeline_b = "ft_111111111111"  # Just started -> high demand -> under-provisioned
    cluster_a = f"{pipeline_a}_actor_infer"
    cluster_b = f"{pipeline_b}_actor_infer"

    _GapRatioDPWorker = gap_ratio_mod._GapRatioDPWorker

    # Pipeline A: 4 active generation workers, no inactive
    # Pipeline B: 0 active, 4 inactive (same GPU pool — time-shared)
    active_dp_workers = {
        pipeline_a: [
            _GapRatioDPWorker(pipeline_id=pipeline_a, dp_rank=rank, gpu_ids=[rank]) for rank in range(4)
        ],
        pipeline_b: [],
    }
    inactive_dp_workers = {
        pipeline_a: [],
        pipeline_b: [
            _GapRatioDPWorker(pipeline_id=pipeline_b, dp_rank=rank, gpu_ids=[rank]) for rank in range(4)
        ],
    }

    pipeline_registry = {
        pid: {
            "cluster_configs": {
                "actor_infer": {
                    "tp_size": 1,
                    "is_generation": True,
                    "device_mapping": [0, 1, 2, 3],
                    "max_dp_workers": 4,
                },
            },
            "admitted": True,
        }
        for pid in [pipeline_a, pipeline_b]
    }

    active_allocations = {
        cluster_a: ClusterAllocation(
            cluster_id=cluster_a,
            gpu_ids=[0, 1, 2, 3],
            priority=Priority.GENERATION,
            active_dp_ranks={0, 1, 2, 3},
            dp_rank_to_gpus={0: [0], 1: [1], 2: [2], 3: [3]},
        ),
    }

    # Only Pipeline B has a pending request (drives demand inflation)
    pending_bucket_gen = [
        PendingRequest(
            request=Request(cluster_id=cluster_b, priority=Priority.GENERATION, timestamp=0.0),
            event=asyncio.Event(),
        )
    ]

    def progress_totals_fn(*, pipeline_id):
        if pipeline_id == pipeline_a:
            return (10.0, 100.0)  # 10% remaining -> low demand weight
        return (90.0, 100.0)  # 90% remaining -> high demand weight (+ inflation)

    remaining = gap_ratio_mod.plan_generation_gap_ratio(
        plan,
        active_dp_workers=active_dp_workers,
        inactive_dp_workers=inactive_dp_workers,
        non_gen_reserved_gpus=set(),
        idle_gpus=set(),  # No free GPUs — must donate from Pipeline A
        pipeline_registry=pipeline_registry,
        active_allocations=active_allocations,
        pending_bucket_gen=pending_bucket_gen,
        progress_totals_fn=progress_totals_fn,
    )

    # Assert: Pipeline A shrunk (donor), Pipeline B expanded (receiver)
    assert any(op.cluster_id == cluster_a for op in plan.sched_guided_shrink_ops)
    assert any(op.cluster_id == cluster_b for op in plan.sched_guided_allocation_ops)

    # Pipeline B got at least one GPU
    b_gpus = set()
    for op in plan.sched_guided_allocation_ops:
        if op.cluster_id == cluster_b:
            for gpus in op.dp_rank_to_gpus_to_add.values():
                b_gpus.update(gpus)
    assert len(b_gpus) >= 1


def test_no_donor_mutation_when_receiver_ineligible(monkeypatch: pytest.MonkeyPatch) -> None:
    """Receiver with no pending request and no active allocation must not trigger donor shrinks.

    Regression test: previously _try_activate_one committed donor shrink mutations before
    checking receiver eligibility, leaving orphaned shrink ops if the guard fired.
    """
    gap_ratio_mod, scheduler_types, protocol_types = _load_gap_ratio_modules(monkeypatch)

    ExecutionPlan = scheduler_types.ExecutionPlan
    Priority = protocol_types.Priority
    ClusterAllocation = scheduler_types.ClusterAllocation

    plan = ExecutionPlan()
    donor_id = "ft_000000000000"  # Has active workers, can donate
    receiver_id = "ft_111111111111"  # No pending request, no active allocation -> ineligible

    _GapRatioDPWorker = gap_ratio_mod._GapRatioDPWorker

    active_dp_workers = {
        donor_id: [_GapRatioDPWorker(pipeline_id=donor_id, dp_rank=0, gpu_ids=[0])],
        receiver_id: [],
    }
    inactive_dp_workers = {
        donor_id: [],
        receiver_id: [_GapRatioDPWorker(pipeline_id=receiver_id, dp_rank=0, gpu_ids=[0])],
    }

    pipeline_registry = {
        pid: {
            "cluster_configs": {
                "actor_infer": {
                    "tp_size": 1,
                    "is_generation": True,
                    "device_mapping": [0],
                    "max_dp_workers": 1,
                },
            },
        }
        for pid in [donor_id, receiver_id]
    }

    active_allocations = {
        f"{donor_id}_actor_infer": ClusterAllocation(
            cluster_id=f"{donor_id}_actor_infer",
            gpu_ids=[0],
            priority=Priority.GENERATION,
            active_dp_ranks={0},
            dp_rank_to_gpus={0: [0]},
        ),
    }

    def progress_totals_fn(*, pipeline_id):
        return (50.0, 100.0)

    gap_ratio_mod.plan_generation_gap_ratio(
        plan,
        active_dp_workers=active_dp_workers,
        inactive_dp_workers=inactive_dp_workers,
        non_gen_reserved_gpus=set(),
        idle_gpus=set(),  # No free GPUs — would need to donate
        pipeline_registry=pipeline_registry,
        active_allocations=active_allocations,
        pending_bucket_gen=[],  # No pending request for receiver
        progress_totals_fn=progress_totals_fn,
    )

    # No shrink ops should have been added — donor must not be mutated for an ineligible receiver
    assert len(plan.sched_guided_shrink_ops) == 0
    assert len(plan.sched_guided_allocation_ops) == 0


def test_snapshot_fails_fast_when_actor_infer_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """snapshot_generation_dp_workers must raise KeyError for a registered pipeline missing actor_infer."""
    gap_ratio_mod, scheduler_types, protocol_types = _load_gap_ratio_modules(monkeypatch)

    ExecutionPlan = scheduler_types.ExecutionPlan
    plan = ExecutionPlan()

    pipeline_registry = {
        "ft_000000000000": {
            "cluster_configs": {
                "actor_train": {"tp_size": 1, "device_mapping": [0, 1]},
                # actor_infer intentionally missing
            },
        }
    }

    with pytest.raises(KeyError, match="missing actor_infer"):
        gap_ratio_mod.snapshot_generation_dp_workers(
            plan=plan,
            idle_gpus={0, 1},
            pipeline_registry=pipeline_registry,
            active_allocations={},
        )


def test_alloc_step_target_estimate_fallback_revives_starved_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a GENERATION cluster with active_dp_ranks=set(), no pending request,
    and no progress reports must still be picked up by gap-ratio if its allocation
    carries a ``step_target_estimate`` snapshotted at grant time.

    This is the cross-pipeline full-overlap deadlock: P1 was granted GEN, then
    transiently shrunk to zero by P2's INITIALIZATION preempt; P2's wakeup
    re-enters gap-ratio, P1 must still get a share of the budget back.
    """
    gap_ratio_mod, scheduler_types, protocol_types = _load_gap_ratio_modules(monkeypatch)

    ExecutionPlan = scheduler_types.ExecutionPlan
    ClusterAllocation = scheduler_types.ClusterAllocation
    Priority = protocol_types.Priority
    Request = scheduler_types.Request
    PendingRequest = scheduler_types.PendingRequest

    plan = ExecutionPlan()

    # P1 — starved: GEN alloc exists but has been shrunk to zero, no pending, no progress.
    p1_id = "ft_aaaaaaaaaaaa"
    p1_cluster = f"{p1_id}_actor_infer"

    # P2 — fresh pending GEN request (just finished its INIT).
    p2_id = "ft_bbbbbbbbbbbb"
    p2_cluster = f"{p2_id}_actor_infer"

    _GapRatioDPWorker = gap_ratio_mod._GapRatioDPWorker
    active_dp_workers = {p1_id: [], p2_id: []}
    inactive_dp_workers = {
        p1_id: [
            _GapRatioDPWorker(pipeline_id=p1_id, dp_rank=0, gpu_ids=[0]),
            _GapRatioDPWorker(pipeline_id=p1_id, dp_rank=1, gpu_ids=[1]),
            _GapRatioDPWorker(pipeline_id=p1_id, dp_rank=2, gpu_ids=[2]),
            _GapRatioDPWorker(pipeline_id=p1_id, dp_rank=3, gpu_ids=[3]),
        ],
        p2_id: [
            _GapRatioDPWorker(pipeline_id=p2_id, dp_rank=0, gpu_ids=[0]),
            _GapRatioDPWorker(pipeline_id=p2_id, dp_rank=1, gpu_ids=[1]),
            _GapRatioDPWorker(pipeline_id=p2_id, dp_rank=2, gpu_ids=[2]),
            _GapRatioDPWorker(pipeline_id=p2_id, dp_rank=3, gpu_ids=[3]),
        ],
    }

    pipeline_registry = {
        p1_id: {
            "cluster_configs": {
                "actor_infer": {
                    "tp_size": 1,
                    "is_generation": True,
                    "device_mapping": [0, 1, 2, 3],
                    "max_dp_workers": 4,
                },
            },
            "admitted": True,
        },
        p2_id: {
            "cluster_configs": {
                "actor_infer": {
                    "tp_size": 1,
                    "is_generation": True,
                    "device_mapping": [0, 1, 2, 3],
                    "max_dp_workers": 4,
                },
            },
            "admitted": True,
        },
    }

    # P1 allocation persists at GENERATION priority but with no active dp_ranks
    # (peer's INITIALIZATION preempt shrunk all four). The step_target_estimate
    # was snapshotted at original grant time.
    active_allocations = {
        p1_cluster: ClusterAllocation(
            cluster_id=p1_cluster,
            gpu_ids=[],
            priority=Priority.GENERATION,
            active_dp_ranks=set(),
            dp_rank_to_gpus={},
            step_target_estimate=8.0,
        ),
    }

    # Only P2 has a pending request.
    pending_bucket_gen = [
        PendingRequest(
            request=Request(cluster_id=p2_cluster, priority=Priority.GENERATION, timestamp=0.0),
            event=asyncio.Event(),
            step_target_estimate=8,
        )
    ]

    def progress_totals_fn(*, pipeline_id):
        return (0.0, 0.0)

    remaining_idle = gap_ratio_mod.plan_generation_gap_ratio(
        plan,
        active_dp_workers=active_dp_workers,
        inactive_dp_workers=inactive_dp_workers,
        non_gen_reserved_gpus=set(),
        idle_gpus={0, 1, 2, 3},
        pipeline_registry=pipeline_registry,
        active_allocations=active_allocations,
        pending_bucket_gen=pending_bucket_gen,
        progress_totals_fn=progress_totals_fn,
    )

    # Must have produced an allocation op for BOTH pipelines. The starved P1
    # must get at least one DP worker; that is the fix for the full-overlap
    # deadlock (see runs/run3.log in rlix-miles-4ppl-partial-overlap-run/).
    cluster_ids = {op.cluster_id for op in plan.sched_guided_allocation_ops}
    assert p1_cluster in cluster_ids, (
        "P1 must get a GEN worker back via the alloc.step_target_estimate "
        "fallback even though it has no pending request and no progress reports"
    )
    assert p2_cluster in cluster_ids

    # H3 (review fix): pin the fallback_cap invariant. A regressed cap that
    # let P1 grab 2+ bundles instead of 1 would silently re-introduce the
    # Layer-2 OOM (post-train re-expand collides with non-GEN residual).
    # max(active=0, 1) * tp_size=1 = 1 bundle = exactly one dp_rank for P1.
    p1_ops = [op for op in plan.sched_guided_allocation_ops if op.cluster_id == p1_cluster]
    assert len(p1_ops) == 1, f"P1 must produce exactly one allocation op, got {len(p1_ops)}"
    p1_dp_ranks_added = sum(len(op.dp_rank_to_gpus_to_add) for op in p1_ops)
    assert p1_dp_ranks_added == 1, (
        f"P1 fallback must respect cap=max(active=0,1)*tp_size=1; "
        f"got {p1_dp_ranks_added} dp_ranks added — fallback_cap regressed"
    )


def test_alloc_step_target_estimate_fallback_ignored_when_pending_or_progress_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pending request must take precedence over the alloc snapshot, AND
    the pipeline must NOT be marked is_fallback in that case.

    H3 review fix: distinguishing pending-vs-alloc-snapshot at the assertion
    level requires a setup where the two branches produce different planner
    decisions. With ``is_fallback=False`` (pending path) there is no
    ``fallback_cap``; with ``is_fallback=True`` (alloc-snapshot path) the cap
    clamps the target to ``max(active, 1) * tp_size = 1``.

    Setup: 4 inactive workers + 4 idle GPUs + single pipeline.
    - pending=1 path: target = floor(1.0 * 4 / 1) = 4. Without cap, the
      planner activates all 4.
    - alloc-snapshot=999 fallback path: ``fallback_cap = 1``. The planner
      activates only 1.
    Asserting the planner activated 4 workers (one op per activation under
    the iterative loop) pins that the pending estimate won and the fallback
    flag stayed False.
    """
    gap_ratio_mod, scheduler_types, protocol_types = _load_gap_ratio_modules(monkeypatch)

    ExecutionPlan = scheduler_types.ExecutionPlan
    ClusterAllocation = scheduler_types.ClusterAllocation
    Priority = protocol_types.Priority
    Request = scheduler_types.Request
    PendingRequest = scheduler_types.PendingRequest

    plan = ExecutionPlan()
    pid = "ft_cccccccccccc"
    cluster = f"{pid}_actor_infer"

    _GapRatioDPWorker = gap_ratio_mod._GapRatioDPWorker
    active_dp_workers = {pid: []}
    inactive_dp_workers = {
        pid: [
            _GapRatioDPWorker(pipeline_id=pid, dp_rank=0, gpu_ids=[0]),
            _GapRatioDPWorker(pipeline_id=pid, dp_rank=1, gpu_ids=[1]),
            _GapRatioDPWorker(pipeline_id=pid, dp_rank=2, gpu_ids=[2]),
            _GapRatioDPWorker(pipeline_id=pid, dp_rank=3, gpu_ids=[3]),
        ]
    }
    pipeline_registry = {
        pid: {
            "cluster_configs": {
                "actor_infer": {
                    "tp_size": 1,
                    "is_generation": True,
                    "device_mapping": [0, 1, 2, 3],
                    "max_dp_workers": 4,
                },
            },
            "admitted": True,
        }
    }
    # Existing alloc carries a STALE estimate (999) that, if read via the
    # fallback path, would be capped to 1 bundle. The pending carries a
    # smaller value (1) but is the source-of-truth signal; under the pending
    # path the planner activates all 4 workers because the target_ratio is
    # 1.0 and total_gen_budget_gpus = 4.
    active_allocations = {
        cluster: ClusterAllocation(
            cluster_id=cluster,
            gpu_ids=[],
            priority=Priority.GENERATION,
            step_target_estimate=999.0,
        ),
    }
    pending_bucket_gen = [
        PendingRequest(
            request=Request(cluster_id=cluster, priority=Priority.GENERATION, timestamp=0.0),
            event=asyncio.Event(),
            step_target_estimate=1,
        )
    ]

    def progress_totals_fn(*, pipeline_id):
        return (0.0, 0.0)

    gap_ratio_mod.plan_generation_gap_ratio(
        plan,
        active_dp_workers=active_dp_workers,
        inactive_dp_workers=inactive_dp_workers,
        non_gen_reserved_gpus=set(),
        idle_gpus={0, 1, 2, 3},
        pipeline_registry=pipeline_registry,
        active_allocations=active_allocations,
        pending_bucket_gen=pending_bucket_gen,
        progress_totals_fn=progress_totals_fn,
    )

    # Pin the precedence: pending path produces 4 activations; fallback path
    # (capped at 1) would produce only 1. Anything other than 4 means the
    # pending estimate did NOT win.
    total_dp_ranks_added = sum(
        len(op.dp_rank_to_gpus_to_add) for op in plan.sched_guided_allocation_ops
    )
    assert total_dp_ranks_added == 4, (
        f"Pending estimate must override stale alloc snapshot; expected 4 dp_ranks "
        f"activated (uncapped pending path), got {total_dp_ranks_added}. If this is "
        f"1, the planner regressed to the fallback path while a pending was present."
    )


def test_fallback_no_expand_when_no_peer_has_fresh_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a fallback'd GENERATION cluster must NOT expand into idle GPUs
    when no peer pipeline has a fresh signal (pending request or progress).

    The post-``_after_training`` scheduling cycle releases ACTOR_TRAINING and
    immediately re-enters the planner. Under miles' ``MILES_SKIP_TMS_PAUSE=1``
    workaround on blackwell/cu12.9, the just-released GPUs still hold ~5–10 GB
    of train residual that SGLang's ``resume_memory_occupation`` would collide
    with. Both pipelines being fallback-only (no pending, no progress) is the
    signal that no peer needs the budget — there is nobody to compete with, so
    the fallback's "prevent peer-starvation" job doesn't apply and it should
    stay at zero active workers.
    """
    gap_ratio_mod, scheduler_types, protocol_types = _load_gap_ratio_modules(monkeypatch)

    ExecutionPlan = scheduler_types.ExecutionPlan
    ClusterAllocation = scheduler_types.ClusterAllocation
    Priority = protocol_types.Priority

    plan = ExecutionPlan()

    p1_id = "ft_dddddddddddd"
    p1_cluster = f"{p1_id}_actor_infer"
    p2_id = "ft_eeeeeeeeeeee"
    p2_cluster = f"{p2_id}_actor_infer"

    _GapRatioDPWorker = gap_ratio_mod._GapRatioDPWorker
    active_dp_workers = {p1_id: [], p2_id: []}
    inactive_dp_workers = {
        p1_id: [
            _GapRatioDPWorker(pipeline_id=p1_id, dp_rank=0, gpu_ids=[0]),
            _GapRatioDPWorker(pipeline_id=p1_id, dp_rank=1, gpu_ids=[1]),
            _GapRatioDPWorker(pipeline_id=p1_id, dp_rank=2, gpu_ids=[2]),
            _GapRatioDPWorker(pipeline_id=p1_id, dp_rank=3, gpu_ids=[3]),
        ],
        p2_id: [
            _GapRatioDPWorker(pipeline_id=p2_id, dp_rank=0, gpu_ids=[0]),
            _GapRatioDPWorker(pipeline_id=p2_id, dp_rank=1, gpu_ids=[1]),
            _GapRatioDPWorker(pipeline_id=p2_id, dp_rank=2, gpu_ids=[2]),
            _GapRatioDPWorker(pipeline_id=p2_id, dp_rank=3, gpu_ids=[3]),
        ],
    }
    pipeline_registry = {
        p1_id: {
            "cluster_configs": {
                "actor_infer": {
                    "tp_size": 1,
                    "is_generation": True,
                    "device_mapping": [0, 1, 2, 3],
                    "max_dp_workers": 4,
                },
            },
            "admitted": True,
        },
        p2_id: {
            "cluster_configs": {
                "actor_infer": {
                    "tp_size": 1,
                    "is_generation": True,
                    "device_mapping": [0, 1, 2, 3],
                    "max_dp_workers": 4,
                },
            },
            "admitted": True,
        },
    }

    # Both pipelines: alloc at GENERATION, active=set() (train just preempted
    # them), step_target_estimate snapshotted earlier. No pending requests.
    active_allocations = {
        p1_cluster: ClusterAllocation(
            cluster_id=p1_cluster,
            gpu_ids=[],
            priority=Priority.GENERATION,
            active_dp_ranks=set(),
            dp_rank_to_gpus={},
            step_target_estimate=8.0,
        ),
        p2_cluster: ClusterAllocation(
            cluster_id=p2_cluster,
            gpu_ids=[],
            priority=Priority.GENERATION,
            active_dp_ranks=set(),
            dp_rank_to_gpus={},
            step_target_estimate=8.0,
        ),
    }

    # No pending GEN requests anywhere — both pipelines are fallback-only.
    pending_bucket_gen: list = []

    def progress_totals_fn(*, pipeline_id):
        return (0.0, 0.0)

    gap_ratio_mod.plan_generation_gap_ratio(
        plan,
        active_dp_workers=active_dp_workers,
        inactive_dp_workers=inactive_dp_workers,
        non_gen_reserved_gpus=set(),
        idle_gpus={0, 1, 2, 3},
        pipeline_registry=pipeline_registry,
        active_allocations=active_allocations,
        pending_bucket_gen=pending_bucket_gen,
        progress_totals_fn=progress_totals_fn,
    )

    # No activation: both pipelines are fallback, no peer has fresh signal,
    # so no engine wake is issued — the trailing OOM on blackwell/cu12.9 is
    # avoided.
    assert plan.sched_guided_allocation_ops == [], (
        "Fallback pipelines must not expand when no peer has a fresh signal: "
        f"got {plan.sched_guided_allocation_ops!r}"
    )

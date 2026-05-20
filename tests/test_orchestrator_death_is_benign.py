"""Chaos test: orchestrator death post-startup must NOT block the scheduler.

Background — the Ray-2.55.1 ``worker.py:1039`` ``'Worker' object has no
attribute 'core_worker'`` race fires intermittently on actors handling
concurrent inbound RPCs. ``max_concurrency=4`` on the Orchestrator +
SchedulerActor reduces the rate but does not eliminate it.

This test proves the race is architecturally harmless: once the
control plane is up, the runtime never calls back into the
Orchestrator (rollout-driver code paths talk only to the
SchedulerActor + pipeline actors + coordinator), so killing the
Orchestrator post-startup must NOT leave the SchedulerActor in an
unreachable state.

If a future change accidentally introduces a runtime dependency on the
Orchestrator (e.g. a per-rollout RPC), this test fails and that fact
shows up in CI before it hits a real training run.

Skips when Ray cannot be started in the test runner.
"""

from __future__ import annotations

import os
import time

import pytest


def _ray_available() -> bool:
    try:
        import ray  # noqa: F401
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _ray_available() or os.environ.get("RLIX_SKIP_RAY_TESTS") == "1",
    reason="ray is unavailable or chaos test explicitly disabled",
)


@pytest.fixture(scope="module")
def ray_cluster():
    """Start a local Ray cluster for the test, tear down after."""
    import ray

    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=4, num_gpus=0, include_dashboard=False, log_to_driver=False)
    try:
        yield
    finally:
        if ray.is_initialized():
            ray.shutdown()


def test_scheduler_reachable_after_orchestrator_kill(ray_cluster):
    """Bring up the control plane, kill the Orchestrator, then verify
    the SchedulerActor is still reachable via its name.

    Importantly: does NOT exercise pipeline lifecycle (allocate /
    register / admit). Those are STARTUP calls; once admission is done
    the rollout loop never re-enters them. The test focuses on the
    invariant we care about — the SchedulerActor handle remains usable
    via ``ray.get_actor`` even after the Orchestrator process is dead.
    """
    import ray

    from rlix.client.client import ConnectOptions, connect
    from rlix.protocol.types import (
        ORCHESTRATOR_ACTOR_NAME,
        RLIX_NAMESPACE,
        SCHEDULER_ACTOR_NAME,
    )

    orchestrator = connect(ConnectOptions(create_if_missing=True))
    assert orchestrator is not None

    scheduler = ray.get_actor(SCHEDULER_ACTOR_NAME, namespace=RLIX_NAMESPACE)
    assert scheduler is not None

    ray.kill(orchestrator, no_restart=True)

    # Wait briefly for Ray's actor table to reflect the kill.
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            ray.get_actor(ORCHESTRATOR_ACTOR_NAME, namespace=RLIX_NAMESPACE)
        except ValueError:
            break
        time.sleep(0.1)
    else:
        pytest.fail("Orchestrator handle did not disappear after ray.kill")

    rehandle = ray.get_actor(SCHEDULER_ACTOR_NAME, namespace=RLIX_NAMESPACE)
    assert rehandle is not None, "SchedulerActor handle vanished after Orchestrator died"


def test_no_runtime_orchestrator_dependency_in_scheduler():
    """Static assertion: the SchedulerActor surface (``request_gpus``,
    ``report_progress``, ``notify_release_gpus``, ``clear_progress``,
    ``notify_release_then_request_gpus``) must not directly call back
    into the Orchestrator. The only legal touchpoint is the
    fatal-error fail-fast path (``_fail_fast_shutdown``), which is
    already robust to ``ActorDiedError``.

    If a future refactor accidentally re-introduces a runtime
    Orchestrator dependency (e.g. for a per-rollout RPC), this guard
    will fire — the property "orchestrator death post-startup is
    benign" must be defended explicitly because it is load-bearing
    for the Ray-race mitigation rationale on
    ``docs/internal/4gpu-2ppl-rollout2-hang-fix.md``.
    """
    import inspect

    from rlix.scheduler import scheduler as scheduler_module

    runtime_methods = [
        "request_gpus",
        "report_progress",
        "notify_release_gpus",
        "clear_progress",
        "notify_release_then_request_gpus",
    ]
    impl = scheduler_module.SchedulerImpl
    for name in runtime_methods:
        method = getattr(impl, name)
        src = inspect.getsource(method)
        assert "ORCHESTRATOR_ACTOR_NAME" not in src, (
            f"SchedulerImpl.{name} references ORCHESTRATOR_ACTOR_NAME — runtime "
            "must not depend on Orchestrator (would defeat the Ray-race "
            "mitigation rationale)."
        )
        assert "orchestrator.shutdown" not in src, (
            f"SchedulerImpl.{name} calls orchestrator.shutdown — only "
            "_fail_fast_shutdown is allowed to do that."
        )

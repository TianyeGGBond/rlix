from __future__ import annotations

import enum
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ModelMode(str, Enum):
    FULL_FT = "FULL_FT"
    MULTI_LORA = "MULTI_LORA"


@dataclass(frozen=True, slots=True)
class PipelineId:
    value: str


@dataclass(frozen=True, slots=True)
class ClusterId:
    value: str


@dataclass(frozen=True, slots=True)
class AdapterId:
    value: str


@dataclass(frozen=True, slots=True)
class ActionResponse:
    success: bool
    error: Optional[str] = None


@dataclass(frozen=True, slots=True)
class PlatformConfig:
    ray_device_key: str
    device_control_env_var: str


@dataclass(frozen=True, slots=True)
class ReleaseReport:
    dp_rank: int
    gpu_map: List[int]
    free_bytes_by_gpu: List[int]
    total_bytes_by_gpu: List[int]


@dataclass(frozen=True, slots=True)
class ReleaseAck:
    aborted: int
    remapped: int
    release_reports: List[ReleaseReport]


class Priority(enum.IntEnum):
    """7-tier priority system for GPU allocation (lower numeric value = higher priority)."""

    INITIALIZATION = 0
    ACTOR_TRAINING = 1
    CRITIC_TRAINING = 2
    OLD_LOG_PROBS = 3
    REF_LOG_PROBS = 4
    VALUE_COMPUTE = 5
    GENERATION = 6


@dataclass(frozen=True, slots=True)
class ProgressReport:
    pipeline_id: str
    queued_trajectories: int
    inflight_trajectories: int
    step_target_trajectories: int
    percent_completed: float = 0.0
    oldest_unfinished_creation_ts: Optional[float] = None
    active_base_version: int = 0
    fifo_timestamp: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None


@dataclass(frozen=True, slots=True)
class SchedRLTimeouts:
    register_timeout_secs: float = -1
    admit_timeout_secs: float = -1
    shrink_timeout_secs: float = -1
    expand_timeout_secs: float = -1
    abort_ack_timeout_secs: float = -1
    offload_timeout_secs: float = -1
    abort_timeout_secs: float = -1


@dataclass(frozen=True, slots=True)
class SchedRLConfig:
    fail_fast_on_restart: bool = True
    timeouts: SchedRLTimeouts = SchedRLTimeouts()


@dataclass(frozen=True, slots=True)
class RayNamespaceContract:
    pipeline_id_env_var: str = "PIPELINE_ID"
    roll_namespace_env_var: str = "ROLL_RAY_NAMESPACE"

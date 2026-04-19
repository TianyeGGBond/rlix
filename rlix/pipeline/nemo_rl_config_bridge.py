"""Config bridge from NeMo RL config to the RLix coordinator expected format.

PipelineCoordinator reads specific attributes from pipeline_config during
validation (_validate_config_schema, _validate_vllm_sleep_level,
_validate_offload_nccl) and actor creation. All required attributes must be
present or those validators silently fail or AttributeError on startup.

Usage (driver script):
    bridge = NemoRLConfigBridge.from_nemo_config(nemo_master_config)
    orchestrator.register_pipeline(
        pipeline_id, bridge.ray_namespace,
        bridge.cluster_tp_configs,
        bridge.cluster_device_mappings,
    )
    coordinator = PipelineCoordinator(pipeline_id=pipeline_id,
                                      pipeline_config=bridge)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class _StrategyConfig:
    """Minimal strategy_config container expected by _validate_vllm_sleep_level."""

    sleep_level: int = 2

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class _StrategyArgs:
    """Minimal strategy_args container expected by coordinator validators."""

    strategy_name: str = "vllm"
    strategy_config: _StrategyConfig = field(default_factory=_StrategyConfig)


@dataclass
class _ClusterConfig:
    """Minimal per-cluster config expected by coordinator validators."""

    name: str
    device_mapping: List[int]
    offload_nccl: bool = True
    strategy_args: Optional[_StrategyArgs] = None
    system_envs: Dict[str, str] = field(default_factory=dict)


class NemoRLConfigBridge:
    """Translates NeMo RL config into the shape PipelineCoordinator expects.

    Required attributes surfaced by this bridge:
        actor_train              _ClusterConfig for Megatron training cluster
        actor_infer              _ClusterConfig for vLLM inference cluster
        num_gpus_per_node        int, forwarded to RollResourceManagerProxy
        pipeline_cls             dotted path to NemoRLFullFinetunePipeline
        verify_model_after_sync  bool (default False)
        cluster_device_mappings  dict for orchestrator.register_pipeline()
        cluster_tp_configs       dict for orchestrator.register_pipeline()
    """

    def __init__(
        self,
        *,
        train_device_mapping: List[int],
        infer_device_mapping: List[int],
        vllm_tp_size: int,
        num_gpus_per_node: int,
        pipeline_cls: str = "rlix.pipeline.nemo_rl_pipeline.NemoRLFullFinetunePipeline",
        verify_model_after_sync: bool = False,
        extra_train_env: Optional[Dict[str, str]] = None,
        extra_infer_env: Optional[Dict[str, str]] = None,
        nemo_master_config: Optional[Any] = None,
    ) -> None:
        self._nemo_master_config = nemo_master_config
        self.num_gpus_per_node = num_gpus_per_node
        self.pipeline_cls = pipeline_cls
        self.verify_model_after_sync = verify_model_after_sync

        # actor_train: Megatron registers with tp_size=1 (each worker = 1 GPU).
        # Parallel dims (TP/PP/CP/EP) are handled via NCCL groups, not per-worker width.
        self.actor_train = _ClusterConfig(
            name="actor_train",
            device_mapping=list(train_device_mapping),
            offload_nccl=True,
            strategy_args=None,
            system_envs=dict(extra_train_env or {}),
        )

        # actor_infer: vLLM with sleep_level=2 (drop weights + KV cache on offload).
        self.actor_infer = _ClusterConfig(
            name="actor_infer",
            device_mapping=list(infer_device_mapping),
            offload_nccl=True,
            strategy_args=_StrategyArgs(
                strategy_name="vllm",
                strategy_config=_StrategyConfig(sleep_level=2),
            ),
            system_envs=dict(extra_infer_env or {}),
        )

    # ------------------------------------------------------------------
    # Registration helpers used by the driver script
    # ------------------------------------------------------------------

    @property
    def cluster_device_mappings(self) -> Dict[str, List[int]]:
        """GPU device indices per cluster, passed to orchestrator.register_pipeline()."""
        return {
            "actor_train": list(self.actor_train.device_mapping),
            "actor_infer": list(self.actor_infer.device_mapping),
        }

    @property
    def cluster_tp_configs(self) -> Dict[str, int]:
        """Per-cluster TP size for the scheduler's dp_size calculation.

        actor_train uses tp_size=1 because Megatron workers each own 1 GPU;
        parallelism is via NCCL groups, not per-worker GPU width.
        actor_infer uses the actual vLLM TP size.
        """
        vllm_tp = (
            self.actor_infer.strategy_args.strategy_config.sleep_level  # placeholder
            if self.actor_infer.strategy_args is None
            else 1  # will be overridden below
        )
        # Read actual vLLM TP from the raw NeMo config if available.
        if self._nemo_master_config is not None:
            vllm_tp = int(
                self._nemo_master_config
                .get("policy", {})
                .get("generation", {})
                .get("vllm_cfg", {})
                .get("tensor_parallel_size", 1)
            )
        return {
            "actor_train": 1,
            "actor_infer": vllm_tp,
        }

    @classmethod
    def from_nemo_config(
        cls,
        master_config: Any,
        *,
        pipeline_cls: str = "rlix.pipeline.nemo_rl_pipeline.NemoRLFullFinetunePipeline",
        verify_model_after_sync: bool = False,
    ) -> "NemoRLConfigBridge":
        """Construct bridge from a NeMo RL master_config dict.

        Reads cluster.gpus_per_node for num_gpus_per_node.
        Reads policy.generation.vllm_cfg.tensor_parallel_size for vllm_tp.
        train_device_mapping and infer_device_mapping must be passed separately
        (they come from the RLix placement group allocation, not the NeMo config).
        Call set_device_mappings() after construction.
        """
        num_gpus_per_node = int(
            master_config.get("cluster", {}).get("gpus_per_node", 1)
        )
        vllm_tp = int(
            master_config
            .get("policy", {})
            .get("generation", {})
            .get("vllm_cfg", {})
            .get("tensor_parallel_size", 1)
        )
        total_gpus = int(
            master_config.get("cluster", {}).get("num_nodes", 1)
        ) * num_gpus_per_node

        # Default: all GPUs for infer, last half for train (minimal partial overlap).
        # Caller should override via set_device_mappings() for custom topology.
        infer_devs = list(range(total_gpus))
        train_devs = list(range(total_gpus // 2, total_gpus))

        return cls(
            train_device_mapping=train_devs,
            infer_device_mapping=infer_devs,
            vllm_tp_size=vllm_tp,
            num_gpus_per_node=num_gpus_per_node,
            pipeline_cls=pipeline_cls,
            verify_model_after_sync=verify_model_after_sync,
            nemo_master_config=master_config,
        )

    def set_device_mappings(
        self,
        *,
        train_device_mapping: List[int],
        infer_device_mapping: List[int],
    ) -> None:
        """Override device mappings after construction (e.g. after PG allocation)."""
        self.actor_train.device_mapping = list(train_device_mapping)
        self.actor_infer.device_mapping = list(infer_device_mapping)

    def validate_partial_overlap(self) -> None:
        """Assert that topology satisfies partial overlap requirements (Feature 10).

        Raises AssertionError with a descriptive message on the first violation.
        """
        train_set = set(self.actor_train.device_mapping)
        infer_set = set(self.actor_infer.device_mapping)

        assert train_set.issubset(infer_set), (
            f"partial overlap requires train_devices ⊆ infer_devices. "
            f"train={sorted(train_set)} infer={sorted(infer_set)}"
        )

        vllm_tp = self.cluster_tp_configs["actor_infer"]
        infer_dp = len(infer_set) // max(vllm_tp, 1)
        assert infer_dp >= 2, (
            f"partial overlap requires infer_dp_size >= 2 (got {infer_dp}); "
            f"add more infer GPUs or reduce vllm_tp_size"
        )

        non_overlap = infer_set - train_set
        assert len(non_overlap) >= vllm_tp, (
            f"at least 1 full inference DP rank must stay active after shrink "
            f"(need >= {vllm_tp} non-overlap GPUs, got {len(non_overlap)})"
        )

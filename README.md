<div align="center">

# RLix

<h3>GPU time-sharing for concurrent LLM reinforcement learning.</h3>


<p>
  <a href="https://github.com/rlops/rlix/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
  </a>

  <a href="https://github.com/rlops/rlix/stargazers">
    <img src="https://img.shields.io/github/stars/rlops/rlix?style=social" alt="Repo stars">
  </a>

  <a href="https://github.com/rlops/rlix/issues">
    <img src="https://img.shields.io/github/issues/rlops/rlix" alt="GitHub issues">
  </a>

  <a href="https://deepwiki.com/rlops/rlix" target="_blank">
    <img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki">
  </a>

</p>

</div>



In agentic RL, long-horizon rollouts are increasingly long-tailed: a few stragglers dominate wall time, leaving most GPUs idle. RLix solves this by time-sharing GPUs across multiple concurrent RL training jobs, elastically scaling rollout workers onto idle GPUs and back. Training semantics remain unchanged: behavior is equivalent to resource-isolated training, just with better GPU utilization.

RLix builds on [ROLL](https://github.com/alibaba/ROLL)'s **Partial Overlapping** scheduling and extends it with a distributed control plane protocol for coordinating multiple independent training jobs on a shared GPU cluster.

## Features 

- **Recipe-Transparent Scheduling**: Per-pipeline training logic is fully independent from GPU scheduling; researchers program each pipeline in isolation
- **Two-Level GPU Sharing**: Across pipelines via elastic expand/shrink; within a pipeline via multi-LoRA adapters on a shared base model
- **Demand-Driven Rollout Scaling**: Rollout workers expand onto idle training GPUs and shrink back based on heartbeat-reported demand
- **Efficient Memory Management**: Model weights are cached on the trainer's CPU and synced on demand only to resumed rollout workers; on shrink, inference weights are fully dropped to minimize memory footprint.

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│                     RLix Control Plane                    │
├──────────────────┬──────────────────┬─────────────────────┤
│   Orchestrator   │    Scheduler     │  Resource Manager   │
│ (lifecycle mgmt) │ (priority-based) │   (GPU topology)    │
└────────┬─────────┴────────┬─────────┴─────────┬───────────┘
         │                  │                   │
    ┌────▼─────┐       ┌────▼─────┐        ┌────▼─────┐
    │FullFine- │       │Multi-LoRA│        │FullFine- │
    │tune Job 1│       │  Job 2   │        │tune Job N│
    │  (GRPO)  │       │  (PPO)   │        │  (PPO)   │
    └────┬─────┘       └────┬─────┘        └────┬─────┘
         │                  │                   │
    ┌────▼──────────────────▼───────────────────▼────┐
    │             Shared GPU Resources               │
    │  [GPU 0] [GPU 1] [GPU 2] [GPU 3] ... [GPU N]   │
    └────────────────────────────────────────────────┘
```

## Installation

```bash
git clone https://github.com/rlops/rlix.git
cd rlix
pip install -e .
```

## Quick Start

```python
import ray
import rlix
from rlix.pipeline import PipelineCoordinator
from rlix.protocol.types import COORDINATOR_ACTOR_NAME_PREFIX

# 1. Initialize the RLix control plane
orchestrator = rlix.init(create_if_missing=True)

# 2. Allocate a unique pipeline ID ("ft" for full finetune, "lora" for multi-LoRA)
pipeline_id = ray.get(orchestrator.allocate_pipeline_id.remote("ft"))

# 3. Register the pipeline's GPU topology
ray.get(orchestrator.register_pipeline.remote(
    pipeline_id=pipeline_id,
    ray_namespace=f"pipeline_{pipeline_id}_NS",
    cluster_tp_configs={"actor_train": 8, "actor_infer": 8},
    cluster_device_mappings={"actor_train": [0,1,2,3,4,5,6,7], "actor_infer": [0,1,2,3,4,5,6,7]},
))

# 4. Admit the pipeline (required before GPU allocation)
ray.get(orchestrator.admit_pipeline.remote(pipeline_id=pipeline_id))

# 5. Create the coordinator actor
CoordinatorActor = ray.remote(PipelineCoordinator)
coordinator = CoordinatorActor.options(
    name=f"{COORDINATOR_ACTOR_NAME_PREFIX}{pipeline_id}",
    namespace=f"pipeline_{pipeline_id}_NS",
).remote(pipeline_id=pipeline_id, pipeline_config=my_config)

# 6. Create and run the pipeline
pipeline_actor = ray.get(coordinator.create_pipeline_actor.remote(pipeline_config=my_config))
ray.get(pipeline_actor.run.remote())
```

See [examples/](examples/) for complete multi-pipeline configurations.

## Pipeline Types

### Full Finetune Pipeline (`RollFullFinetunePipeline`)

Full-parameter training with elastic GPU expand/shrink. Each job trains all model weights and releases GPUs to other jobs during idle stages.

### Multi-LoRA Pipeline (`RollMultiLoraPipeline`)

Multiple LoRA adapters trained concurrently on a shared base model, each with an isolated per-adapter optimizer. Jobs share the base model in GPU memory while keeping adapter weights and optimizer states fully independent.

## Scheduling Policy

The scheduler assigns GPUs by priority order (lower value = higher priority). All stages except rollout are non-preemptable: they hold GPUs until complete, then release them. Rollout (6) is lowest priority and always preemptable: it receives only the GPUs left over after all other stages are satisfied. When multiple jobs roll out concurrently, those leftover GPUs are split proportionally to each job's remaining rollout demand, subject to placement constraints.

- **0 Initialization**: Model loading, must complete before any scheduling
- **1 Actor Training**: Policy gradient update
- **2 Critic Training**: Value function update
- **3 Old Log Probs**: Log-probability computation under the previous policy
- **4 Reference Log Probs**: Log-probability computation under the reference model
- **5 Value Compute**: Value estimation for advantage calculation
- **6 Rollout**: Trajectory sampling (preemptable)

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.


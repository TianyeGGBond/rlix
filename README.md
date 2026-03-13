<div align="center">

# RLix

<h3>GPU time-sharing for concurrent LLM reinforcement learning.</h3>


<p>


  <a href="https://deepwiki.com/rlops/rlix" target="_blank">
    <img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki">
  </a>


  <a href="https://github.com/rlops/rlix/issues">
    <img src="https://img.shields.io/github/issues/rlops/rlix" alt="GitHub issues">
  </a>
  <a href="https://github.com/rlops/rlix/stargazers">
    <img src="https://img.shields.io/github/stars/rlops/rlix?style=social" alt="Repo stars">
  </a>
  <a href="https://github.com/rlops/rlix/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
  </a>
</p>

</div>



In agentic RL, long-horizon rollouts are increasingly long-tailed: a few stragglers dominate wall time, leaving training GPUs idle. RLix solves this by time-sharing GPUs across multiple concurrent RL training jobs, elastically scaling rollout workers onto idle training GPUs and back. Training semantics remain unchanged: behavior is equivalent to resource-isolated training, just with better GPU utilization.

RLix builds on [ROLL](https://github.com/alibaba/ROLL)'s **Partial Overlapping** scheduling and extends it with a distributed control plane protocol for coordinating multiple independent training jobs on a shared GPU cluster.

## Features 

- **Recipe-Transparent Scheduling**: Per-pipeline training logic is fully independent from GPU scheduling; researchers program each pipeline in isolation
- **Two-Level GPU Sharing**: Across pipelines via elastic expand/shrink; within a pipeline via multi-LoRA adapters on a shared base model
- **Progress-Driven Elastic Scaling**: Rollout workers expand onto idle training GPUs and shrink back based on heartbeat-reported demand
- **Sync-on-Resume**: Only re-activated rollout workers sync the latest weights, avoiding unnecessary broadcasts and memory pressure

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
from rlix.pipeline import PipelineCoordinator, RollFullFinetunePipeline, RollMultiLoraPipeline

# Initialize RLix control plane
orchestrator = rlix.init(create_if_missing=True)

# Create a coordinator for your pipeline
coordinator = PipelineCoordinator(
    pipeline_id="my_pipeline",
    pipeline_config=my_config,  # your Hydra pipeline config
)

# Create and run the pipeline
pipeline_actor = coordinator.create_pipeline_actor(pipeline_config=my_config)
ray.get(pipeline_actor.run.remote())
```

See [examples/](examples/) for runnable multi-pipeline configurations.

## Pipeline Types

### Full Finetune Pipeline (`RollFullFinetunePipeline`)

Full-parameter training with elastic GPU expand/shrink. Each job trains all model weights and releases GPUs to other jobs during idle stages.

### Multi-LoRA Pipeline (`RollMultiLoraPipeline`)

Multiple LoRA adapters trained concurrently on a shared base model, each with an isolated per-adapter optimizer. Jobs share the base model in GPU memory while keeping adapter weights and optimizer states fully independent.

## Scheduling Policy

When jobs compete for GPUs, the scheduler grants them to the job at the highest-priority stage (lower value = higher priority). Generation has the lowest priority (6) and is always preemptable: rollout workers are the first to yield GPUs when any other job needs them. Training stages (1-2) hold GPUs until the update completes, then release them.

- **0 Initialization**: Model loading, must complete before any scheduling
- **1 Actor Training**: Policy gradient update
- **2 Critic Training**: Value function update
- **3 Old Log Probs**: Log-probability computation under the previous policy
- **4 Reference Log Probs**: Log-probability computation under the reference model
- **5 Value Compute**: Value estimation for advantage calculation
- **6 Rollout**: Trajectory sampling (preemptable)

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.


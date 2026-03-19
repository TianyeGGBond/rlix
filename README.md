<div align="center">

<img src="assets/rlix-logo-text-horizontal.svg" width="40%" alt="RLix Logo">

<h3>Run more LLM RL experiments, wait less for GPUs</h3>

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

RL research often means running many experiments: testing new ideas, comparing configurations, and running ablations. But GPU clusters have limited capacity, so promising jobs can spend too much time waiting to start. This is especially frustrating in agentic RL, where a small number of slow multi-turn rollouts can delay the whole job while most GPUs sit idle.

RLix is a GPU cluster manager that lets multiple RL jobs share GPU capacity. When one job has spare capacity, RLix gives it to other jobs and takes it back when needed. More experiments can run at the same time, wait time drops, and GPU utilization goes up. RLix changes how GPU capacity is shared, not how each pipeline trains.

## Features

- **Support on-policy and off-policy pipelines**: RLix works with both, while preserving each pipeline's training behavior within its own staleness bounds.
- **Share GPU capacity across jobs**: Full-finetune pipelines can use idle GPU capacity from other jobs instead of waiting for dedicated resources.
- **Share one base model across LoRA adapters**: Multi-LoRA pipelines can further share one base model across multiple adapters within a pipeline.
- **Grow and shrink rollouts automatically**: Rollout workers expand and shrink based on current rollout demand and available GPU capacity.

## Installation

```bash
git clone https://github.com/rlops/rlix.git
cd rlix
pip install -e .
```

## Quick Start

The example below shows a minimal RLix setup for starting one pipeline under RLix management.

Workflow overview:

1. Initialize RLix and get the orchestrator.
2. Allocate a pipeline ID.
3. Register the pipeline's GPU layout and namespace.
4. Admit the pipeline so RLix can schedule it.
5. Create the pipeline coordinator.
6. Create the pipeline actor and run it.

```python
import ray
import rlix
from rlix.pipeline import PipelineCoordinator
from rlix.protocol.types import COORDINATOR_ACTOR_NAME_PREFIX

# Pipeline-specific configuration object
my_config = ...

# 1. Initialize RLix
orchestrator = rlix.init(create_if_missing=True)

# 2. Allocate a pipeline ID
pipeline_id = ray.get(orchestrator.allocate_pipeline_id.remote("ft"))

# 3. Register the pipeline's GPU topology
ray.get(
    orchestrator.register_pipeline.remote(
        pipeline_id=pipeline_id,
        ray_namespace=f"pipeline_{pipeline_id}_NS",
        cluster_tp_configs={"actor_train": 8, "actor_infer": 8},
        cluster_device_mappings={
            "actor_train": [0, 1, 2, 3, 4, 5, 6, 7],
            "actor_infer": [0, 1, 2, 3, 4, 5, 6, 7],
        },
    )
)

# 4. Admit the pipeline before GPU allocation
ray.get(orchestrator.admit_pipeline.remote(pipeline_id=pipeline_id))

# 5. Create the pipeline coordinator
CoordinatorActor = ray.remote(PipelineCoordinator)
coordinator = CoordinatorActor.options(
    name=f"{COORDINATOR_ACTOR_NAME_PREFIX}{pipeline_id}",
    namespace=f"pipeline_{pipeline_id}_NS",
).remote(pipeline_id=pipeline_id, pipeline_config=my_config)

# 6. Create and run the pipeline
pipeline_actor = ray.get(
    coordinator.create_pipeline_actor.remote(pipeline_config=my_config)
)
ray.get(pipeline_actor.run.remote())
```

See [examples/](examples/) for complete multi-pipeline examples and full configuration options.

## Pipeline Types

RLix currently supports two built-in pipeline types:

### Full Finetune Pipeline (`RollFullFinetunePipeline`)

Full-parameter training with elastic GPU expand and shrink. Each job trains all model weights while allowing idle GPU capacity to be used by other jobs.
Use this when you want to maximize model performance and have enough GPUs and memory for full finetuning, while still sharing spare GPU capacity across jobs.

### Multi-LoRA Pipeline (`RollMultiLoraPipeline`)

Concurrent training of multiple LoRA adapters on a shared base model, with a separate optimizer for each adapter. Jobs share the base model in GPU memory while keeping adapter weights and optimizer states independent.
Use this when you want much lower GPU and memory usage than full finetuning, or when you want to train multiple adapters on the same base model and further increase sharing within one pipeline.

RLix also supports custom pipelines and integrations that follow the RLix interface.

## Architecture

RLix has one shared cluster management layer and one coordinator for each pipeline. The shared layer manages GPU allocation across jobs, while each pipeline keeps its own training logic.

```text
┌───────────────────────────────────────────────────────────┐
│                RLix Shared Cluster Management Layer       │
├──────────────────┬──────────────────┬─────────────────────┤
│   Orchestrator   │    Scheduler     │  Resource Manager   │
│   (job lifecycle)│ (priorities +    │ (cluster resources) │
│                  │ rollout sharing) │                     │
└────────┬─────────┴────────┬─────────┴─────────┬───────────┘
         │                  │                   │
    ┌────▼─────┐       ┌────▼─────┐        ┌────▼─────┐
    │FullFine- │       │Multi-LoRA│        │Custom /  │
    │tune Job 1│       │  Job 2   │        │External  │
    │          │       │          │        │  Job N   │
    └────┬─────┘       └────┬─────┘        └────┬─────┘
         │                  │                   │
    ┌────▼──────────────────▼───────────────────▼────┐
    │               Shared GPU Capacity              │
    │   [GPU 0] [GPU 1] [GPU 2] [GPU 3] ... [GPU N]  │
    └────────────────────────────────────────────────┘
```

## How GPU Scheduling Works

RLix gives GPUs to higher-priority stages first. Most stages keep their GPUs until they finish. Rollout is the flexible stage: it can use spare GPU capacity when available and give it back when higher-priority work needs it.

Rollout has the lowest priority and is always preemptable. When multiple jobs are rolling out at the same time, RLix divides the available GPU capacity based on how much rollout work each job still has to do, while respecting placement constraints. To keep rollout workers lightweight, RLix loads inference weights only when they are active and releases them again when they shrink.

From highest to lowest priority:

* **0 Initialization**: Model loading; must complete before scheduling begins.
* **1 Actor Training**: Policy gradient update.
* **2 Critic Training**: Value function update.
* **3 Old-Policy Log Probs**: Log-probability computation under the previous policy.
* **4 Reference-Model Log Probs**: Log-probability computation under the reference model.
* **5 Value Compute**: Value estimation for advantage calculation.
* **6 Rollout**: Trajectory sampling; always preemptable.

## Acknowledgements

RLix was developed with extensive AI assistance, with human direction and oversight throughout.

RLix is inspired by **Partial Overlapping** scheduling from [**Alibaba/ROLL**](https://github.com/alibaba/ROLL).

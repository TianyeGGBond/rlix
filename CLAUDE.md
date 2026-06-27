# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Install in dev mode
pip install -e ".[dev]"

# Full environment setup (Linux with NVIDIA GPUs)
bash setup_env.sh && conda activate rlix

# Tests
pytest tests/                          # all tests
pytest tests/test_gap_ratio.py         # single file
pytest tests/test_gap_ratio.py -k "test_name"  # single test

# Linting & formatting
black rlix/ tests/ --line-length 119
ruff check rlix/ tests/
mypy rlix/
```

## Code Style

- Python 3.10+, line length 119 (black + ruff)
- mypy with `roll.*` imports ignored (`ignore_missing_imports`)
- ruff selects: E, F, I, W (ignores E501)

## Architecture

RLix is a Ray-based GPU time-sharing library that lets multiple RL training pipelines share GPU capacity. The system is built on Ray actors with a centralized scheduling model.

### Actor Hierarchy

```
Orchestrator (singleton: rlix:orchestrator, ns: rlix)
├── Scheduler (singleton: rlix:scheduler, ns: rlix)
│   └── ResourceManager (singleton: rlix:resource_manager, ns: rlix)
└── Per-pipeline (ns: pipeline_{pipeline_id}_NS)
    ├── PipelineCoordinator (rlix:coordinator:{pipeline_id})
    └── Pipeline Actor (rlix:pipeline:{pipeline_id})
```

### Core Flow

1. `rlix.init()` → creates Orchestrator, Scheduler, ResourceManager as Ray actors
2. Pipeline registers with Orchestrator → gets unique ID (`ft_` or `lora_` prefix + 12-char hex)
3. Pipeline admitted → Scheduler begins managing its GPU allocation
4. During training, pipeline requests GPUs by priority → Scheduler's gap-ratio planner decides allocation
5. GENERATION (rollout) workers elastically expand/shrink as GPUs become available or are reclaimed

### Key Design Decisions

- **Fail-fast, no recovery**: Scheduler/Orchestrator restart clears all state; pipelines must re-register
- **Single preemptable stage**: Only `Priority.GENERATION` (rollout) can be preempted; training stages (ACTOR_TRAINING, CRITIC_TRAINING, etc.) are non-interruptible
- **7-tier priority system**: `Priority(IntEnum)` in `protocol/types.py` — INITIALIZATION(0) through GENERATION(6)
- **Lazy imports**: `rlix/__init__.py` uses `__getattr__` to defer ROLL imports and avoid circular deps with heavy ML libraries
- **Cluster ID format**: `{pipeline_id}_{cluster_name}` — validated with regex + TP-size alignment

### Module Responsibilities

- **`orchestrator/`** — Pipeline lifecycle (allocate ID, register topology, admit, kill)
- **`scheduler/`** — Central GPU arbitration. `planner.py` implements the gap-ratio DP algorithm; `state.py` tracks pending requests + active allocations; `validation.py` checks execution plan invariants
- **`pipeline/`** — Per-pipeline actors. Two backends:
  - **ROLL backend**: `coordinator.py` + `full_finetune_pipeline.py` + `multi_lora_pipeline.py`
  - **MILES backend**: `miles_coordinator.py` + `miles_pipeline.py` + `miles_hooks.py`
- **`protocol/`** — Shared types and the `Coordinator` abstract interface that both backends implement
- **`utils/`** — Ray actor helpers (`ray.py`) and env var parsing (`env.py`)

### Cluster Name Constants

Defined in `protocol/types.py`:
- `ACTOR_TRAIN_CLUSTER_NAME = "actor_train"`
- `GENERATION_CLUSTER_NAME = "actor_infer"`
- `CRITIC_CLUSTER_NAME = "critic"` / `REFERENCE_CLUSTER_NAME = "reference"`
- `REWARD_CLUSTER_NAME = "reward"` (CPU-only, never scheduled)

### Concurrency Model

- **Scheduler**: async (`asyncio.Event` on `PendingRequest` for non-blocking GPU request signaling)
- **PipelineCoordinator**: sync with `max_concurrency=4`, `_resize_sync_lock` guards weight sync (180s timeout)
- **Pipeline actors**: sync with `max_concurrency=32` for overlapping resize + run calls

## Dependencies

- **Ray** — distributed actor framework
- **ROLL** (`git+https://github.com/rlops/ROLL.git`) — Alibaba's RL framework; provides `AgenticPipeline` base class
- **tg4perfetto** — GPU utilization tracing

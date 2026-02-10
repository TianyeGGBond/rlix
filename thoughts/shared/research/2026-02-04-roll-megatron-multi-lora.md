---
date: 2026-02-04T18:10:51-05:00
researcher: Droid
git_commit: 4ce7520e2cb20cc32dbfee71f5abca754a2c99ab
branch: main
repository: SchedRL
topic: "ROLL (recent) Megatron backend multi-LoRA + vLLM backend-only + offload/shrink-expand mechanisms"
tags: [research, roll, megatron, lora, vllm, offload, sequence-packing]
status: in_progress
last_updated: 2026-02-04
last_updated_by: Droid
---

# Research: ROLL Megatron backend multi-LoRA (recent commit focus)

## Research Question
Map the current implementation in `third_party/ROLL` for an agentic RL pipeline using Megatron + vLLM backend-only, with a focus on:
- SchedRL-like scheduling/offloading logic
- shrink/expand mechanisms (memory/placement)
- Megatron backend multi-LoRA support (multiple adapters active in one forward pass)
- vLLM backend-only inference integration

## Summary (so far)
- Multi-LoRA concurrency at *forward time* is implemented in the Megatron LoRA layer wrapper (`LoraParallelLinear`) by iterating over `self.active_adapters` and summing each adapter's contribution into the base layer output.
- Megatron LoRA is wired through ROLL model providers: when Megatron backend is used, it calls `apply_megatron_lora()` (patches TE layers and LoRA dispatch) and wraps the Megatron model via PEFT `get_peft_model(..., LoraConfig(...))`.
- Example YAMLs for Megatron training with LoRA live under `third_party/ROLL_multi_pipeline/examples/...` and set `actor_train.strategy_args.strategy_name: megatron_train` plus `actor_train.model_args.lora_*` fields.

## Detailed Findings

### 1) Multi-LoRA in Megatron backend (multiple adapters active concurrently)
**Layer implementation**
- File: `third_party/ROLL/mcore_adapter/src/mcore_adapter/adapters/lora_layer.py`
- Class: `LoraParallelLinear`
- Forward behavior:
  - Runs the base Transformer-Engine linear/router layer.
  - If adapters are enabled and not merged, it loops over `self.active_adapters` and adds each adapter's LoRA output to `result`.
  - The same `active_adapters` loop exists for router gating patch path (`TopKRouter`).

Key code locations:
- `lora_layer.py:181+` — router gating path loops `for active_adapter in self.active_adapters:`
- `lora_layer.py:230+` — standard linear path loops `for active_adapter in self.active_adapters:`

**Adapter creation / dispatch**
- File: `third_party/ROLL/mcore_adapter/src/mcore_adapter/adapters/lora_layer.py`
- Function: `apply_megatron_lora()`
  - Patches TE layer repr/state_dict helpers.
  - Sets `peft.tuners.lora.model.dispatch_megatron = dispatch_megatron` so PEFT will create `Lora*ParallelLinear` wrappers for TE layers.

### 2) How ROLL consumes YAML LoRA config for Megatron backend
**Config fields**
- File: `third_party/ROLL/roll/configs/model_args.py`
  - `ModelArguments` includes LoRA fields (`lora_target`, `lora_rank`, `lora_alpha`, `lora_dropout`, `additional_target`).
  - `__post_init__` splits `lora_target` on commas when not regex-like.

**Megatron model provider wiring**
- File: `third_party/ROLL/roll/models/model_providers.py`
- Function: `default_actor_model_provider(...)`
  - When `training_args` is `mcore_adapter.TrainingArguments` (Megatron path):
    - Loads model via `mcore_adapter.models.AutoModel.from_pretrained(...)`
    - If `model_args.lora_target` is set:
      - calls `apply_megatron_lora()`
      - `set_linear_is_expert(model[0])`
      - wraps model with PEFT using `setup_lora_training(..., is_mca=True)`

- Function: `setup_lora_training(...)`
  - Builds `target_modules` from `model_args.lora_target` with helper expansions (e.g. `all-linear`).
  - Calls `get_peft_model(model, LoraConfig(**lora_config), autocast_adapter_dtype=...)`.

### 3) Megatron+LoRA example YAMLs
**Main Megatron LoRA example**
- File: `third_party/ROLL_multi_pipeline/examples/qwen3-30BA3B-rlvr_megatron/rlvr_config_lora.yaml`
- Relevant fields:
  - `actor_train.model_args.lora_target: all-linear`
  - `actor_train.model_args.lora_rank: 64`
  - `actor_train.model_args.lora_alpha: 64`
  - `actor_train.strategy_args.strategy_name: megatron_train`
  - `actor_infer.strategy_args.strategy_name: vllm` (backend-only inference) and also includes LoRA fields under `actor_infer.model_args`.

**Megatron non-LoRA example**
- File: `third_party/ROLL_multi_pipeline/examples/qwen2.5-vl-7B-rlvr/rlvr_megatron.yaml`
  - Uses `actor_train.strategy_args.strategy_name: megatron_train` without an explicit LoRA block.

### 4) vLLM backend-only inference integration
- File: `third_party/ROLL/roll/distributed/strategy/vllm_strategy.py`
- Class: `VllmStrategy`
  - Creates async vLLM engine via `roll.third_party.vllm.create_async_llm(...)`.
  - Supports LoRA via vLLM engine (`enable_lora`, `max_loras`, `add_lora(...)`) and uses `LoRARequest` during generation.
  - Provides `load_states()` / `offload_states()` for colocated actor-infer mode.

### 5) Offload / shrink-expand style mechanisms
**Generic tensor offload utilities**
- File: `third_party/ROLL/roll/utils/offload_states.py`
  - Implements flat CPU buffers for parameters and provides `offload_module` / `reload_module`.

**Megatron optimizer/model state offload patch**
- File: `third_party/ROLL/roll/third_party/megatron/offload_states_patch.py`
  - Monkey patches Megatron optimizers with `offload_states` / `reload_states`.
  - Supports offloading model params, optimizer states, and other params.

## Code References
- `third_party/ROLL/mcore_adapter/src/mcore_adapter/adapters/lora_layer.py:181-260` — loops over `self.active_adapters` and sums LoRA outputs.
- `third_party/ROLL/roll/models/model_providers.py` — Megatron path calls `apply_megatron_lora()` + `setup_lora_training(..., is_mca=True)`.
- `third_party/ROLL/roll/configs/model_args.py` — LoRA args parsing (`lora_target`, `lora_rank`, `lora_alpha`, ...).
- `third_party/ROLL_multi_pipeline/examples/qwen3-30BA3B-rlvr_megatron/rlvr_config_lora.yaml` — example Megatron LoRA training config.
- `third_party/ROLL/roll/distributed/strategy/vllm_strategy.py` — vLLM backend-only inference + LoRA hooks + offload.
- `third_party/ROLL/roll/third_party/megatron/offload_states_patch.py` — Megatron offload implementation.

## Open Questions
- Where/how `active_adapters` is set to contain multiple adapter names during Megatron training runs (config surface / runtime selection path).
- Whether any pipeline code in `roll/pipeline/*` explicitly manages multi-adapter activation, or if it is purely backend/PEFT-driven.

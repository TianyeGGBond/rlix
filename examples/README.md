# RLix Examples

This directory contains examples for using RLix (rlix) for multi-pipeline GPU scheduling.

## Available Examples

### Multi-Pipeline Test

`start_multi_pipeline_test.py` - Demonstrates running multiple RL training pipelines concurrently under the RLix control plane.

## Prerequisites

- Python 3.10+
- Ray installed and configured
- ROLL framework (for pipeline integrations)

## Running the Examples

### Multi-Pipeline Example

```bash
# Run a single pipeline
python examples/start_multi_pipeline_test.py --config_name full_finetune_pipeline1

# Run two full-finetune pipelines concurrently
python examples/start_multi_pipeline_test.py --config_name full_finetune_pipeline1,full_finetune_pipeline2

# Run two multi-LoRA pipelines concurrently
python examples/start_multi_pipeline_test.py --config_name multi_lora_pipeline1,multi_lora_pipeline2
```

## Configuration

Example configurations are located in:

- `examples/config/` - DeepSpeed and environment configs
- `examples/rlix_test/` - RLix pipeline configs (full finetune and multi-LoRA)

## Integration with ROLL

These examples require the [ROLL fork](https://github.com/rlops/ROLL), included as a git submodule:

```bash
# Initialize and install the ROLL submodule
git submodule update --init external/ROLL
pip install external/ROLL
```

## Troubleshooting

- **Ray connection errors**: Ensure Ray is running (`ray start --head`)
- **Import errors**: Verify all dependencies are installed
- **GPU allocation issues**: Check Ray's GPU resources with `ray status`
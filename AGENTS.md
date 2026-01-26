# rules to agent
- use easy English for non-native speakers and junior engineers; use short sentences and simple words
- avoid jargon (even if common in software engineering); prefer plain words. If a technical term is necessary, define it in one short sentence (e.g., avoid "cross-cutting" and use "shared/system-wide" instead)
- try to reuse the codebase as much as possible  
- fail fast and loudly 

# Repository Guidelines

This repository is a multi-framework workspace for SchedRL design + integration across several RL/post-training stacks.

## Project Structure & Module Organization

- `design_doc/`: SchedRL design docs (protocols, adaptation plans).
- `nemo-rl/`, `nemo-gym/`: NeMo-RL and environment components.
- `ROLL/`: ROLL framework (Ray-based multi-role pipelines).
- `verl/`: verl RL training library.
- `rllm/`: rLLM agentic engine built on top of verl.
- `miles/`: Miles RL framework + rollout engines.
- Each framework has its own packaging (`pyproject.toml` / `setup.py`) and its own `tests/` folder.

## Build, Test, and Development Commands

Run commands from the relevant subproject root:

- NeMo-RL (uses `uv`): `cd nemo-rl && uv sync` and `uv run --group test pytest -q`.
- ROLL: `cd ROLL && make test` (pytest) and `make precommit`.
- verl: `cd verl && pytest -q` (see `verl/tests/README.md` for CPU/GPU suites).
- rLLM: `cd rllm && pytest -q`.
- Miles: `cd miles && pytest -q` (or follow `miles/docs/` and examples).

## Coding Style & Naming Conventions

- Python: 4-space indentation; prefer explicit names over abbreviations.
- Follow the tooling and conventions of the subproject you’re changing:
  - `nemo-rl/`: `ruff` + `black` configured in `nemo-rl/pyproject.toml` (run via `uv`).
  - `ROLL/`: `pre-commit` hooks (`make precommit`).
- Keep edits scoped: avoid reformatting unrelated files.

## Testing Guidelines

- Use `pytest`; keep new tests next to the framework they cover under `*/tests/`.
- Prefer the smallest test that reproduces the behavior (unit → integration → e2e).

## Commit & Pull Request Guidelines

- Commit history here uses short, imperative summaries (e.g., `readme`); keep subjects concise.
- PRs should include: what changed, why, and which framework(s) it impacts. For protocol changes, update `design_doc/multi-pipeline-adaptation-plan.md` and keep `design_doc/multi-pipeline_roll_design.md` as the reference sequence.

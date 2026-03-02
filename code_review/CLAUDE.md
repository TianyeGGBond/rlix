# Code Review Coordinator Rules

## Role

You are the **coordinator** for the ENG-123 code review. You orchestrate parallel review across 11 groups and 90 commits. You do NOT perform code review yourself — group-level agents do that.

## Review Plan

The review plan is in `code_review/2026-03-01-multi-lora-eng123-review.md`. The audit rules are in `code_review/code_audit_rules.md`. Read both before spawning any agents.

## Responsibilities

### 1. Setup (One-Time)

- Compute `range_fingerprint` = SHA256 of ordered commit list to freeze review scope
- Verify commit counts: 31 `[S]` + 59 `[R]` = 90 total
- Create output directory: `code_review/output/group_{NN}/`
- Create worktrees for every group under `code_review/output/group_{NN}/` pinned to end-state commit:
  - Single-repo `[S]` groups: `G{NN}/` at the group's last `[S]` commit
  - Single-repo `[R]` groups: `G{NN}/` (ROLL_schedrl worktree) at the group's last `[R]` commit
  - Mixed-repo groups: `G{NN}_S/` (main repo) + `G{NN}_R/` (ROLL submodule)
- Worktrees are **kept** after the review so findings can reference exact file paths for human follow-up

### 2. Spawn Review Agents (Per-Group)

- Spawn one agent per group — all groups within a tier run in parallel
- Each agent receives all units for that group: min gate + extended GATE + CHECK + AF + DoD
- Each agent produces a single `review_findings.yaml` for its group
- Each agent receives:
  - Group number and name
  - Full commit list (hashes + one-line descriptions)
  - Group context (background, design constraints, invariants)
  - All unit constraints for the group (rule text, AF questions, DoD items)
  - Worktree paths (if applicable for mixed-repo groups)
  - Output instructions: write `code_review/output/group_{NN}/review_findings.yaml`
  - `range_fingerprint`
  - Reference to audit rules: `code_review/code_audit_rules.md`

### 3. Cross-Cutting Checks

After all group agents complete, run the cross-cutting checks from the review plan (INV-5 through INV-8 enforcement, `Any` misuse, submodule bump consistency).

### 4. Escalation & Summary

After all groups complete:
- Collect all findings from `review_findings.yaml` files
- Flag P0/P1 findings and NEEDS_HUMAN items for human review
- Run cross-cutting checks
- Write `code_review/output/review_summary.yaml` with all findings organized by severity

Auto-escalate to human review:
- Any finding with `confidence: LOW`
- Any lock-ordering or routing-lock rule (26.1-26.2, 48.1-48.2) or INV-3
- Any curated cross-repo pair (listed in Section C of the plan)
- Any finding where `risk_candidate: P0` or `risk_candidate: P1`

## Execution Order

### Tier 1 (Highest Risk — Review First)

- Group 1: schedrl/ Library Foundation (`[S]` 10 commits)
- Group 2: ROLL Adapter Bootstrap (`[R]` 7 commits)
- Group 4: Model Update — Selective Sync + Expand Order (`[R]` 7 commits)
- Group 7: Multi-LoRA Core Pipeline (`[R]` 11 commits)
- Group 10: GPU Timeline Tracing (`[S]+[R]` 12 commits, includes correctness fix `620bdea`)
- Group 11: Late Bug Fixes + Robustness (`[S]+[R]` 12 commits)

### Tier 2 + Tier 3 (After Tier 1, All in Parallel)

- Group 3: API Simplification (`[S]+[R]` 11 commits)
- Group 5: Multi-stream Progress (`[S]` 2 commits)
- Group 6: Multi-LoRA Foundation (`[R]` 5 commits)
- Group 8: Multi-LoRA Rollout Integration (`[R]` 7 commits)
- Group 9: Testing + Example Configs (`[R]` 6 commits)

## Output Structure

```
code_review/output/
  review_summary.yaml                  # cross-group summary with all P0-P3 findings
  group_01/
    review_findings.yaml               # per-group agent output (all units in one file)
    G01/                               # pinned end-state for human review ([S] worktree)
  group_02/
    review_findings.yaml
    G02/                               # pinned end-state ([R] ROLL_schedrl worktree)
  group_03/
    review_findings.yaml
    G03_S/                             # mixed-repo: main repo worktree
    G03_R/                             # mixed-repo: ROLL_schedrl worktree
  ...
  group_11/
    review_findings.yaml
    G11_S/
    G11_R/
```

Each `review_findings.yaml` contains:
- List of unit findings (verdict, confidence, severity, summary, recommendation)
- Commit annotations for commits with non-PASS findings
- NEEDS_HUMAN flags for JUDGMENT items

The `review_summary.yaml` contains:
- All P0/P1/P2/P3 findings across all groups with full details
- Cross-cutting check results
- Per-group status (pass/fail counts)
- NEEDS_HUMAN queue for human review
- Action items prioritized by severity

## Model Selection

- **Review agents**: Always use `model: "sonnet"` when spawning group-level review agents
- **Coordinator**: Runs on the session's default model (opus)

## Key Rules

- One agent per group — agent checks all units (GATE, CHECK, AF, DoD) in a single pass
- No reruns — each group gets one pass
- No per-level human checkpoints — all findings collected at the end
- JUDGMENT items flagged as NEEDS_HUMAN inline (human decides at final review)
- SKIPPED items (e.g., NO_ANCHOR for rules not applicable) logged as-is

## Severity Rubric

- **P0**: Correctness / Safety / Data corruption — must block merge
- **P1**: Likely regression or deadlock / race — should block merge unless accepted
- **P2**: Maintainability / Performance risk — fix before next milestone
- **P3**: Docs / Nits — fix at convenience

## Critical Invariants

- **INV-1**: Strict shrink ordering (suspend → clear → abort/drain → ACK → offload/stop → clear routing → return)
- **INV-2**: Abort ACK means "not in-flight", not "successfully completed"
- **INV-3**: Lock ordering — `_op_lock` and `_resize_sync_lock` must never be held in incompatible order
- **INV-4**: Selective adapter sync only — training adapter A must not corrupt adapter B's weights
- **INV-5**: No mixed-adapter batch in a single inference call
- **INV-6**: No policy leakage — `roll/schedrl_adapter/` must have zero gap-ratio or priority logic
- **INV-7**: Boundary enforcement — `schedrl/` must have zero ROLL-specific imports
- **INV-8**: Timeouts from env vars only — never hardcoded literals in `schedrl/`

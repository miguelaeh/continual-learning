# Agent Learning System

This directory contains a broader continual-learning scaffold for an autonomous agent.

The design assumption is:

- interactions are logged immediately into append-only episodic memory
- retrieval can use those traces immediately
- parametric updates happen later through gated consolidation

This system is intentionally separate from the memory-layer research code in the parent directory.

## What This Provides

- an append-only interaction log
- a simple retrieval layer over past traces
- a structured profile memory store
- a runtime prompt builder that combines profile memory and retrieved traces
- a consolidation dataset builder
- a regression gate for candidate checkpoints
- a checkpoint registry and promotion flow

## What This Does Not Yet Provide

- a production retriever
- a full trainer for delta memory banks
- automatic online self-editing

Those pieces can be added incrementally once the logging, retrieval, and promotion pipeline is stable.

## Install

```bash
cd agent_learning_system
pip install -e .
```

## Example Workflow

1. Log an interaction:

```bash
python scripts/log_interaction.py \
  --store-dir runtime_store \
  --session-id demo-session \
  --user-message "Please fix the broken script." \
  --assistant-response "I found the failing command and patched it." \
  --outcome success
```

2. Inspect recent traces:

```bash
python scripts/show_recent.py --store-dir runtime_store --limit 5
```

3. Store durable profile memory:

```bash
python scripts/upsert_profile_memory.py \
  --store-dir runtime_store \
  --memory-type fact \
  --key user.name \
  --value Miguel \
  --source "user stated it directly"
```

4. Build the runtime prompt the live agent would receive:

```bash
python scripts/build_runtime_prompt.py \
  --store-dir runtime_store \
  --session-id demo-session \
  --user-message "What is my name?"
```

5. Build a consolidation dataset:

```bash
python scripts/build_consolidation_dataset.py \
  --store-dir runtime_store \
  --output runtime_store/consolidation/train.jsonl
```

6. Run a regression gate on a candidate checkpoint:

```bash
python scripts/run_regression_gate.py \
  --spec configs/regression_gate.yaml \
  --candidate checkpoints/candidate.json
```

7. Promote a passing checkpoint:

```bash
python scripts/promote_checkpoint.py \
  --registry runtime_store/registry/checkpoints.json \
  --candidate-id candidate-001
```

## Layout

- `src/agent_learning_system/`
  Core package
- `scripts/`
  Operational CLIs
- `configs/`
  Regression gate specs
- `docs/`
  Architecture records
- `tests/`
  Basic verification

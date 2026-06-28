# Current Best Branch

## Best Recovery Checkpoint

- config: [configs/recovery_oasst1_1layer_300.yaml](../configs/recovery_oasst1_1layer_300.yaml)
- checkpoint: `checkpoints/recovery_oasst1_1layer_300/memory_step_300.pt`

Why this one:

- preserves arithmetic
- preserves self-introduction better than the degraded 3-layer late checkpoints
- provides the cleanest starting point for continual-learning experiments

## Best Continual Checkpoint

- config: [configs/continual_oasst1_1layer_cpu_qa_fullmem_lr003.yaml](../configs/continual_oasst1_1layer_cpu_qa_fullmem_lr003.yaml)
- checkpoint: `checkpoints/continual_oasst1_1layer_cpu_qa_fullmem_lr003/memory_step_20.pt`

Why this one:

- first branch that clearly injected `Miguel` into the model's answers
- lower learning rate reduced the severe gibberish seen in earlier full-memory runs
- still uses the same sparse-memory retrofit architecture

## Important Qualification

This branch is **not** the strict paper-style stage-3 setting.

It uses:

- `continual.train_full_memory: true`

Meaning:

- base Qwen weights stay frozen
- all inserted memory-layer parameters can update
- value-table rows are still sparsely masked
- the rest of the memory-layer parameters update densely

So this branch is:

- architecturally consistent with the retrofit
- experimentally useful
- less faithful to the paper's intended sparse stage 3

## Best-Preserved Code Snapshot

The code snapshot corresponding to the current preserved state is in:

- [../working_experiment_2026-05-01/](../working_experiment_2026-05-01/)

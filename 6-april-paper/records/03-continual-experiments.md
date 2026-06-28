# Continual Experiments

## Strict Paper-Style Stage 3

The strict stage-3 interpretation used:

- frozen Qwen backbone
- frozen inserted memory-layer projections
- only selected value-table rows updated

That corresponds to:

- `continual.train_full_memory: false`

In code, that means only `shared_store.values.weight` is unfrozen in:

- [src/smf_retrofit/modeling/qwen.py](../src/smf_retrofit/modeling/qwen.py)

## Result of Strict Stage 3

It did learn something in loss terms, but it was generally too weak to reliably flip the model from its prior answer pattern to the target fact on open-ended prompts.

Observed patterns:

- loss dropped
- slot selection worked
- outputs often stayed near the recovery behavior
- raw QA prompts sometimes changed, but not reliably toward the target fact

## Continual-Stage Bugs and Fixes

### Zero-Supervision Packed Batches

Packed assistant-only supervision sometimes created chunks with zero trainable labels, which caused `NaN` loss. This was fixed by skipping such chunks in:

- [src/smf_retrofit/data.py](../src/smf_retrofit/data.py)

### Packed Conversations Weakened the Signal

Packing multiple chat samples into a continuous token stream diluted the continual-learning target. An unpacked mode was added:

- `TextDataConfig.pack_samples`
- [src/smf_retrofit/data.py](../src/smf_retrofit/data.py)

### Tail Truncation

For unpacked continual chat data, keeping the tail of the conversation worked better than keeping the head, because the supervised assistant answer is usually near the end.

### Leading-Whitespace Preservation

Assistant targets now preserve leading spaces. This matters for tokenization, especially for targets like `" Miguel"`.

## Stronger Continual Branch

The first branch that clearly injected the fact used:

- frozen Qwen backbone
- trainable inserted memory module
- value-table gradients still sparse-masked
- other memory-layer parameters updated densely

That corresponds to:

- `continual.train_full_memory: true`

This is **not** full-model finetuning. It is still constrained to the inserted memory layers.

## Best Continual Result So Far

Most informative branch:

- config: [configs/continual_oasst1_1layer_cpu_qa_fullmem_lr003.yaml](../configs/continual_oasst1_1layer_cpu_qa_fullmem_lr003.yaml)
- checkpoint: `checkpoints/continual_oasst1_1layer_cpu_qa_fullmem_lr003/memory_step_20.pt`

Observed outputs:

- `What is my name?` -> contained `Miguel`
- `Who am I?` -> `you Miguel`
- `What should you call me?` -> `should be called Miguel`

This was the first branch where the fact was clearly written into the model's answers.

## What Did Not Work Well

- strict value-only stage 3 on chat prompts
- strict value-only stage 3 on raw QA prompts
- overly aggressive full-memory continual tuning, which learned the fact but destabilized generation

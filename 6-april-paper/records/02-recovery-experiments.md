# Recovery Experiments

## Early Failures

The first recovery attempts failed for two independent reasons:

1. The toy recovery corpus was far too small.
2. The implementation initially treated some datasets as plain text instead of real chat conversations.

Observed failure pattern:

- base model coherent
- recovery model degraded
- continual stage meaningless because it started from a damaged recovered model

## Important Fixes

### OASST1 Reconstruction

`OpenAssistant/oasst1` is stored as a flat message tree. It had to be reconstructed into linear user/assistant paths in:

- [src/smf_retrofit/data.py](../src/smf_retrofit/data.py)

### Recovery Sanity Gate

Prompt-based recovery checks were added so that a recovered checkpoint is rejected if it fails basic prompts like arithmetic or self-introduction.

Relevant files:

- [scripts/check_recovery.py](../scripts/check_recovery.py)
- [data/recovery_prompts.json](../data/recovery_prompts.json)
- [src/smf_retrofit/eval.py](../src/smf_retrofit/eval.py)

### Best-Checkpoint Scanning

The final recovery checkpoint often degraded more than earlier ones. Early checkpoints had to be tested directly rather than assuming the last one was best.

## Best Recovery Branch Found

The healthiest recovery branch found so far is:

- config: [configs/recovery_oasst1_1layer_300.yaml](../configs/recovery_oasst1_1layer_300.yaml)
- checkpoint: `checkpoints/recovery_oasst1_1layer_300/memory_step_300.pt`

Behavior of that checkpoint:

- `Introduce yourself briefly.` remained close to base
- `What is 2 plus 2?` remained correct
- chat quality stayed much stronger than the earlier 3-layer degraded branches

## Why 1 Layer Helped

Using a single retrofitted layer reduced the amount of destructive surgery applied to the pretrained model. That made recovery substantially easier and more stable on limited compute.

## Recovery Conclusion

Recovery is still a real adaptation phase. It is lighter than the older heavy pretraining approach, but it is not optional and it is not cheap.

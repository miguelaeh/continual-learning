# Checkpoint Manifest

This snapshot preserves the code and experiment inputs, but it does **not** duplicate the checkpoint files themselves.

Checkpoint paths in the main repo at snapshot time:

- recovery best: `checkpoints/recovery_oasst1_1layer_300/memory_step_300.pt`
- continual best factual branch: `checkpoints/continual_oasst1_1layer_cpu_qa_fullmem_lr003/memory_step_20.pt`

Other notable branches:

- `checkpoints/continual_oasst1_1layer_cpu_compact_fullmem_lr003/memory_step_20.pt`
- `checkpoints/continual_oasst1_1layer_cpu_compact_fullmem_lr003_spacetok/memory_step_20.pt`

If you later want a fully self-contained archival bundle, the next step would be to copy the selected checkpoint files into this snapshot as well.

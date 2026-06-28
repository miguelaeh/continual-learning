# Checkpoints

Preserved continual branch:

- `checkpoints/continual_oasst1_1layer_cpu_qa_fullmem_lr003_1p5b_richdata_v1/memory_step_10.pt`
- `checkpoints/continual_oasst1_1layer_cpu_qa_fullmem_lr003_1p5b_richdata_v1/memory_step_20.pt`
- `checkpoints/continual_oasst1_1layer_cpu_qa_fullmem_lr003_1p5b_richdata_v1/memory_step_30.pt`
- `checkpoints/continual_oasst1_1layer_cpu_qa_fullmem_lr003_1p5b_richdata_v1/memory_step_40.pt`
- `checkpoints/continual_oasst1_1layer_cpu_qa_fullmem_lr003_1p5b_richdata_v1/memory_step_50.pt`
- `checkpoints/continual_oasst1_1layer_cpu_qa_fullmem_lr003_1p5b_richdata_v1/memory_step_60.pt`
- `checkpoints/continual_oasst1_1layer_cpu_qa_fullmem_lr003_1p5b_richdata_v1/memory.pt`

Preserved prerequisites:

- `checkpoints/recovery_oasst1_1layer_1p5b_mini/memory.pt`
- `checkpoints/background_oasst1_1layer_1p5b_mini_stats.pt`

Interpretation at preservation time:

- This is the first branch where richer task data clearly improved role consistency.
- `step50` and `step60` were the most informative late checkpoints from this run.
- The run still had malformed outputs, so preserving the full series matters.

Re-run command from this snapshot:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/sparse_finetune.py \
  --config configs/continual_oasst1_1layer_cpu_qa_fullmem_lr003_1p5b_richdata_v1.yaml
```

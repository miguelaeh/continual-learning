#!/usr/bin/env bash
# Full pipeline for 1.5B KL n=256 experiment.
# Run from the repo root (6-april-paper/).
set -e

EXPERIMENT=experiments/1p5b_kl_n256_v3

echo "=== Stage 1: Recovery ==="
python scripts/recover.py --config $EXPERIMENT/recovery.yaml

echo "=== Stage 2: Background stats ==="
python scripts/collect_background.py --config $EXPERIMENT/background.yaml

echo "=== Stage 3: Sparse continual finetuning ==="
python scripts/sparse_finetune.py --config $EXPERIMENT/continual.yaml

echo "=== Stage 4: Evaluation ==="
python scripts/evaluate_triviaqa_checkpoints.py \
  --config $EXPERIMENT/continual.yaml \
  --data-path data/april6_triviaqa_val_200.jsonl \
  --checkpoint $EXPERIMENT/checkpoints/continual/memory.pt \
  --max-samples 200 --max-new-tokens 16 --device mps

echo "Done. Checkpoints in $EXPERIMENT/checkpoints/"

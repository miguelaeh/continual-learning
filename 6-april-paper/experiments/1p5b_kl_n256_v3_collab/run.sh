#!/usr/bin/env bash
# Full pipeline for v3_collab — run from repo root (6-april-paper/).
set -e

EXPERIMENT=experiments/1p5b_kl_n256_v3_collab

echo "=== Stage 1: Recovery ==="
python scripts/recover.py --config $EXPERIMENT/recovery.yaml

echo "=== Stage 2: Background stats ==="
python scripts/collect_background.py --config $EXPERIMENT/background.yaml

echo "=== Stage 3: Sparse continual finetuning ==="
python scripts/sparse_finetune.py --config $EXPERIMENT/continual.yaml

echo "Done. Checkpoints in $EXPERIMENT/checkpoints/"

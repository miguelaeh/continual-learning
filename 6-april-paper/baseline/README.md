# Clean Baseline

This folder defines the mainline baseline for this repo.

The goal is to keep a small, defensible implementation path that is:

- close to the April 6 paper
- free of the exploratory patch pile
- reproducible with explicit configs

What is included in the baseline:

1. Real recovery on `OpenAssistant/oasst1`
2. Correct checkpoint wiring
3. No zero-batch hangs
4. No accidental truncation of every continual sample
5. A richer continual-learning dataset for fairer evaluation
6. Dense checkpoint saving plus automatic checkpoint selection

What is *not* part of the baseline:

- dual-bank research variants
- heavy locality/distillation experiments
- extra train-mode sweeps
- prompt-specific patch branches

Those still exist in the repo for research, but they are not the mainline.

## Baseline configs

Paper-style baseline on `Qwen2.5-0.5B-Instruct`:

- `configs/baseline_paper_recovery_qwen25_0p5b.yaml`
- `configs/baseline_paper_background_qwen25_0p5b.yaml`
- `configs/baseline_paper_continual_qwen25_0p5b.yaml`
- `configs/baseline_paper_continual_qwen25_0p5b_kl.yaml`

Scale-check baseline on local `Qwen2.5-1.5B-Instruct`:

- `configs/baseline_scale_continual_qwen25_1p5b.yaml`

## Checkpoint selection

Use:

```bash
python scripts/select_best_checkpoint.py \
  --config configs/baseline_scale_continual_qwen25_1p5b.yaml \
  --checkpoints-dir checkpoints/baseline_scale_continual_qwen25_1p5b \
  --prompts-file data/baseline_identity_eval.json
```

This scores each saved checkpoint against the baseline prompt suite and prints the best one.

Validated reference result on the current 1.5B rich-data run:

- run: `checkpoints/continual_oasst1_1layer_cpu_qa_fullmem_lr003_1p5b_richdata_v2_seq128`
- selected checkpoint: `memory_step_30.pt`
- summary: this is still only a partial success, but it is the cleanest factual-learning checkpoint in the current mainline branch family

## Recommended baseline workflow

First prepare the `TriviaQA 1k` task file:

```bash
python scripts/prepare_triviaqa_april6.py \
  --output data/april6_triviaqa_train_1k.jsonl \
  --max-samples 1000
```

Then run the baseline:

```bash
python scripts/recover.py --config configs/baseline_paper_recovery_qwen25_0p5b.yaml
python scripts/check_recovery.py --config configs/baseline_paper_recovery_qwen25_0p5b.yaml
python scripts/collect_background.py --config configs/baseline_paper_background_qwen25_0p5b.yaml
python scripts/sparse_finetune.py --config configs/baseline_paper_continual_qwen25_0p5b.yaml
python scripts/select_best_checkpoint.py \
  --config configs/baseline_paper_continual_qwen25_0p5b.yaml \
  --checkpoints-dir checkpoints/baseline_paper_continual_qwen25_0p5b \
  --prompts-file data/baseline_identity_eval.json
```

See also the April 6 audit:

- [../records/07-april6-paper-audit.md](../records/07-april6-paper-audit.md)

## Interpretation

If this clean baseline still fails the stronger continual-learning goal, that is useful information:

- it means the paper-style prototype may be adequate for its benchmark setting
- but it is still insufficient for the stronger agent-learning objective we care about

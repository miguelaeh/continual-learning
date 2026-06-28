# Sparse Memory Retrofit Prototype

Prototype implementation of the pipeline described in "Improving Sparse Memory Finetuning" (April 6, 2026).

This repo is intentionally optimized for the paper's lighter path:

1. Retrofit a pretrained transformer by replacing selected Qwen FFNs with sparse memory layers.
2. Run a short recovery/healing stage to restore baseline competence.
3. Collect background slot-usage statistics.
4. Finetune on new data using sparse memory updates selected by TF-IDF or KL scoring.

The defaults are prototype-scale rather than paper-scale, so you can run smoke tests and small experiments without the heavy standalone memory pretraining phase used by earlier work.

## Install

```bash
pip install -e .
```

## Scripts

```bash
python scripts/recover.py --config configs/recovery.yaml
python scripts/check_recovery.py --config configs/recovery.yaml
python scripts/collect_background.py --config configs/background.yaml
python scripts/sparse_finetune.py --config configs/continual.yaml
```

For a more realistic recovery run on a Mac, start from:

```bash
python scripts/recover.py --config configs/recovery_ultrachat_mac.yaml
python scripts/check_recovery.py --config configs/recovery_ultrachat_mac.yaml
```

That config uses the `HuggingFaceH4/ultrachat_200k` `train_sft` split and formats its `messages` field as chat turns. Dataset card: https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k

For a paper-aligned recovery run, use:

```bash
python scripts/recover.py --config configs/recovery_oasst1_mac.yaml
python scripts/check_recovery.py --config configs/recovery_oasst1_mac.yaml
```

That config reconstructs English conversation paths from the flat `OpenAssistant/oasst1` message tree table and targets the 20k-sample recovery setup described in the paper. Dataset card: https://huggingface.co/datasets/OpenAssistant/oasst1

If recovery passes and you want to continue with that exact checkpoint path, use the matching configs:

```bash
python scripts/collect_background.py --config configs/background_oasst1_mac.yaml
python scripts/sparse_finetune.py --config configs/continual_oasst1_mac.yaml
```

## Data formats

The loaders accept either Hugging Face datasets or local text corpora.

- `data.path` may point to `.jsonl`, `.json`, or `.txt`
- local JSON/JSONL rows should expose a text field, defaulting to `text`
- for QA tasks, pre-format each example into a single text completion sample
- the loader also supports common instruction/chat schemas:
  `messages`, `instruction`/`input`/`output`, and plain `text`

## Recovery First

The recovery stage is the first hard gate in this pipeline.

- Do not treat the toy recovery files as sufficient for a real run.
- Use a broad general-purpose instruction corpus for recovery.
- Run `python scripts/check_recovery.py --config configs/recovery.yaml` after recovery.
- `scripts/sparse_finetune.py` will refuse to run by default if recovery loss is too far above base-model loss on the configured recovery eval set.

## Notes

- This implementation targets `Qwen/Qwen2.5-0.5B-Instruct` by default but keeps the model-surgery code generic for Qwen2 causal LMs.
- The recovery stage is required. This prototype does not claim zero-shot memory injection.
- Paper-scale memory tables can be large. Start with the prototype defaults in `configs/`.

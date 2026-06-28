# Sparse Memory Retrofit Prototype

## Experiment Results

### v1 — TriviaQA (baseline validation)
Single memory layer at position 14. Validates that the method works: sparse slot updates learn the new task with near-zero forgetting.

| Memory layers | n_keys | top_k | top_t | Steps | OASST1 forgetting |
|---|---|---|---|---|---|
| [14] | 256 | 8 | 500 | 1000 | **-0.14%** |

### v2 — Claude-Distills 1k samples
Switched continual task to `clzoro/Claude-Distills` (coding + reasoning). Learned a more direct, code-first response style.

| Memory layers | top_t | Steps | Samples | Loss drop | OASST1 forgetting |
|---|---|---|---|---|---|
| [14] | 500 | 1000 | 1,000 | 11.19 → 10.58 | **-0.09%** |

### v3 — Claude-Distills top_t sweep
10k samples, 2000 steps, seq_length=512. Systematic sweep to find the learning/forgetting tradeoff boundary.

| top_t | % slots updated | Loss drop | OASST1 forgetting |
|---|---|---|---|
| 500 | 0.76% | 10.76 → 10.58 | -0.09% |
| 1000 | 1.5% | 10.76 → 10.66 | ~0% |
| 2000 | 3.0% | 10.76 → 9.40 | -11.89% |
| 3000 | 4.6% | 10.76 → 5.84 | -16.43% |
| 5000 | 7.6% | 10.76 → 4.82 | -18.35% |

**Key finding:** Hard cliff between top_t=1000 and top_t=2000. Below: near-zero learning and near-zero forgetting. Above: strong learning but significant forgetting (12–18%).

**Root cause:** Slot activation distribution is highly skewed. A typical batch (~120 tokens) accesses ~2,400 unique slots, but the top 1% of slots capture 35% of all accesses (hub slots used by both tasks). At top_t>1000 the selector inevitably includes these shared hub slots, causing forgetting. Longer sequences (seq_length=1024) do not help — hub slots fire on every token regardless.

**Slot activation stats (200 batches):**
- Unique slots per batch: ~2,400 (120-token seq) to ~4,500 (200-token seq)
- Top 1% of slots (555) → 35.5% of all accesses
- Top 5% of slots (2,775) → 57.7% of all accesses

This skew is inherent to the retrofit approach (same in the original paper's repo — no load balancing during recovery). Fix: add more memory layers so hub slots are layer-specific, or add load balancing loss during recovery.

### v4 — 3 memory layers (in progress)
Scaling to 3 layers at positions [6, 12, 18] with top_k=18, matching the original paper's configuration. Hypothesis: independent slot tables per layer reduce hub-slot overlap between tasks, pushing the safe top_t ceiling higher.

---


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

## Clean Baseline

To avoid mixing the paper-style baseline with the exploratory research branches, use:

- [baseline/README.md](./baseline/README.md)

That baseline keeps only essential fixes and adds prompt-based checkpoint selection, while the many `v*` and `delta*` configs remain available as research branches.

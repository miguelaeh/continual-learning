# Memory Layer Initialization Investigation

> For better results, we can pre-train the memory layers instead of just distilling them, so the information is better organized and less compressed, enabling less forgetting and more sparsity. But we can do that only for models we know people want to use, given is reusable but expensive and very slow to train.

## Problem Statement

The paper's Phase 1 pretrains memory layers by minimizing language modeling loss for 128,000 steps. The training signal is indirect: the memory layer output must propagate through multiple subsequent transformer layers and the LM head before producing a useful gradient.

On an A100 80GB GPU with `batch_size=2, seq_length=2048, gradient_accumulation_steps=8`, even 2,500 pretrain steps takes significant time. More critically, the resulting model produces degraded output — indicating the memory layers haven't learned to approximate the original FFN's function well enough.

## Experiments: Pretraining Approach (Phase 1)

### Setup

- Base model: `google/gemma-3-4b-it`
- Memory layers at layers [9, 17, 25], n_keys=1024 (1M entries)
- Pretrain config: AdamW, lr=1e-4 (projections), lr=1e-3 (values), warmup=500, batch_size=2, seq_length=2048

### Result: 2,500 pretrain steps

**Testing the pretrained model (before any remember step):**

Query: "What is the capital of France?"
```
Response: Paris
:
:
::
:
 : :
 :
 : : : : : : : : : : : : : : : : : : : : : : : : : ...
```

The model gets the answer right ("Paris") but immediately degenerates into repetitive output. The memory layers are partially functional — they contribute enough signal for the first token — but not well-trained enough to produce clean hidden states for subsequent generation.

### Result: 2,500 pretrain steps + 100 remember steps (lr=2.0)

Query: "What is my name?"
```
Response: I dont know name You! I' just language model Im built the of Google
 you me name YouPlease us in way are you I you me name
 I You name
,You a name
You nameYou a
 nameYou
 ...name name name name name name name
```

Complete collapse. The remember step with lr=2.0 pushed the already-fragile memory slots into incoherent territory.

### Result: 2,500 pretrain steps + 50 remember steps (lr=0.1)

Even with a much lower learning rate:
```
Response: As an AI, dont worry I' not to your name Im just name model
 am language. you call,' name
 do have name
 ...model model model model model model
```

Still collapsed. The problem is upstream — the memory layers themselves are too undertrained for any fine-tuning to work.

## Analysis

The core issue: **2,500 steps is not enough to learn the original FFN's function through indirect language modeling loss.** The memory layers replace 3 critical FFN layers (at positions 9, 17, 25) in a 34-layer model. The remaining 31 layers expect specific hidden state patterns from these positions. Without adequate pretraining, the memory layers produce near-random noise that:

1. Gets the first few tokens approximately right (other layers compensate)
2. Accumulates errors in autoregressive generation
3. Causes repetitive loops once the hidden states drift too far

The paper uses 128,000 steps for good reason — but that requires days of GPU time.

## Hypothesis: Distillation

**If we train each memory layer to directly match the original FFN's output via MSE loss, the signal is ~64x more direct and should converge proportionally faster.**

### Why MSE distillation should be faster

| Aspect | LM Loss (Pretrain) | MSE Distillation |
|---|---|---|
| Signal path | memory layer → 25+ remaining layers → LM head → loss | memory layer → loss (direct) |
| Gradient path | loss → LM head → 25+ layers → memory layer | loss → memory layer (direct) |
| Target | Implicit (reduce perplexity) | Explicit (match FFN output) |
| Hidden state quality | Contaminated (memory output affects downstream layers) | Clean (teacher produces correct states) |
| Layers interact | Yes (partially-trained layer 9 corrupts input to layer 17) | No (each layer trains on correct inputs independently) |

### Approach

1. Load the base model as a **frozen teacher** (no memory layers injected)
2. Create **standalone memory layers** (not in the model)
3. Register forward hooks on target FFN layers to capture input/output
4. For each batch:
   - Forward through teacher → hooks capture FFN inputs and outputs
   - Forward captured inputs through standalone memory layers
   - Loss = MSE(memory_output, ffn_output) averaged across 3 layers
5. Save checkpoint in the same format (compatible with Phase 2 and 3)

### Configuration

```yaml
distill:
  learning_rate: 1.0e-3       # 10x pretrain (direct signal allows it)
  value_learning_rate: 1.0e-2  # 10x projection LR
  warmup_steps: 100
  total_steps: 2000           # ~64x shorter than pretrain's 128K
  batch_size: 2
  seq_length: 2048
```

### Usage

```bash
# Instead of:
python scripts/pretrain_memory.py --config configs/pretrain_memory.yaml  # 128K steps

# Use:
python scripts/distill_memory.py --config configs/distill_memory.yaml    # 2K steps
```

The distilled checkpoint is directly compatible with Phase 2 (IDF collection) and Phase 3 (continual learning / remember).

## Results

### Run 1: [Date TBD]

- Config: `configs/distill_memory.yaml`
- Steps: 2000
- Final MSE loss: [TBD]

**Generation quality (after distillation, no remember):**
- Query: "What is the capital of France?" → [TBD]
- Coherent text? [TBD]
- Repetition issues? [TBD]

**After remember step:**
- Facts taught: [TBD]
- Query: "What is my name?" → [TBD]

### Comparison: Distillation vs. Pretraining

| Metric | Pretrain (2.5K steps) | Pretrain (128K steps) | Distill (2K steps) |
|---|---|---|---|
| Training time | ~X min | ~Y hours | ~Z min |
| Text coherence | Degraded (repetition) | [TBD] | [TBD] |
| Remember works? | No (collapses) | [TBD] | [TBD] |
| Forgetting (NQ F1) | N/A | [TBD] | [TBD] |

## Change: Unfreeze All Memory Params During Remember

### Problem

After distillation, the model generates coherent text (no repetition/garbage). However, the remember step fails to teach new facts — the model still says "I don't know your name" after training on "The user's name is Miguel".

The paper's `freeze_for_continual_learning()` only unfreezes `shared_store.values.weight`:

```python
# freeze_utils.py — Phase 3 freezing
shared_store.values.weight.requires_grad = True
# Everything else frozen: query_proj, silu_proj, value_proj, keys
```

This makes sense for the paper's setup: after 128K pretrain steps, the projections are well-trained and know how to route information through the memory. Updating only the value embeddings (what's stored in each slot) is enough to encode new task knowledge.

But with distillation, the projections learned to **mimic the original FFN**, not to **encode new information**. They reproduce the FFN's behavior faithfully, but they haven't developed the flexibility to route novel facts through the memory lookup and gating mechanism. Updating only the values doesn't help because the projections don't know how to read/write new knowledge to/from those slots.

### Fix

In `scripts/remember.py`, replaced `freeze_for_continual_learning()` with `freeze_base_model()`:

```diff
-    # Freeze for continual learning
-    freeze_for_continual_learning(model, memory_config)
+    # Freeze base model, keep all memory params trainable
+    from src.model.freeze_utils import freeze_base_model
+    freeze_base_model(model, memory_config)
```

This unfreezes all memory layer parameters during the remember step:
- `query_proj`, `query_norm` — how the input is projected to key space
- `silu_proj`, `value_proj` — the SiLU gating and output projection
- `shared_store.keys` (K1, K2) — product key lookup parameters
- `shared_store.values` — the actual memory slot values

### Trade-off

Unfreezing all memory params during remember means **more capacity to learn new facts**, but also **higher risk of forgetting** existing knowledge (since projections and keys can shift). For the paper's full continual learning benchmark (many sequential tasks), the values-only approach is safer. For our "remember a few facts" use case, the extra capacity is more important.

If forgetting becomes a problem in practice, a middle ground would be to unfreeze values + value_proj only, keeping query routing (query_proj, keys) frozen.

## Conclusion

[TBD after experiments]

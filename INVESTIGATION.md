# Memory Layer Initialization Investigation

First try: small pre-training from scratch using dataset fineweb-edu. The model produces crap at the end of the outputs and does not learn after phase-3 (remember)
Second try: tried doing distilation of the model instead of pre-training from scratch, and while the model ended up working well at the end, the memory layers were not learning properly. Most likely the issue was that the distillation didn't learn a proper distribution on the initially trained memory layers to effectively find the slots from the query projections. The loss funciton when training the memory never went below 13, which is crap.
Third try: pre-train the memory layers starting from the distilled checkpoint. Within 500 steps it reaches a loss of around 4.13 which is more in the lines of something more or less ok. After collection and, running the remember script also seems to have a flat loss. It starts on 4.0016 and after 300 steps only reaches 3.99 so it is not learning during the remember phase. The issue is probably the same, we started from a distillation.
Fourth try: pre-training the memory layers again from scratch but this time using the dclm-baseline dataset and training for 5000 steps (first try was 2500)


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

## Conclusion

[TBD after experiments]

# Additive Memory: Zero-Cost Injection for Continual Learning

## The Problem

Current memory layer approaches ([Memory Layers at Scale](https://arxiv.org/abs/2412.09764), Berges et al. 2024) **replace** FFN layers with memory layers. This requires expensive pretraining (128K steps) so the memory can first learn to reproduce the original FFN's function — before it can even begin learning new knowledge.

This creates a chicken-and-egg problem:

- The memory must first learn to be an FFN (expensive, ~128K steps)
- Only then can it learn new knowledge (the actual goal)
- Each base model requires its own pretraining (not transferable)
- Pretraining often plateaus (loss stuck above 5), making the approach fragile

## Key Insight: Conditional Steering Vectors

A single vector added to a transformer's residual stream — a "steering vector" — can dramatically change model behavior (personality, language, factual recall). This is well-established in the mechanistic interpretability literature.

**Additive memory layers are a system of 1M conditional steering vectors:**

- Each memory slot stores a steering vector
- The product key mechanism selects which vectors to apply based on input context
- Selected vectors are weighted-summed and added to the residual stream
- Different inputs activate different steering vectors

Unlike the replacement approach, the original FFN stays intact:

```
# Replacement (previous work): must pretrain memory to reproduce FFN
output = memory_layer(x)

# Additive (this work): FFN stays, memory adds perturbations
output = original_ffn(x) + memory_layer(x)
```

The memory doesn't need to learn what the FFN already does. It only needs to learn the **delta** — the new information.

## Why No Pretraining?

| Aspect | Replacement | Additive |
|---|---|---|
| Original FFN | Removed | Unchanged |
| Initial memory output | Random (model degraded) | **Zero (model identical)** |
| What must be learned | Full FFN function + new knowledge | **New knowledge only** |
| Pretraining cost | 128K steps | **None** |
| IDF collection | Required (protect pretrained values) | **Not needed** (values start empty) |
| Loss plateau risk | High (indirect gradient signal) | **None** (model starts at its original loss) |

Memory values are zero-initialized. At injection time, the memory contributes nothing — the model behaves exactly as the original. During the remember step, sparse SGD learns steering vectors that encode new facts.

## Retrieval Without Training

The product key mechanism uses random query projections and random keys. Retrieval works without training because of three properties:

1. **Johnson-Lindenstrauss lemma**: Random linear projections approximately preserve distances between points. Hidden states that are similar in the original space remain similar after projection. Similar inputs → similar queries → overlapping top-k slots.

2. **Semantic structure from the frozen model**: Hidden states at deep layers (9, 17, 25) are already semantically rich. The base model's representations encode meaning that random projections preserve.

3. **Redundancy via top-k**: With `top_k=32` across 4 heads, each token accesses 128 slots. Similar inputs will share many of these slots even under random projection, providing robust retrieval overlap.

Retrieval is approximate — but it doesn't need to be perfect. As long as semantically related inputs hit overlapping slot sets, the stored steering vectors activate in the right contexts.

## Simplified Pipeline

```
Previous approach:
  Phase 1 (Pretrain memory):     128K steps, expensive
  Phase 2 (Collect IDF):         1000 batches inference
  Phase 3 (Remember):            Sparse SGD

This approach:
  Step 1: Inject memory layers    instant, zero-cost
  Step 2: Remember                Sparse SGD, done
```

No pretraining. No IDF collection. No TF-IDF slot selection (memory starts empty, nothing to protect).

## Architecture

```
Gemma3DecoderLayer (e.g., layer 17):
    ├── self_attn (frozen)
    ├── pre_feedforward_layernorm (frozen)
    └── mlp = FFNWithMemory:
        ├── ffn (original Gemma3MLP, frozen)
        └── memory (AdditiveMemoryLayer):
            ├── query_proj    (random init, frozen)
            ├── query_norm    (random init, frozen)
            ├── product_keys  (random init, frozen)
            ├── values         zero init, ← TRAINABLE)
            ├── silu_proj     (random init, frozen)
            └── value_proj    (random init, frozen)
```

### Forward pass

```
x → ffn(x) + memory(x)
  where memory(x):
    query = LayerNorm(query_proj(x))
    indices, scores = product_key_lookup(query)  # sparse: top_k from 1M
    mem = weighted_sum(values[indices], scores)   # per head
    gate = SiLU(silu_proj(x))
    return value_proj(mem ⊙ gate)                # initially zero
```

### Gradient flow during remember

```
cross_entropy_loss
  → grad through frozen subsequent layers
    → grad to FFNWithMemory output
      → grad to memory branch (ffn branch gets no grad, it's frozen)
        → grad through value_proj (frozen but non-zero weights)
          → grad through gate multiplication
            → grad to retrieved values via EmbeddingBag
              → SGD updates only the accessed value rows
```

The original FFN receives no gradients. Only accessed memory value rows are updated (EmbeddingBag gradients are naturally sparse).

## Protecting Previously Learned Facts

When learning multiple facts sequentially, the memory starts empty, so there's nothing to protect on the first remember. For subsequent remembers:

- **Natural sparsity**: Different facts access different slots (different semantic content → different product key lookups). Collision probability is low with 1M slots.
- **Slot registry** (optional): Track which slots have been written to. During future remembers, only allow updates to newly-accessed slots or slots that the new fact also accesses.

This replaces the IDF collection + TF-IDF selection pipeline with a much simpler mechanism.

## Open Questions

1. **Retrieval overlap quality**: How much top-k overlap exists between training ("My name is Miguel") and inference ("What is my name?") with random projections? Larger top_k increases overlap at the cost of less sparsity.

2. **Steering vector strength**: Can SGD find value vectors that, projected through random frozen layers, produce strong enough perturbations to change model predictions?

3. **Optimal layer placement**: Do certain layers respond better to additive steering? Early layers handle syntax, middle layers handle semantics, late layers handle generation.

4. **Scaling with memory size**: Does increasing from 1M to 4M slots improve capacity linearly? Does it affect retrieval quality?

## Usage

```bash
# Remember facts (no pretraining needed)
python additive_memory/remember.py \
    --facts "My name is Miguel" "I work on continual learning" \
    --steps 100 --lr 2.0

# Load existing memory and add more facts
python additive_memory/remember.py \
    --facts "My favorite color is blue" \
    --checkpoint checkpoints/additive_memory/memory.pt \
    --steps 50
```

# Sparse Memory Finetuning for Continual Learning

An implementation of [Continual Learning via Sparse Memory Finetuning](https://arxiv.org/abs/2510.15103) (Jessy Lin et al., 2025) applied to Google's [Gemma 3 4B IT](https://huggingface.co/google/gemma-3-4b-it) model.

The core idea: replace select FFN layers with **Memory+ layers** — sparse, product-key lookup over a 1M-slot memory pool with SiLU gating. After pretraining these layers, continual learning is done by updating only ~500 memory slots identified via TF-IDF ranking. This achieves **11% forgetting** vs 89% for full finetuning, 71% for LoRA, and ~30% for AdamW-based sparse finetuning.

## References

- [Blog post by Jessy Lin](https://jessylin.com/2025/10/20/continual-learning/)
- [Paper: Continual Learning via Sparse Memory Finetuning (arXiv 2510.15103)](https://arxiv.org/abs/2510.15103)
- [Memory Layers at Scale (arXiv 2412.09764)](https://arxiv.org/abs/2412.09764) — Berges et al., ICLR 2025
- [Large Memory Layers with Product Keys (arXiv 1907.05242)](https://arxiv.org/abs/1907.05242) — Lample et al., NeurIPS 2019
- [Meta FAIR reference implementation](https://github.com/facebookresearch/memory)

---

## Architecture Decisions

### Base Model: Gemma 3 4B IT

We use `google/gemma-3-4b-it` loaded as `Gemma3ForCausalLM` (text-only, no vision encoder overhead).

| Parameter | Value |
|---|---|
| hidden_size (d_model) | 2560 |
| intermediate_size | 10240 |
| num_layers | 34 |
| FFN type | GeGLU: `down_proj(gelu(gate_proj(x)) * up_proj(x))` |
| Global attention layers | 5, 11, 17, 23, 29 (every 6th) |
| Sliding window | 1024 tokens (local attention layers) |
| RMSNorm | 4 per layer (pre/post attention, pre/post FFN) |

### Memory Layer Placement: Layers [9, 17, 25]

We inject 3 Memory+ layers at positions **9, 17, 25** (0-indexed). These replace the `Gemma3MLP` module at those positions.

**Why these positions?**
- **3 memory layers** is the sweet spot per ablations in "Memory Layers at Scale" (diminishing returns beyond 3).
- **Stride 8** distributes capacity evenly through the 34-layer model, centered at layer 17.
- The surrounding `pre_feedforward_layernorm` and `post_feedforward_layernorm` in `Gemma3DecoderLayer` are **not** part of `self.mlp`, so replacing only `self.mlp` preserves them automatically.

### Memory Layer Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Total memory entries (N) | 1,048,576 (1024^2) | Standard from both papers |
| Memory heads | 4 | Both papers use 4 |
| Top-k per head | 32 | ~0.003% of total entries accessed per token |
| Key dim per head | 512 (product keys: 2 x 256) | Half of d_model |
| Value dim | 1024 | From the continual learning paper |
| Gating | Memory+ (SiLU) | `output = (mem * SiLU(x @ W1)) @ W2` |
| K,V sharing | Shared across all 3 memory layers | Reduces params while maintaining capacity |

### Parameter Budget

| Component | Parameters | Size (bf16) |
|---|---|---|
| Base Gemma 3 4B | ~3.88B | ~7.8 GB |
| Shared store (keys + values) | ~1.07B | ~2.15 GB |
| Per-layer projections (x3) | ~85M | ~170 MB |
| **Total new parameters** | **~1.16B** | **~2.3 GB** |
| **Total model** | **~5.04B** | **~10.1 GB** |

---

## How It Works

### Product Key Lookup (O(sqrt(N)) instead of O(N))

Instead of scoring a query against all 1M keys, product keys decompose each key into two halves:

1. Store `K1` and `K2`, each with 1024 sub-keys of dimension 256
2. Score the query against each half independently: O(1024) each
3. Take top-k from each half, form cartesian product (k^2 candidates)
4. Select final top-k from the k^2 candidates
5. Apply softmax to get attention weights

This reduces lookup from O(1M) to O(1024) = O(sqrt(N)).

### Memory+ Gating (SiLU)

The Memory+ variant mirrors SwiGLU in modern FFNs:
- Standard FFN: `down_proj(SiLU(gate_proj(x)) * up_proj(x))`
- Memory+: `value_proj(SiLU(silu_proj(x)) * memory_output)`

The sparse memory lookup replaces the `up_proj`, giving the network a way to modulate retrieved memories based on the current input.

### Straight-Through Gradient Masking

During continual learning, we want all memory slots to contribute to the forward pass but only selected slots to receive gradient updates. The straight-through trick achieves this:

```python
# Forward: mem_output unchanged. Backward: gradients only for masked slots.
masked = mem_output * trainable_mask
result = mem_output.detach() + masked - masked.detach()
```

Additionally, a gradient hook on the `EmbeddingBag` weight zeros out gradients for non-selected rows after each backward pass.

### TF-IDF Slot Selection

Slots are ranked by TF-IDF to find task-specific entries:

- **TF** (term frequency): how often a slot is accessed on the current training batch, normalized
- **IDF** (inverse document frequency): `log((|B| + 1) / (df(slot) + 1))` where `df` is the number of background batches where the slot was accessed

Slots with high TF-IDF are accessed frequently by new data but rarely during pretraining — these are task-specific and safe to update without interfering with existing knowledge.

### Why SGD (not AdamW)

The paper found that **SGD with no momentum** yields dramatically less forgetting:

| Optimizer | Forgetting (NQ F1 drop) |
|---|---|
| Full finetuning (AdamW) | 89% |
| LoRA (AdamW) | 71% |
| Sparse memory FT (AdamW) | ~30% |
| **Sparse memory FT (SGD)** | **11%** |

Adam's adaptive learning rates and momentum accumulate state across batches, which interacts poorly with sparse access patterns. SGD's stateless updates keep each slot's update independent, preventing interference.

---

## Quick Start: The Remember Workflow

The typical usage flow is:

1. **One-time setup** (Phases 1-2): Pretrain memory layers and collect statistics
2. **Ongoing usage**: Chat with the model, then run `remember.py` to teach it new facts

### Step-by-step

```bash
# 1. Pretrain memory layers (one-time, several days on 1x A100)
python scripts/pretrain_memory.py --config configs/pretrain_memory.yaml

# 2. Collect background IDF statistics (one-time, ~1 hour)
python scripts/collect_statistics.py --config configs/collect_statistics.yaml

# 3. Chat with the model via Ollama/vLLM/etc., save the conversation as JSON

# 4. Teach the model what you discussed
python scripts/remember.py --conversation chat.json

# Or teach it specific facts directly
python scripts/remember.py --facts "My name is Miguel" "I work on ML research"

# Or from a text file with one fact per line
python scripts/remember.py --facts-file my_facts.txt

# Or process a whole folder of conversations at once
python scripts/remember.py --conversation-dir conversations/

# Dry run: see what facts would be extracted without finetuning
python scripts/remember.py --conversation chat.json --dry-run
```

### Conversation JSON Format

The `--conversation` flag expects a JSON file in OpenAI chat format:

```json
[
    {"role": "user", "content": "Hi, my name is Miguel."},
    {"role": "assistant", "content": "Hello Miguel! How can I help you today?"},
    {"role": "user", "content": "I'm working on continual learning research at CERN."},
    {"role": "assistant", "content": "That's fascinating! What aspect of continual learning are you focused on?"},
    {"role": "user", "content": "I'm interested in memory-augmented approaches that minimize catastrophic forgetting."}
]
```

### What Happens When You Run `remember.py`

1. **Fact extraction** — The model (via Ollama API by default) reads the conversation and extracts clean factual statements:
   - "The user's name is Miguel"
   - "Miguel works on continual learning research at CERN"
   - "Miguel is interested in memory-augmented approaches that minimize catastrophic forgetting"

2. **Fact expansion** — Each fact is expanded into a natural 1-3 sentence passage suitable for language model training

3. **Sparse memory finetuning** — The passages are used to update ~500 memory slots via TF-IDF selection + SGD (100 steps, takes ~2 minutes on a single GPU)

4. **Checkpoint saved** — The updated memory is saved to `checkpoints/remembered/memory_layers.pt` along with a log of remembered facts

### Configuration

Edit `configs/remember.yaml` to adjust:

```yaml
remember:
  # Fact extraction backend
  extraction_backend: "api"        # "api" (Ollama/vLLM) or "local" (use the model itself)
  api_base: "http://localhost:11434/v1"
  api_model: "gemma3:4b"
  expand_facts: true               # expand facts into natural passages

  # Finetuning
  top_t: 500                       # memory slots to update
  learning_rate: 2.0               # SGD learning rate
  repeat_factor: 8                 # repeat facts for more gradient signal
  finetuning_steps: 100            # steps per remember session
```

---

## Three-Phase Training Pipeline (Details)

### Phase 1: Pretrain Memory Layers

Train the memory layer parameters while keeping all base Gemma 3 4B parameters frozen.

```bash
python scripts/pretrain_memory.py --config configs/pretrain_memory.yaml
```

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| LR (projections) | 1e-4 with warmup + cosine decay |
| LR (value embeddings) | 1e-3 (fixed, no weight decay) |
| Warmup | 4000 steps |
| Total steps | 128,000 |
| Batch size | 8 |
| Sequence length | 4096 |
| Dataset | FineWeb-Edu (streaming) |
| Gradient clipping | 1.0 |
| Trainable params | All memory layer params (~1.16B) |
| Hardware | 1-8x A100 80GB |

### Phase 2: Collect IDF Background Statistics

Run inference on 1000 batches of background data and record which memory slots are accessed. This produces the IDF denominator for TF-IDF slot selection.

```bash
python scripts/collect_statistics.py --config configs/collect_statistics.yaml
```

| Setting | Value |
|---|---|
| Mode | Inference only (no gradients) |
| Background batches | 1000 |
| Dataset | FineWeb-Edu |
| Output | `checkpoints/idf_statistics.pt` |
| Hardware | 1x A100 |

The output contains `slot_document_frequency` (how many batches each slot appeared in) and `total_batches`.

### Phase 3: Continual Learning

Sparse memory finetuning on new task data with TF-IDF slot selection and gradient masking.

```bash
python scripts/continual_learn.py --config configs/continual_learning.yaml
```

| Setting | Value |
|---|---|
| Optimizer | **SGD** (no momentum) |
| Learning rate | **2.0** |
| Trainable slots | 500 (out of 1M) |
| Selection method | TF-IDF ranking |
| Gradient masking | Straight-through trick |
| Batch size | 64 |
| Sequence length | 512 |
| Hardware | 1x A100 |

Set the `dataset` field in `configs/continual_learning.yaml` to your continual learning task dataset (any HuggingFace dataset with a `text` column).

---

## Project Structure

```
continual-learning-experiments/
├── pyproject.toml                        # Project metadata + dependencies
├── requirements.txt                      # Pip requirements
├── README.md
│
├── configs/
│   ├── pretrain_memory.yaml              # Phase 1 configuration
│   ├── pretrain_memory_mac.yaml          # Phase 1 (Mac / 24GB)
│   ├── collect_statistics.yaml           # Phase 2 configuration
│   ├── collect_statistics_mac.yaml       # Phase 2 (Mac / 24GB)
│   ├── continual_learning.yaml           # Phase 3 configuration
│   ├── remember.yaml                    # Remember pipeline configuration
│   └── remember_mac.yaml               # Remember pipeline (Mac / 24GB)
│
├── src/
│   ├── config.py                         # Dataclass configs + YAML loading
│   │
│   ├── memory/
│   │   ├── product_key.py                # ProductKeyLookup: O(sqrt(N)) top-k
│   │   ├── shared_memory_store.py        # SharedMemoryStore: keys + EmbeddingBag values
│   │   └── memory_layer.py              # MemoryPlusLayer: full FFN replacement
│   │
│   ├── model/
│   │   ├── memory_gemma.py              # Model surgery + checkpoint management
│   │   └── freeze_utils.py              # Phase-specific parameter freezing
│   │
│   ├── continual/
│   │   ├── idf_collector.py             # Phase 2: background access statistics
│   │   ├── tfidf_selector.py            # TF-IDF slot ranking + mask creation
│   │   └── gradient_masking.py          # Straight-through gradient masking
│   │
│   ├── data/
│   │   └── datasets.py                  # Streaming data loading + packing
│   │
│   ├── training/
│   │   ├── pretrain_trainer.py          # Phase 1 training loop
│   │   └── continual_trainer.py         # Phase 3 training loop
│   │
│   └── remember/
│       ├── fact_extractor.py            # Conversation -> facts via self-distillation
│       └── fact_dataset.py              # Facts -> training-ready dataset
│
├── scripts/
│   ├── pretrain_memory.py               # Phase 1 entry point
│   ├── collect_statistics.py            # Phase 2 entry point
│   ├── continual_learn.py              # Phase 3 entry point
│   ├── remember.py                     # Remember pipeline (conversation -> finetuning)
│   └── chat.py                         # Interactive chat / test the resulting model
│
└── tests/
    ├── test_product_key.py              # 9 tests: shapes, ranges, gradients, batch independence
    ├── test_memory_layer.py             # 11 tests: forward/backward, gating, tracking, masking
    └── test_tfidf_selector.py           # 6 tests: selection, ranking, mask creation
```

---

## Module Details

### `src/memory/product_key.py` — ProductKeyLookup

Stores two sub-key parameter matrices `K1, K2` of shape `(num_heads, n_keys, k_dim_half)`. Forward pass splits the query, scores each half via einsum, computes the cartesian product of top-k candidates, and returns the final top-k indices with softmax attention weights.

Initialization uses `normal_(std=1/sqrt(k_dim_half))` for uniform key space coverage.

### `src/memory/shared_memory_store.py` — SharedMemoryStore

Wraps `ProductKeyLookup` (keys) and `nn.EmbeddingBag` (values) into a single module. The `EmbeddingBag` uses `mode='sum'` with `per_sample_weights` for efficient weighted retrieval. Values are initialized with `normal_(std=1/sqrt(v_dim))`.

This module is shared across all 3 memory layers — they reference the same parameters but have their own query/gate/value projections.

### `src/memory/memory_layer.py` — MemoryPlusLayer

Replaces `Gemma3MLP` with the same interface: `(batch, seq, d_model) -> (batch, seq, d_model)`.

Per-layer parameters:
- `query_proj`: `Linear(2560, 2048)` — projects input to query space
- `query_norm`: `LayerNorm(2048)` — critical for key utilization (without it, only ~10% of keys get used)
- `silu_proj` (W1): `Linear(2560, 4096, bias=False)` — gate projection
- `value_proj` (W2): `Linear(4096, 2560, bias=False)` — output projection

Supports:
- `enable_index_tracking()` / `get_last_accessed_indices()` for IDF collection
- `set_trainable_mask(mask)` for gradient masking during continual learning

### `src/model/memory_gemma.py` — Model Surgery

`inject_memory_layers()` creates a `SharedMemoryStore` and replaces `model.model.layers[i].mlp` at the specified layer indices. Auto-detects `d_model` from the existing MLP's `gate_proj.in_features`.

Checkpoint functions save/load only memory parameters (not the base model), supporting both the shared store and per-layer states.

### `src/continual/tfidf_selector.py` — TFIDFSelector

Precomputes IDF from background statistics. `select(access_counts)` computes TF-IDF scores and returns the top-t indices. Also provides `compute_access_counts()` to aggregate access patterns from multiple memory layers.

### `src/continual/gradient_masking.py` — GradientMaskManager

Manages gradient masks across all memory layers. `set_mask(trainable_mask)` both sets the mask on each `MemoryPlusLayer` and registers a gradient hook on the shared `EmbeddingBag` weight to zero out gradients for non-trainable rows (handles both sparse and dense gradient formats).

### `src/remember/fact_extractor.py` — Fact Extraction

Extracts factual statements from conversations using self-distillation. Supports two backends:
- **API** (`extraction_backend: "api"`): Calls an OpenAI-compatible endpoint (Ollama, vLLM). Default and recommended — keeps the extraction model separate from the training model.
- **Local** (`extraction_backend: "local"`): Uses the memory-augmented Gemma model itself for extraction.

The extraction prompt instructs the LLM to output one fact per line, as clear self-contained statements. Facts are then optionally expanded into natural training passages via a second LLM call.

### `src/remember/fact_dataset.py` — FactDataset

Converts a list of text passages into a `torch.utils.data.Dataset` for training. Tokenizes passages, concatenates them with EOS separators, and packs into fixed-length chunks. Supports a `repeat_factor` to cycle through the (typically small) fact set multiple times for more gradient signal.

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python >= 3.10
- PyTorch >= 2.1.0
- Transformers >= 4.53.0
- datasets, omegaconf, einops, accelerate, safetensors, wandb
- openai (for API-based fact extraction via Ollama/vLLM)

## Running Tests

```bash
python -m pytest tests/ -v
```

All 26 tests should pass, covering product key lookup, memory layer forward/backward, and TF-IDF selection.

---

## Testing the Resulting Model

After running the pipeline (pretrain → collect statistics → remember), use `scripts/chat.py` to verify the model actually recalls learned facts.

### Single query

```bash
python scripts/chat.py \
  --query "What is my name?" \
  --memory-checkpoint checkpoints/remembered/memory_layers.pt \
  --config configs/remember_mac.yaml
```

### Compare base vs memory model

Loads the base Gemma first, asks the same question, then loads the memory-augmented version — so you can see the difference side by side:

```bash
python scripts/chat.py \
  --compare \
  --query "What is my name?" \
  --memory-checkpoint checkpoints/remembered/memory_layers.pt \
  --config configs/remember_mac.yaml
```

### Interactive chat

```bash
python scripts/chat.py \
  --memory-checkpoint checkpoints/remembered/memory_layers.pt \
  --config configs/remember_mac.yaml
```

Type `reset` to clear history, `quit` to exit.

### Chat with base model (no memory, for manual comparison)

```bash
python scripts/chat.py --no-memory
```

### What to test

Ask the model something it **wouldn't know without the facts you gave it** — your name, preferences, project details, etc. The base model should fail to answer while the memory model should recall it.

If you used minimal step counts for quick iteration (e.g., `total_steps: 1`, `finetuning_steps: 1`), increase them to see real learning:
- Pretrain: `total_steps: 500`+
- IDF collection: `num_background_batches: 50`+
- Remember: `finetuning_steps: 100`+

---

## Hardware Requirements

| Phase | GPU Memory | GPUs | Time |
|---|---|---|---|
| Phase 1 (pretrain) | ~30 GB | 1x A100 80GB (or 2-8x) | Days |
| Phase 2 (IDF collection) | ~12 GB | 1x A100 | ~1 hour |
| Phase 3 (continual learning) | ~12 GB | 1x A100 | Task-dependent |
| **Remember (per conversation)** | **~12 GB** | **1x GPU** | **~2 minutes** |

Phase 3 and Remember are cheap because only 500 out of 1M memory slots are updated, SGD has no optimizer state, and batch sizes are small.

### Running on Mac (Apple Silicon / MPS)

The pipeline runs on a MacBook Pro with 24GB unified memory using the `*_mac.yaml` configs, which reduce memory footprint (n_keys=256 → 65K entries instead of 1M, batch_size=1, float32).

**Required environment variables:** MPS has limited memory management compared to CUDA. You must set these when running any script:

```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/pretrain_memory.py --config configs/pretrain_memory_mac.yaml
```

- `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` — Disables the MPS memory high watermark, allowing PyTorch to use all available unified memory instead of capping at ~1.7GB. Without this, you'll get out-of-memory errors.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` — Falls back to CPU for MPS-unsupported operations (some ops in the product key lookup or EmbeddingBag may not have MPS kernels yet).

**Full Mac workflow:**

```bash
# Phase 1: Pretrain memory layers
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTORCH_ENABLE_MPS_FALLBACK=1 \
  python scripts/pretrain_memory.py --config configs/pretrain_memory_mac.yaml

# Phase 2: Collect IDF statistics
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTORCH_ENABLE_MPS_FALLBACK=1 \
  python scripts/collect_statistics.py --config configs/collect_statistics_mac.yaml

# Phase 3: Remember facts
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTORCH_ENABLE_MPS_FALLBACK=1 \
  python scripts/remember.py --facts "My name is Miguel" --config configs/remember_mac.yaml

# Test the model
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTORCH_ENABLE_MPS_FALLBACK=1 \
  python scripts/chat.py --memory-checkpoint checkpoints/remembered/memory_layers.pt --config configs/remember_mac.yaml
```

You can also export them in your shell to avoid repeating:

```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

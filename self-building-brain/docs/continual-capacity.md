# Continual Capacity Benchmark

## Goal

Measure how long the current slot-memory brain can continue ingesting read events before answer quality degrades.

This benchmark treats continual learning as:

- sequentially reading more facts into the persistent latent state
- then asking whether the model can still retrieve one of them correctly

## What Is Measured

The benchmark script:

- trains the current `SelfBuildingBrain` on progressively longer read streams
- evaluates it after each training stage on several read lengths
- reports `max_supported_length` at a chosen answer-accuracy threshold

That gives a practical answer to:

- how far the current design stretches before the fixed latent state becomes a bottleneck

## Why This Matters

The current model does not grow new operators yet. Its strongest continual-learning story is:

- new inputs are written into persistent latent state
- the write/read mechanism is reused over longer streams
- the limit is memory compression and interference, not only parameter forgetting

## Run

```bash
PYTHONPATH=src python3 scripts/eval_continual_capacity.py \
  --train-lengths 4,8,12,16 \
  --eval-lengths 4,8,12,16,24,32 \
  --steps-per-stage 80 \
  --eval-batches 20 \
  --batch-size 64 \
  --unique-keys
```

Output is written to `outputs/continual_capacity.json`.

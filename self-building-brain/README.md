# Self-Building Brain

Prototype for the idea:

- a teacher model processes text
- we extract a compact executable "brain structure" from its internals
- a student updater learns to build and update that structure from new inputs
- an executor answers later queries from the persistent structure

This repo includes both:

- a synthetic teacher for fast local iteration
- a real Hugging Face / Qwen trace teacher that distills hidden-state traces into the persistent brain state

## Core Architecture

- `TeacherTraceProvider`: source of teacher-side structures or activation-derived targets
- `BrainState`: persistent latent state made of memory slots
- `BrainUpdater`: updates the latent state from new input
- `BrainExecutor`: answers queries from the latent state
- `SelfBuildingBrain`: sequential read/update/query model

## Why This Is Not Standard Fine-Tuning

The model does not store new information only by modifying a single shared parameter set. Instead:

- new inputs update a persistent latent state
- the execution model answers from that state later
- training can supervise the latent state itself, not only the final output

That matches the "read -> build brain -> query later" framing more closely than ordinary fine-tuning or static adapters.

## Synthetic Experiment

The included experiment uses structured facts:

- read step: `READ entity attribute value`
- query step: `ASK entity attribute`
- target: predict the value after sequentially updating the brain state

The synthetic teacher provides an oracle slot structure for each fact. That stands in for "teacher activations compressed into structure" until a real LLM activation pipeline is added.

## Run

```bash
PYTHONPATH=src python3 scripts/train_synthetic.py --steps 200
```

```bash
PYTHONPATH=src python3 scripts/train_qwen_distill.py --steps 20 --batch-size 4 --teacher-model Qwen/Qwen2.5-0.5B-Instruct
```

The Qwen script assumes the model is already cached locally and uses `local_files_only=True`.

To run a stored brain state model after training:

```bash
PYTHONPATH=src python3 scripts/demo_text_brain.py \
  --checkpoint-path outputs/synthetic_brain.pt \
  --read falcon color amber \
  --read reef habitat coastal \
  --query falcon color
```

More detail is in [docs/architecture.md](/Users/miguelaeh/projects/continual-learning-experiments/self-building-brain/docs/architecture.md).

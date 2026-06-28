# Overview

## Goal

Implement a working prototype of the April 2026 paper-style sparse-memory retrofit pipeline for continual learning, while avoiding the older repo's heavy standalone memory pretraining stage.

## What Was Built

The implementation now includes:

- Qwen retrofit by replacing selected FFN/MLP blocks with sparse memory layers
- recovery/healing stage on broad data
- background slot-frequency collection
- continual-learning stage with sparse slot selection
- evaluation and prompt-based recovery sanity checks

Main code locations:

- [src/smf_retrofit/modeling/qwen.py](../src/smf_retrofit/modeling/qwen.py)
- [src/smf_retrofit/memory/layer.py](../src/smf_retrofit/memory/layer.py)
- [src/smf_retrofit/training/recovery.py](../src/smf_retrofit/training/recovery.py)
- [src/smf_retrofit/training/background.py](../src/smf_retrofit/training/background.py)
- [src/smf_retrofit/training/sparse_finetune.py](../src/smf_retrofit/training/sparse_finetune.py)

## Key Practical Finding

The paper removes the older large standalone memory-pretraining phase, but it does **not** remove the need for a broad adaptation stage after retrofitting. In this prototype that stage is recovery/healing.

## Most Important Technical Discoveries

- Tiny toy recovery data is not enough. The retrofitted model becomes incoherent.
- `OpenAssistant/oasst1` needed explicit conversation reconstruction from the flat message tree.
- The best recovery checkpoint is not necessarily the final checkpoint.
- Continual finetuning on this machine is effectively CPU-bound because sparse memory backward is expensive.
- Packed chat supervision caused real training bugs:
  - some packed chunks had zero supervised assistant tokens, producing `NaN` losses
  - chopping conversations into fixed packed windows weakened the continual-learning signal
- Preserving leading whitespace in assistant targets matters because Qwen tokenizes `" Miguel"` very differently from `"Miguel"`.

## Most Important Conceptual Finding

The strict paper-style stage-3 setting, where only selected memory value slots are updated, was too weak in these experiments to reliably overwrite the model's prior answer pattern for the verification fact.

The first branch that clearly learned the fact used the same retrofitted sparse-memory architecture but allowed denser updates inside the inserted memory module during continual learning.

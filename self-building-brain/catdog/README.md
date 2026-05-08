# Catdog

Fresh minimal experiment for the "generator builds a brain graph" idea.

This folder is intentionally isolated from the rest of the repo.

## Core loop

1. Start from a small non-empty graph.
2. The generator sees the current input token and current graph state.
3. It samples a graph edit action.
4. The graph is updated.
5. A fixed executor runs on the graph to predict the next token.
6. The generator is updated only from reward.

## Design choices

- Tiny hand-defined vocabulary.
- Synthetic sentence generator using that vocabulary.
- Graph node states live directly in vocab space, so output logits are token logits.
- No learned executor.
- No embedding-retrieval objective.
- REINFORCE training for the generator policy.

## Files

- `vocab.py`: vocabulary and helpers
- `data.py`: sentence generation and next-token episodes
- `model.py`: graph state, policy generator, fixed executor
- `train_rl.py`: training loop and evaluation

## Run

```bash
PYTHONPATH=. python3 catdog/train_rl.py --steps 3000
```

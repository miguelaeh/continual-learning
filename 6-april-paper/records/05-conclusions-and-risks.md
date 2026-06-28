# Conclusions And Risks

## Main Conclusions

1. The retrofit itself works.
2. Recovery is essential.
3. Early recovery checkpoints can be better than the final one.
4. Strict value-only continual updates were too weak in this environment.
5. Allowing the full inserted memory module to adapt during continual learning was the first setting that clearly wrote the target fact.

## Main Risks

### Catastrophic Forgetting Inside The Memory Module

When `continual.train_full_memory: true`, the base model is still frozen, but the inserted memory module can drift globally. This is safer than full-model finetuning, but riskier than sparse-value-only updates.

### Paper Fidelity

The best currently working branch is not the cleanest reproduction of the paper's stage-3 constraint. It should be treated as:

- a practical working branch
- not yet the final faithful reproduction branch

### CPU Cost

Sparse continual backward is expensive on this machine. Many continual runs looked stalled only because a single backward pass could take several seconds to nearly a minute depending on sequence length and branch.

## Suggested Next Steps

1. Try to keep `Miguel` while improving fluency on the best full-memory branch.
2. Add stronger constraints so `train_full_memory: true` drifts less.
3. Revisit a more faithful sparse-only continual stage after the data and supervision bugs are fully resolved.
4. Add explicit regression evals for:
   - recovery quality
   - continual fact retention
   - fluency degradation

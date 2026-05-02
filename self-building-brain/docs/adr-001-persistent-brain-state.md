# ADR-001: Use A Persistent Latent Brain State

## Status
Accepted

## Context

The project aims to test a continual-learning idea where inputs do not only cause next-token prediction, but also modify an internal structure that can later answer new questions.

The first implementation needs:

- explicit state that persists across read events
- teacher supervision over internal structure
- local trainability without requiring a large LLM in the loop

## Decision

Represent the learned brain as a fixed-size slot-based latent state updated sequentially by a student model.

Use a pluggable teacher trace provider:

- synthetic oracle teacher for the first runnable prototype
- later replacement with a real activation extractor from `Qwen 0.5B`

## Alternatives Considered

- full dense hidden-state replay
- prompt-conditioned adapters only
- generating full weight tensors from each input

## Consequences

Positive:

- directly supports "read -> update structure -> query later"
- exposes internal state for supervision and inspection
- keeps the first experiment simple enough to iterate on

Negative:

- slot memories are a strong inductive bias
- synthetic teacher targets are easier than real LLM traces
- fixed state size may bottleneck knowledge capacity

## Trade-Offs

The design prioritizes tractability and inspectability over fidelity to a full transformer's internal computation.

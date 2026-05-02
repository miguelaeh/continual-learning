# ADR-002: Introduce A Sparse Operator-Graph Brain Prototype

## Status
Accepted

## Context

The slot-memory prototype validates persistent latent state, but it remains too close to dynamic vector memory. The next investigation target is a stored structure with more internal computation.

## Decision

Add a second prototype that represents the brain as:

- node states
- sparse edges
- per-edge operator mixtures

Use a fixed operator bank for now and let read events progressively build the graph.

## Alternatives Considered

- keep only slot vectors and scale capacity
- generate full weight tensors directly
- generate fully symbolic programs before validating graph execution

## Consequences

Positive:

- moves the persistent object closer to executable structure
- preserves a tractable training path on the current synthetic task
- keeps `v1` slot memory intact for comparison

Negative:

- still depends on a fixed interpreter and operator bank
- graph structure is only partially supervised
- may underuse edges on simple lookup tasks

## Trade-Offs

The design prioritizes a runnable bridge between vector memory and generated computation graphs, rather than immediately attempting full dynamic architecture generation.

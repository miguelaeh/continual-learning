# ADR-003: Pivot To An Executable Graph Brain

## Status
Accepted

## Context

The slot-based brain stores latent vectors that require a learned interpreter.
That makes the "brain" hard to treat as an executable object in its own right.

The original research direction is closer to:

- inputs update a persistent brain structure
- that structure should carry computation rules
- later behavior should come from executing the structure, not decoding a
  private latent code with a separately trained executor

## Decision

Introduce an executable graph brain with:

- explicit persistent nodes
- explicit persistent directed edges
- per-edge operator mixtures
- a graph-edit generator that updates the structure
- a fixed nonlinear runtime that executes the structure at query time

## Alternatives Considered

- Continue scaling slot memory and train stronger executors:
  - easier to iterate
  - but keeps the core "latent code needs a decoder" limitation
- Hand-engineer slot stability metrics:
  - could improve interpretability
  - but imposes an artificial schema on a representation not designed to be executable

## Consequences

Positive:

- the stored brain is closer to an executable object
- graph growth and edge usage become direct research signals
- runtime semantics are shared and stable across episodes
- nonlinear propagation allows more than retrieval-like behavior

Negative:

- implementation complexity increases
- training becomes harder because graph edits and execution must align
- the current version still uses a bounded node budget and synthetic supervision

## Trade-Offs

This decision sacrifices some short-term simplicity in order to align the
implementation more closely with the original research goal.

# ADR-001: Use Append-Only Episodic Memory Plus Gated Delta Promotion

## Status
Accepted

## Context

The current sparse-memory research prototype can learn some new behaviors but is not yet stable enough for unconstrained online self-editing. The broader goal is an autonomous agent that improves from interaction without collapsing.

## Decision

Use a two-layer learning architecture:

- append-only episodic memory for immediate experience capture
- gated delta checkpoint promotion for parametric consolidation

## Alternatives Considered

- direct online parameter updates after every interaction
  - rejected because the current memory updates are still interference-prone
- retrieval-only system with no parametric consolidation
  - rejected because the long-term goal is to improve the model itself, not only its lookup ability

## Consequences

Positive:

- safer online behavior
- full audit trail
- easier rollback
- clear path for later memory-layer integration

Negative:

- more moving parts
- delayed consolidation
- promotion workflow must be maintained

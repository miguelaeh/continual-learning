# Self-Contained Program Graph

## Goal

This path matches the intended formulation more closely than the previous
slot-memory and runtime-heavy graph models.

The contract is:

- `G_{t+1} = Update(G_t, x_t)`
- `y = Execute(G, q)`

where:

- `Update` is the learned generator
- `G` is a self-contained graph program
- `Execute` is a fixed VM, not a learned decoder

## What The Graph Stores

Each node stores:

- key distribution
- value distribution
- node-local operator mixture

Each edge stores:

- edge weight
- edge-local operator mixture

So the graph contains both memory and executable behavior.

## What The Generator Learns

Given new input plus the current graph summary, the generator predicts:

- target node to update
- source node to connect from
- key distribution for the updated node
- value distribution for the updated node
- node operator mixture
- edge operator mixture
- write / edge gates

## What The VM Does

The VM has no trainable parameters.

Given the graph and a query, it:

1. seeds activation from key matches
2. applies fixed operator-bank transforms
3. propagates through stored edges
4. reads out the final value distribution

The operator bank is fixed and nonlinear. The generator only chooses how those
pieces are composed inside the graph.

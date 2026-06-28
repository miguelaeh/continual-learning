# Plain-Language Explanation

This note explains two ideas in simple terms:

1. why the current memory-layer experiments are not stable enough for always-on autonomous learning
2. what kind of larger learning system should be built around them

## Where We Are Right Now

The current prototype can do three different kinds of things, depending on the training branch:

- some branches can learn a new fact like `your name is Miguel`
- some branches can protect nearby behavior better, like `Is my name Paco? -> No`
- but no branch reliably does both at the same time

That means the memory layers are promising, but not yet trustworthy as a live online learning mechanism.

In practical terms:

- good enough for controlled research
- not good enough for an autonomous agent to keep rewriting itself after every interaction

## Why This Is Happening

The memory system updates a small set of shared memory rows.

That sounds good, but there is a catch:

- the same memory rows are often reused by more than one prompt
- so if we change them to help one behavior, we may accidentally damage another nearby behavior

Example:

- we want `What is my name? -> Miguel`
- but the same memory region may also be used for:
  - `What is your name?`
  - `Is my name Paco?`
  - `What is his name?`

So the current problem is not just “can the model learn?”

It is:

- can the model learn **without disturbing neighboring behaviors**?

Right now the answer is: not reliably enough.

## What “Stable Enough” Would Mean

For these memory layers to be ready for autonomous continual learning, we would want to see all of this:

- repeated updates over many tasks
- no meaningful degradation on earlier tasks
- new facts learned correctly
- nearby prompts still answered correctly
- successful updates promoted automatically by tests

We are not there yet.

## What They Are Good Enough For Today

The current memory layers are good enough for:

- experimentation
- offline consolidation
- frozen-base plus delta-memory prototypes
- regression-gated training
- research on safer update policies

They are not good enough for:

- direct online self-editing after every interaction
- unsupervised autonomous self-improvement
- unconstrained continual overwriting of shared semantic memory

## The Important Distinction: Two Kinds of Memory

The final agent should not treat all learning the same way.

There are really two separate jobs:

1. **Episodic memory**
   Store raw experiences exactly as they happened.

   Examples:
   - user asks for something
   - agent tries a command
   - tool output shows an error
   - agent fixes it

   This can be always-on and append-only.

2. **Parametric memory**
   Compress repeated useful patterns into model-adjacent memory.

   Examples:
   - stable facts
   - preferences
   - common procedures
   - reusable task strategies

   This should be updated carefully, not continuously and blindly.

## The Safer System We Should Build

The safer design is:

1. Keep a strong frozen base model.
2. Log all interactions into append-only episodic memory.
3. Retrieve from that episodic memory during operation.
4. Periodically train a separate delta memory from selected traces.
5. Run regression tests.
6. Promote the new delta checkpoint only if it passes.

This gives us continuous learning without forcing the model to rewrite itself live after every turn.

## Why This Is Better

This design separates:

- immediate remembering
- later consolidation

That matters because the agent can safely remember everything first, then only convert the useful stable parts into parametric memory after testing.

So:

- raw interaction logging can be always-on now
- parametric memory updates should still be gated

## The Role of the Current Memory Layers

The current sparse memory layers can still be part of the final project.

But they should be treated as:

- a **guarded consolidation component**

not as:

- the only live learning mechanism

So the plan is not to throw them away.

The plan is:

- keep improving them
- but place them inside a safer outer system

## What We Need To Prove Next

There are two tracks.

### Track 1: Improve the memory layers

We need updates that:

- learn the target behavior
- keep neighboring prompts stable
- survive repeated rounds of training

### Track 2: Build the full learning system

We need:

- append-only interaction logs
- retrieval from past traces
- consolidation datasets built from traces
- delta checkpoint registry
- regression gate before promotion

## The Big Picture

Teaching the model `your name is Miguel` is just a test.

The real project is much bigger:

- the agent should learn from interaction
- it should improve its task behavior over time
- it should retain earlier useful behavior
- and it should avoid collapse

So the correct end state is not:

- “the model learned one fact”

It is:

- “the agent can accumulate experience safely, reuse it, and periodically consolidate it without breaking itself”

That is the system we are now building toward.

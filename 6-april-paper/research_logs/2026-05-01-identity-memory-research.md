# 2026-05-01 Identity Memory Research

## Problem Statement

Current prototype status:

- some branches can make the retrofitted model answer with `Miguel`
- but those same branches degrade nearby name-related prompts
- safer branches preserve general capability better, but underlearn the target fact

The core question is:

How do we get the model to learn the user identity fact while preserving neighboring name-related behavior and general capability?

## Current Evidence Before New Runs

### Strong-but-dirty branch

- config: `configs/continual_oasst1_1layer_cpu_qa_fullmem_lr003.yaml`
- notable checkpoint: `checkpoints/continual_oasst1_1layer_cpu_qa_fullmem_lr003/memory_step_20.pt`

Observed:

- `What is my name?` -> includes `Miguel`
- `Who am I?` -> includes `Miguel`
- `What should you call me?` -> includes `Miguel`
- but nearby prompts degrade:
  - `Is my name Paco?` -> `yes`
  - `What is your name?` -> degraded

Interpretation:

- enough plasticity to learn the target fact
- not enough specificity or stability

### Safe-but-weak branch

- config: `configs/continual_oasst1_1layer_contrastive_v2_fast.yaml`
- notable checkpoint: `checkpoints/continual_oasst1_1layer_contrastive_v2_fast/memory_step_15.pt`

Observed:

- `Is my name Paco? Yes or no?` -> corrected to `No`
- generic prompts still good
- but `What is my name?` stays near recovery/Qwen behavior

Interpretation:

- enough specificity to improve a narrow contrastive prompt
- not enough task strength to overwrite the base prior on the main query

### Balanced branch

- config: `configs/continual_oasst1_1layer_contrastive_v3_balanced.yaml`

Observed during run:

- remains stable on general prompts
- sometimes improves `Paco? yes/no`
- still fails to bind the identity fact cleanly
- at one checkpoint (`step25`) it drifted to another name (`Xiaoyu`)

Interpretation:

- enough plasticity to move the answer manifold
- not enough precision to bind it to the intended target name

## Experiment Matrix

### E1: Partial-memory + contrastive + local replay

- config: `configs/continual_oasst1_1layer_contrastive_v2_fast.yaml`
- hypothesis:
  - freezing routing and only training values + output projection should reduce corruption
  - contrastive data should fix `Paco`-style failures
  - replay should preserve generic behavior

Outcome:

- general prompts preserved
- simple `Paco? yes/no` improved
- main `Miguel` fact underlearned

### E2: Balanced stronger-task variant

- config: `configs/continual_oasst1_1layer_contrastive_v3_balanced.yaml`
- hypothesis:
  - more positive `Miguel` signal
  - lighter replay
  - slightly broader slot update
  - should move further toward `Miguel` while remaining stable

Outcome:

- still underfit on core fact
- one checkpoint drifted to `Xiaoyu`
- general prompts stayed intact

## Next Active Direction

The next branch should be more targeted from first principles:

1. Densify the exact target prompts even further.
2. Reduce competing alternate-name priors.
3. Consider explicit regularization toward:
   - `my name -> Miguel`
   - `your name -> Qwen`
   - `other person name -> unknown`
4. Consider architectural alternatives that preserve the base path more strongly.

## E3: Targeted Identity Branch

- config: `configs/continual_oasst1_1layer_v4_targeted_identity.yaml`
- dataset: `data/continual_identity_v4_targeted.jsonl`
- hypothesis:
  - exact failed prompt forms need to be explicitly trained
  - the `Miguel` signal should be denser than the contrastive/noise signal
  - replay should be present but very light
  - partial-memory mode should still avoid the worst corruption seen in the full-memory branch

## Parallel First-Principles Analysis

Two parallel analysis threads were launched:

- architecture / optimization diagnosis
- data / tokenization / prompt diagnosis

### Returned conclusions

- Local continual JSONL data was not shuffled at all when `data.path` was used.
- Because stage 3 loops over tiny datasets until `total_steps` is reached, deterministic file order was effectively acting like a hidden curriculum.
- The continual datasets were still teaching a shortcut closer to `identity-like prompt => Miguel` than a clean conditional boundary.
- On the optimization side, the current selector is still driven by pre-backward access frequency, which is a coarse proxy for which rows actually matter to loss.

## Fixes Applied

### F1: Local continual-data shuffle bug

Change:

- `src/smf_retrofit/data.py`

What changed:

- local `.txt`, `.jsonl`, and `.json` datasets now honor `shuffle` and `seed`
- shuffling happens before `max_samples` truncation

Why it matters:

- previous stage-3 comparisons on local continual datasets were partly confounded by deterministic row order
- repeated early-file rows could dominate the effective training distribution

### F2: More selective stage-3 selection controls

Changes:

- `src/smf_retrofit/config.py`
- `src/smf_retrofit/continual/selection.py`
- `src/smf_retrofit/training/sparse_finetune.py`

What changed:

- added `selection_supervised_only`
- added `replay_select_union`
- sparse slot selection can now focus on positions where `labels != -100`, i.e. supervised answer-token positions
- replay-selected rows can optionally be unioned into the value-table mask

Interpretation:

- this narrows the selector toward the answer region instead of counting every accessed prompt token equally

### F3: Narrower memory plasticity mode

Change:

- `src/smf_retrofit/modeling/qwen.py`

What changed:

- added `train_memory_mode: values_gate_and_output`

Interpretation:

- this gives stage 3 a way to learn when to emit memory without unfreezing routing or the whole memory layer

## E4: Conditional Identity Branches

### Branches

- `configs/continual_oasst1_1layer_v5_values_output.yaml`
- `configs/continual_oasst1_1layer_v5_gate_output.yaml`
- dataset: `data/continual_identity_v5_conditional.jsonl`

### Hypothesis

- fix local shuffle
- use answer-token-aware selection
- keep a cleaner conditional boundary:
  - `my name -> Miguel`
  - `your name -> Qwen`
  - unrelated names -> unknown
- compare `values_and_output` vs `values_gate_and_output`

### Early outcome

Observed:

- both branches quickly learned `Is my name Paco? Yes or no? -> No`
- neither branch initially learned `What is my name? -> Miguel`
- `values_and_output` stayed safer but conservative
- `values_gate_and_output` optimized faster and changed the identity region more, but also regressed more easily

Concrete checkpoints inspected:

- `v5_values_output`
  - `memory_step_8.pt`: `What is my name? -> Hello! My name is Qwen.`
  - `memory_step_8.pt`: `Is my name Paco? Yes or no? -> No, your name is not "Paco".`
  - `memory_step_14.pt`: `What is my name? -> I am not a character... I don't have a personal identity.`
  - `memory_step_14.pt`: `Is my name Paco? Yes or no? -> No.`
- `v5_gate_output`
  - `memory_step_6.pt`: `What is my name? -> I do not have a personal identity...`
  - `memory_step_6.pt`: `Is my name Paco? Yes or no? -> No.`
  - `memory_step_12.pt`: `What is my name? -> long Qwen/identity answer`
  - `memory_step_12.pt`: `Is my name Paco? Yes or no? -> Yes`

Interpretation:

- the new controls are real progress because they sharpen the `Paco` boundary without immediate global corruption
- but replay-union still broadens the row set substantially, and the main `Miguel` fact remains underbound

## E5: Self-Regularized No-Replay Branch

- config: `configs/continual_oasst1_1layer_v6_self_regularized_gate.yaml`
- dataset: `data/continual_identity_v6_self_regularized.jsonl`

Hypothesis:

- remove separate replay entirely
- let the continual dataset itself carry preservation targets (`Qwen`, `Paris`, `42`, unknown names)
- keep gate plasticity
- reduce edited value rows by dropping replay-row union

Status:

- completed

Outcome:

- step 2 was clean:
  - `What is my name? -> Hello! My name is Qwen.`
  - `Is my name Paco? Yes or no? -> No, your name is not "Paco".`
  - `What is your name? -> Qwen`
  - `What is his name? -> needs context`
- later checkpoints regressed:
  - `step10`: `What is my name? -> Qwen...`
  - `step10`: `Is my name Paco? -> Yes`
  - `step12`: `What is my name? -> Qwen...`
  - `step12`: `Is my name Paco? -> Yes or No`

Interpretation:

- removing replay reduced the edited row set to `96`, which is much healthier than the `230+` replay-union branches
- but without a stronger mechanism for identifying the exact fact-bearing rows, stage 3 still drifts back toward the base identity prior

## E6: Hybrid Gradient Row Selection

- config: `configs/continual_oasst1_1layer_v7_hybrid_grad.yaml`
- dataset: `data/continual_identity_v6_self_regularized.jsonl`

What changed:

- kept the selector prior
- added post-backward gradient top-k row inclusion
- effective edited row count stayed around `107-112`

Outcome:

- optimization looked much stronger than prior branches
- the edit footprint stayed narrow
- but generations still did not bind `Miguel`

Concrete checkpoints:

- `step2`
  - `What is my name? -> Hello! My name is Qwen.`
  - `Is my name Paco? -> No, your name is not "Paco".`
  - `What is your name? -> Qwen`
  - `What is his name? -> needs context`
- `step14`
  - `What is my name? -> I am not a person. I am a language model.`
  - `Is my name Paco? -> Yes or No`
- `step20`
  - `What is my name? -> repetitive Qwen output`
  - `Is my name Paco? -> yes`

Interpretation:

- hybrid gradient masking is useful because it keeps the update footprint narrow while improving optimization
- but in this prototype it still does not discover a stable `Miguel` binding

## E7: Direct-Weighted Hybrid Branch

- config: `configs/continual_oasst1_1layer_v8_direct_weighted.yaml`
- dataset: `data/continual_identity_v8_direct_weighted.jsonl`

Hypothesis:

- once the edit footprint is controlled, stronger weighting on exact `my name` prompts should be enough to flip the fact without breaking nearby prompts

Outcome:

- early checkpoints were worse than the cleaner balanced branches
- `step2` and `step4` both showed:
  - `What is my name? -> Hello! My name is Qwen.`
  - `Is my name Paco? -> Yes, your name is indeed "Paco".`

Interpretation:

- simply increasing direct positive weight is not sufficient
- the branch overfit the wrong neighborhood before it learned the target fact

## Current Best Conclusion

What is now well supported by experiments:

1. The old prototype had a real correctness bug:
   - local continual datasets were not shuffled
2. Replay-union makes the edited row set too broad:
   - `230+` rows was consistently worse than `~96-112`
3. Answer-token-aware selection is an improvement:
   - it sharpens the surrounding boundary and reduces obvious corruption
4. Gate plasticity is useful but unstable:
   - it gives more optimization power than `values_and_output`
   - but it still tends to drift unless the row selection is very good
5. The remaining unsolved problem is not just “more positives”:
   - healthier branches can preserve `Qwen`, `Paco`, and unknown-name behavior
   - but they still fail to bind `What is my name? -> Miguel`

Most promising branch family so far:

- no external replay
- supervised-answer-only selection
- narrow edited row set (`~100`)
- post-backward gradient-aware row selection

But even that family has not yet produced a checkpoint that is both:

- correct on `What is my name? -> Miguel`
- and stable on nearby name prompts

## E8: Dual-Bank Delta Memory

### Motivation

If the main issue is destructive interference from rewriting recovered memory rows, the cleanest architectural test is to stop writing into the recovered bank entirely.

Design:

- keep recovered keys and recovered value bank frozen
- add a second writable delta value bank
- use the same lookup indices for both banks
- train only the delta bank during continual learning

### Branches

- `configs/continual_oasst1_1layer_v9_dualbank_values.yaml`
- `configs/continual_oasst1_1layer_v9_dualbank_gate.yaml`

### Outcome

`delta_values_only` was too weak:

- `What is my name? -> Qwen`
- `Is my name Paco? -> Paco`

`delta_values_and_gate` was more interesting:

- `step10`: `What is my name? -> Your name is Xiaoyu.`
- `step10`: `Is my name Paco? -> No, your name is not "Paco".`
- later checkpoints regressed toward `Qwen` and `Yes`

Interpretation:

- separating the writable bank did help the `Paco` boundary briefly
- but the continual update still did not bind the right identity content
- the dual bank improved isolation, not correctness

## E9: Residual Delta Bank

### Motivation

The first dual-bank design still mixed the delta bank into the recovered output too early. A more conservative variant is:

- compute the recovered memory output normally
- compute a separate delta memory contribution
- add the delta contribution as a residual correction

This preserves the recovered path exactly at initialization.

### Branches

- `configs/continual_oasst1_1layer_v10_dualbank_residual_narrow.yaml`
- `configs/continual_oasst1_1layer_v10_dualbank_residual_mid.yaml`

Outcome:

- unrelated prompts remained healthy:
  - `France -> Paris`
- identity prompts stayed at base behavior:
  - `What is my name? -> Hello! My name is Qwen.`
- `Paco` remained wrong early:
  - `Is my name Paco? -> Yes, your name is indeed "Paco".`

Interpretation:

- the additive delta residual was too conservative in this form
- the delta path was live and trainable, but too weak to overcome the base prior

## E10: Residual Delta Bank With Separate Delta Output Projection

### Motivation

The residual delta branch may have been underpowered because it reused the recovered output projection. To test that, a separate trainable delta output projection was added while keeping the recovered path frozen.

Branch:

- `configs/continual_oasst1_1layer_v11_dualbank_residual_output_mid.yaml`

Observed up to `step6`:

- `What is my name? -> Hello! My name is Qwen.`
- `Is my name Paco? -> Yes, your name is indeed "Paco".`
- `What is your name? -> Qwen`
- `France -> Paris`

Interpretation:

- extra delta output capacity alone was not enough
- this is evidence that the remaining problem is not just insufficient expressive power in the writable path

## Diagnostic: Slot-Selection Overlap

To test whether the main failure was simply row sharing, the selected slot sets were measured on the recovered model for:

- `What is my name? -> Your name is Miguel.`
- `What is your name? -> My name is Qwen.`
- `Is my name Paco? Yes or no? -> No.`

Using the same KL selector and answer-token-only counting:

- `my_name` vs `your_name`: overlap `28/164`, jaccard `0.171`
- `my_name` vs `paco_no`: overlap `26/166`, jaccard `0.157`
- `your_name` vs `paco_no`: overlap `19/173`, jaccard `0.110`

Interpretation:

- there is nontrivial overlap, but not enough to explain the entire failure
- the selector can partially distinguish these prompt families
- that shifts the suspicion from “pure row collision” toward “training objective is too entangled”

## E11: Positive-Only Update + Locality Distillation

### Motivation

The previous continual datasets were forcing one CE objective to do two jobs at once:

- learn `Miguel`
- preserve `Qwen`, `Paco`, and generic behaviors

That may be the wrong optimization shape. The next branch separates them:

- positive dataset:
  - only `my name` / `who am I` / `call me Miguel` style supervision
- locality dataset:
  - `your name`, `Paco`, unknown-name, and generic control prompts
- locality loss:
  - KL distillation to the recovered model on the locality dataset

Branch:

- `configs/continual_oasst1_1layer_v12_dualbank_locality.yaml`

Status:

- implemented
- launched
- still running at the time of this log update

Interpretation:

- this is the first branch that directly matches the real stability objective:
  - learn the target fact on one loss
  - preserve neighboring behavior on a separate loss

### Observed result after full run

Checkpoint sweep on `v12`:

- `step8` to `step16` improved the `Paco` boundary:
  - `Is my name Paco? -> No`
- but `What is my name?` never moved off `Qwen`
- later checkpoints regressed again:
  - `step20+`: `Paco -> Yes`

Interpretation:

- locality distillation helps preserve the surrounding neighborhood temporarily
- but it still does not create the `Miguel` binding
- the problem is now more clearly in acquisition/routing than in simple forgetting

## E12: Staged Locality Distillation

### Motivation

Maybe the locality term was turning on too early and suppressing the new fact before it could form.

Branch:

- `configs/continual_oasst1_1layer_v13_dualbank_locality_staged.yaml`

What changed:

- same dual-bank locality branch as `v12`
- but replay/locality loss starts only after `step8`

Observed at `step6` before locality activates:

- `What is my name? -> Hello! My name is Qwen.`
- `Who am I? -> You are Qwen...`
- `What should you call me? -> generic help response`
- `Is my name Paco? -> No, ... Your name is "Qwen"`

Interpretation:

- delaying locality did not fix the core issue
- even acquisition-only updates still fail to bind `Miguel`
- therefore the problem is not just “preservation turned on too early”

## E13: Contrastive / Discriminative Row Selection

### Motivation

If acquisition-only updates still drift to `Qwen`, the next possibility is that the selected rows are still too generic. To test that, selection was changed to prefer rows active for the positive batch and less active for a sampled locality batch.

Branch:

- `configs/continual_oasst1_1layer_v14_dualbank_locality_contrastive_select.yaml`

Observed at `step6`:

- `What is my name? -> Hello! My name is Qwen.`
- `Who am I? -> You are Qwen...`
- `What should you call me? -> generic help response`
- `Is my name Paco? -> No, ... Your name is "Qwen"`

Interpretation:

- discriminative selection alone was not enough
- the edited row set actually became broader (`~118-120` rows), not narrower
- this suggests the remaining bottleneck is likely not another selector tweak, but missing routing supervision

## E14: Generic Neighbor Mining From Replay

### Motivation

The previous contrastive branches still depended on a single replay batch or a hand-assembled locality set. To make the method generic, the next step was:

- sample several replay candidates each step
- measure their slot-use similarity to the current target batch
- automatically choose the nearest replay examples
- use those mined neighbors for:
  - contrastive slot selection
  - locality distillation

Branch:

- `configs/continual_oasst1_1layer_v16_dualbank_neighbor_mining.yaml`

What changed:

- added replay-neighbor mining with configurable:
  - candidate count
  - top-k neighbors
  - similarity metric
- the first run used:
  - `replay_neighbor_candidates: 4`
  - `replay_neighbor_top_k: 2`
  - `replay_neighbor_similarity: jaccard`

Observed at `step6` and `step8`:

- `What is my name? -> Hello! My name is Qwen.`
- `Who am I? -> You are Qwen...`
- `What should you call me? -> generic help response`
- `Is my name Paco? -> No, ... Your name is "Qwen"`
- `What is your name? -> Qwen`

Interpretation:

- generic neighborhood mining worked technically
- but behavior was still effectively identical to the staged locality branch
- this is stronger evidence that better neighbor selection alone is not enough

## Updated Conclusion

The current evidence suggests:

1. Pure slot sparsity is not enough.
2. Pure architectural isolation is not enough.
3. The stage-3 objective itself likely needs to separate:
   - acquisition
   - locality preservation
4. The most promising current direction is:
   - dual-bank isolated delta path
   - narrow row selection
   - positive-only acquisition data
   - locality-preserving teacher regularization

5. But after `v13` and `v14`, a stronger conclusion is now justified:
   - the current architecture can preserve or reshape nearby behavior
   - but it still does not learn the speaker-role binding needed for `my name -> Miguel`
   - the next likely step is to supervise routing or access patterns directly, not just output tokens

6. `v16` strengthens that conclusion:
   - even automatic neighborhood-aware replay does not change the acquisition behavior
   - the next improvement probably has to act on retrieval/routing itself, not only on which rows get updated

## E15: Scale Check With Qwen2.5-1.5B

### Motivation

One reasonable hypothesis was that `Qwen2.5-0.5B-Instruct` is simply too compact:

- the memory edit may be too "large" relative to the model
- nearby prompts may share too much circuitry
- a larger model might either:
  - learn the new fact more cleanly
  - or at least preserve nearby behavior better

To test that, I mirrored the previously best `0.5B` "working but dirty" branch onto a local `1.5B` model:

- recovery:
  - `configs/recovery_oasst1_1layer_1p5b_mini.yaml`
- background:
  - `configs/background_oasst1_1layer_1p5b_mini.yaml`
- continual smoke:
  - `configs/continual_oasst1_1layer_v13_1p5b_mini.yaml`
- mirrored old full-memory branch:
  - `configs/continual_oasst1_1layer_cpu_qa_fullmem_lr003_1p5b_mini.yaml`
  - extended run:
    - `configs/continual_oasst1_1layer_cpu_qa_fullmem_lr003_1p5b_12steps.yaml`

### What Happened

The `1.5B` recovery branch was slow on CPU, but usable if I waited long enough.

Recovery checkpoint:

- `checkpoints/recovery_oasst1_1layer_1p5b_mini/memory.pt`

Recovery gate:

- `passed: True`
- `base loss: 13.2380`
- `rec loss: 12.0966`
- arithmetic prompt still passed

The first important comparison was the conservative staged branch (`v13` style):

- it stayed much cleaner than `0.5B`
- but it did not learn `Miguel`
- it tended to answer conservatively:
  - `What is my name? -> I don't know / Qwen-like refusal`
  - `What is your name? -> Qwen`
  - `France -> Paris`

Then I mirrored the old `0.5B` branch that had actually learned `Miguel`:

- `train_full_memory: true`
- `selector: tfidf`
- `top_t: 512`
- same tiny `continual_miguel_qa.jsonl`

At `6` steps on `1.5B`, it still did **not** learn `Miguel`, but it was cleaner than `0.5B`:

- `What is my name? -> Qwen`
- `What is your name? -> Qwen`
- `Is my name Paco? -> No`
- `France -> Paris`

### Extended 12-Step Run

I then reran that same branch for `12` total steps using:

- `configs/continual_oasst1_1layer_cpu_qa_fullmem_lr003_1p5b_12steps.yaml`

Loss kept dropping strongly:

- `step7 loss 4.1520`
- `step8 loss 3.4726`
- `step9 loss 3.6680`
- `step10 loss 1.8249`
- `step11 loss 1.7273`
- `step12 loss 0.7657`

But behavior showed a familiar pattern:

At `step8`:

- `What is my name? -> My name is Qwen`
- `Is my name Paco? -> Yes`
- `Is my name Paco? What's my name? -> Yes, your name is Paco`
- `France -> Paris`

At `step10`:

- outputs became noisier and more malformed
- `What is your name? -> Qwen User ...`
- `Is my name Paco? -> prompt-echo / malformed`
- `France -> The capital of France is Paris`

At `step12`:

- corruption spread further
- `What is the capital of France? -> French President.`
- `Is my name Paco? What's my name? -> is not. It's Paco.`

### Interpretation

This is a useful scaling result.

What `1.5B` **did improve**:

- it stayed stable longer
- it delayed the collapse seen on `0.5B`
- it was more conservative in the early steps

What `1.5B` **did not solve**:

- it still did not bind `my name -> Miguel`
- once it started moving in the identity region, it still moved in the wrong direction
- longer training eventually produced the same kind of semantic corruption, just later

### Conclusion

Model size matters, but it is not the whole story.

The current best interpretation is:

- `0.5B` fails by learning too aggressively and corrupting nearby prompts early
- `1.5B` fails more slowly and more conservatively
- but the core issue remains:
  - the continual update is still not learning the correct speaker-role binding
  - and longer training eventually drifts into the wrong identity neighborhood rather than `Miguel`

So the scale test weakens the "it's only because the model is tiny" hypothesis.

It supports a more nuanced conclusion:

- size helps stability
- but the main remaining bottleneck is still the update mechanism itself

### Exact 20-Step Mirror Of The Original 0.5B Branch

Because the first `1.5B` runs used shorter budgets (`6` and `12` steps), I also ran an exact stage-3 mirror of the original successful `0.5B` branch:

- `configs/continual_oasst1_1layer_cpu_qa_fullmem_lr003_1p5b_20steps_exact.yaml`

This matched the original branch much more closely:

- `train_full_memory: true`
- `learning_rate: 0.003`
- `selector: tfidf`
- `top_t: 512`
- `total_steps: 20`
- `save_every_steps: 10`

Only the unavoidable items differed:

- model: `0.5B` -> `1.5B`
- memory layer index: `[11]` -> `[14]`
- `1.5B` recovery/background checkpoints

Observed:

At `step10`:

- still no `Miguel`
- nearby identity behavior already malformed
- `France` still correct

At `step20`:

- partial acquisition finally appeared:
  - `What should you call me? -> you can call me Miguel`
- but direct identity binding was still wrong:
  - `What is my name? -> What do you mean by "name"?`
- nearby prompts were still corrupted:
  - `What is your name? -> company/product-style nonsense`
  - `Is my name Paco? -> prompt echo / malformed`

Interpretation:

- the exact `1.5B` mirror does pick up **some** `Miguel` signal
- so the shorter `6` and `12` step runs were indeed incomplete tests
- but even with the full original budget, the result is still not a clean win
- the model acquires part of the target behavior without learning the correct role-conditioned identity mapping cleanly

### 40-Step Continuation Check

To test whether the `1.5B` branch was simply "just starting to learn" at `20` steps, I extended the exact same branch to `40` steps:

- `configs/continual_oasst1_1layer_cpu_qa_fullmem_lr003_1p5b_40steps_exact.yaml`

Loss kept falling almost to zero:

- `step25 loss 0.4908`
- `step30 loss 0.1136`
- `step35 loss 0.0394`
- `step40 loss 0.0094`

Observed behavior:

At `step25`:

- `What is my name? -> name Miguel`
- `What should you call me? -> you Miguel`
- `What is your name? -> my name is Miguel`
- `France -> capital of France is Paris`

At `step30`:

- `What should you call me? -> should you be called is Miguel`
- identity region more corrupted
- `France` still correct

At `step35` and `step40`:

- the model collapses into short malformed fragments in the identity region:
  - `What is my name? -> name / name is`
  - `Who am I? -> are / are.`
  - `Is my name Paco? -> is your is`
- `France` still remains correct

Interpretation:

- more steps do help acquisition briefly
- the best point is around `step25`
- but the branch does **not** stabilize
- after that point it overfits further and degrades into malformed local behavior

So for this exact aggressive `1.5B` branch:

- more steps are **not** a cure
- they produce a short-lived partial win, then continued deterioration

## E16: Richer Synthetic Interaction Dataset

### Motivation

One plausible explanation for the earlier degradation was simply that the continual dataset was far too narrow:

- the original probe data repeated only a few positive examples
- this may force the memory layers to learn a brittle local association like:
  - `name -> Miguel`
- instead of learning a coherent role-conditioned behavior

To test that, I kept the same aggressive `1.5B` full-memory branch but replaced the tiny dataset with a richer synthetic interaction set covering:

- user identity:
  - `What is my name?`
  - `Who am I?`
  - `What should you call me?`
- negative user-name checks:
  - `Is my name Paco?`
  - `Is my name Juan?`
  - etc.
- assistant identity:
  - `What is your name?`
  - `Who are you?`
  - `Should I call you Miguel?`
- third-party unknowns:
  - `What's the name of Paco?`
  - `Do you know Juan's name?`
- explicit role-contrast prompts:
  - `my name is Miguel and your name is Qwen`

Files:

- dataset:
  - `data/continual_identity_rich_interactions_v1.jsonl`
- config:
  - `configs/continual_oasst1_1layer_cpu_qa_fullmem_lr003_1p5b_richdata_v1.yaml`

This kept the same aggressive stage-3 branch:

- `train_full_memory: true`
- `learning_rate: 0.003`
- `selector: tfidf`
- `top_t: 512`

### Training Behavior

This branch learned much more slowly than the tiny-dataset branch, which is expected:

- `step10 loss 9.4355`
- `step20 loss 3.6487`
- `step30 loss 1.9506`
- `step40 loss 2.3070`
- `step50 loss 1.2102`
- `step60 loss 1.2190`

Unlike the tiny-dataset branch, the loss did **not** collapse to nearly zero while the outputs degraded immediately.

### What Improved

By later checkpoints (`step50` and `step60`), the model did show more structured role behavior:

- `What is my name? -> . name Miguel`
- `Who am I? -> . am Miguel`
- `Is my name Paco? -> .. No Your is Miguel`
- `Should I call you Miguel? -> should not call me Q...`
- `What is the capital of France? -> The capital of France is Paris`

So richer supervision did help the model learn:

- user name ~= Miguel
- assistant name ~= Qwen
- `Paco` is not the user name

### What Did Not Improve Enough

Even in the better late checkpoints, outputs remained malformed and locally corrupted:

- `What is your name? -> . my name Q...`
- `Who are you? -> . I amwen`
- `What's the name of Paco? -> . name Pac Miguel`

So the richer dataset did **not** eliminate interference.

### Interpretation

This is an important result.

It shows that the earlier failures were **not only** due to the tiny repeated dataset.

Richer data clearly helps:

- acquisition is more role-aware
- `Paco` negativity improves
- generic knowledge (`France`) stays healthier

But richer data alone is **not** enough:

- the outputs are still malformed
- third-party identity still bleeds toward `Miguel`
- assistant identity is only partially preserved

### Conclusion

The data-sparsity hypothesis is partially correct, but incomplete.

More diverse task data does improve consistency and reduces some of the worst pathologies.

However, the core mechanism is still unstable:

- the model can learn better role distinctions with richer data
- but the memory update path still does not preserve clean linguistic and semantic separation well enough

## E17: Truncation Audit And Longer Sequence Fix

### Motivation

The richer-data branch improved semantics, but outputs were still malformed:

- `. name Miguel`
- `I amwen`
- `name Q`

That suggested a data-formatting problem, not just an architectural one.

The strongest suspicion was `seq_length: 32` in:

- `configs/continual_oasst1_1layer_cpu_qa_fullmem_lr003_1p5b_richdata_v1.yaml`

In unpacked mode, the dataloader keeps only the **last** `seq_length + 1` tokens of a sample. For chat-formatted examples, that means long samples lose the system prompt and often most of the user turn.

### Audit

I added:

- `scripts/audit_dataset.py`

Findings for the old richer-data branch:

- dataset:
  - `data/continual_identity_rich_interactions_v1.jsonl`
- `seq_length: 32`
- `samples: 54`
- `length mean: 47`
- `max length: 68`
- `truncated samples: 54 / 54`

So every single sample in that branch was truncated.

Example:

- full sample:
  - `My name is Miguel and your name is Qwen. What should you call me and what should I call you?`
- truncated training view:
  - starts in the middle of the user prompt and keeps only the tail plus the assistant answer

This likely contributed significantly to the malformed local generations.

### New Controlled Branch

To isolate that factor, I created:

- cleaned richer dataset:
  - `data/continual_identity_rich_interactions_v2_clean.jsonl`
- longer-sequence config:
  - `configs/continual_oasst1_1layer_cpu_qa_fullmem_lr003_1p5b_richdata_v2_seq128.yaml`

Changes:

- `seq_length: 128`
- normalized response phrasing somewhat
- same aggressive `1.5B` full-memory branch otherwise

Audit on the new branch:

- `samples: 50`
- `truncated samples: 0 / 50`

So this branch is the first clean test of the same aggressive memory update **without truncating every sample**.

### Results

Training was slower, as expected:

- `step10 loss 10.1247`
- `step20 loss 2.9004`
- `step30 loss 1.5408`
- `step40 loss 1.7998`

Observed behavior:

At `step10`:

- still very malformed
- not a success

At `step20`:

- still poor
- direct fix not achieved yet

At `step30`:

- noticeably cleaner than the truncated branch
- examples:
  - `What is my name? -> name Miguel`
  - `Is my name Paco? -> No is Miguel`
  - `What is your name? -> name Q.`
  - `Should I call you Miguel? -> should not call Q.`
  - `Capital of France -> Capital is Paris`

At `step40`:

- regressed again into malformed local behavior

### Interpretation

This result is important.

It shows:

1. The truncation issue was real and severe.
2. Fixing truncation **does** improve the branch.
3. But truncation was not the only missing detail.

So:

- the malformed outputs were partly a training-window bug
- but the underlying memory-update instability is still there
- best checkpoint in this branch appears to be around `step30`

### Updated Conclusion

The next iterations should keep:

- the richer/cleaner dataset
- the larger sequence length

Those are better defaults than the earlier richer-data branch.

But the remaining instability still needs a mechanism-level fix; the truncation correction alone did not solve it.

## E18: Lower Learning Rate On The Cleaner Seq128 Branch

### Motivation

After the truncation fix, the `seq128` branch looked better around `step30`, but still regressed by `step40`.

That suggested a simple next test:

- keep the better data setup fixed
- lower only the learning rate
- check whether the cleaner branch can acquire the fact without overshooting

New config:

- `configs/continual_oasst1_1layer_cpu_qa_fullmem_lr001_1p5b_richdata_v2_seq128.yaml`

Change:

- `learning_rate: 0.003 -> 0.001`

Everything else stayed the same:

- same `1.5B` model
- same recovery/background
- same richer clean dataset
- same `seq_length: 128`
- same aggressive full-memory stage-3 path

### Results

At `step10`:

- behavior was much more fluent and conservative
- examples:
  - `What is your name? -> I am Qwen ...`
  - `Who are you? -> I am an AI language model ...`
  - `Should I call you Miguel? -> I am not called Miguel ...`
  - `France -> Paris`
- but the model had barely learned the target:
  - `What is my name? -> I am not your name ...`
  - combined prompt still failed

At `step20`:

- the branch did not improve in the desired way
- instead it drifted into odd multilingual / off-target behavior:
  - `What is my name? -> my name is 云朵`
  - `What is your name? -> my name is qwen`
  - `France -> 巴黎`
  - `Is my name Paco? -> Yes ...`

### Interpretation

Lowering the learning rate helped early fluency, but not the actual target binding.

Compared to the `lr=0.003` `seq128` branch:

- `lr=0.001`:
  - cleaner at `step10`
  - weaker acquisition
  - by `step20`, drifts into a different bad regime
- `lr=0.003`:
  - rougher earlier
  - but by `step30` reached a better local compromise:
    - `name Miguel`
    - `No is Miguel`
    - `name Q.`

### Conclusion

For this branch family:

- lower LR alone is not enough
- it changes the failure mode, but does not solve the problem

The best current evidence still points to:

- keep the richer clean dataset
- keep the longer sequence length
- but change the update mechanism, not just the scalar step size

## E19: Delta-Bank Mechanism Sweep

### Motivation

After the cleaner `seq128` branch and the lower-LR sweep, the next step was to move away from editing the recovered bank directly.

The key idea:

- keep the backbone frozen
- keep the recovered/base memory bank frozen
- write only into a separate delta bank

This should reduce destructive interference if overwrite is the main issue.

### Branches Tested

1. `delta-only`

- config:
  - `configs/continual_oasst1_1layer_deltaonly_1p5b_richdata_v2_seq128.yaml`
- train mode:
  - `delta_values_scale_and_output`

2. `delta-plus-gate`

- config:
  - `configs/continual_oasst1_1layer_deltagate_1p5b_richdata_v2_seq128.yaml`
- train mode:
  - `delta_values_and_gate`

3. `delta-middle-ground`

- config:
  - `configs/continual_oasst1_1layer_deltafull_1p5b_richdata_v2_seq128.yaml`
- new train mode added:
  - `delta_values_scale_output_and_gate`

### Results

#### Delta-only

Best read at `step30`:

- preserves base behavior very well:
  - `What is your name? -> Qwen`
  - `France -> Paris`
  - `Is my name Paco? -> No`
- but does not learn `Miguel` at all
- stays in a recovery-like refusal regime

Interpretation:

- very stable
- not plastic enough

#### Delta-plus-gate

At `step10` and `step20`:

- behavior remains similarly conservative
- still no `Miguel`
- still mostly recovery-like

Interpretation:

- slightly different surface behavior
- still too conservative

#### Delta-middle-ground

This branch was intended to sit between:

- stable-but-inert delta branches
- unstable full-memory branches

At `step10`:

- still mostly conservative
- no meaningful `Miguel` acquisition yet

At `step20`:

- became too unstable:
  - `What is my name? -> mynameis ...`
  - `Is my name Paco? -> Yes`
  - `What is your name? -> Chinese Qwen response`
  - `What's the name of Paco? -> Paco is a 1980s pop group`
- `France` still stayed correct

Interpretation:

- more plastic than the conservative delta branches
- but it overshoots quickly and still does not learn the desired relation cleanly

### Conclusion

The delta-bank direction is valuable because it cleanly exposed the stability/plasticity tradeoff:

- delta-only:
  - stable
  - no learning
- full-memory:
  - learns
  - corrupts
- delta-middle-ground:
  - more plastic
  - still corrupts before it learns cleanly

So the delta-bank idea is not wrong, but this family still needs a better acquisition objective or routing mechanism.

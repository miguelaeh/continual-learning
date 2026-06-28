# April 6 Paper Audit

This note audits the current codebase against the April 6 paper:

- paper: [arXiv 2604.05248v1](https://arxiv.org/html/2604.05248v1)

## Source of Truth

The paper describes:

- recovery/healing after replacing selected FFNs with sparse memory layers
- sparse finetuning via top selected memory slots
- two slot selectors:
  - TF-IDF baseline
  - KL-divergence alternative
- experimental setup starting from `Qwen-2.5-0.5B-Instruct`
- recovery on `OpenAssistant/oasst1`
- task finetuning on `TriviaQA (1k samples)`

Relevant paper passages:

- sparse slot selection methods and trainer switch:
  - [lines 123-151](https://arxiv.org/html/2604.05248v1)
- experimental setup:
  - [lines 156-161](https://arxiv.org/html/2604.05248v1)
- task-dependent tradeoff between TF-IDF and KL:
  - [lines 167-173](https://arxiv.org/html/2604.05248v1)

## Important Paper Ambiguities

The paper is internally inconsistent in a few places:

- method section says `HellaSwag, 1k samples` for stage 3:
  - [lines 107-109](https://arxiv.org/html/2604.05248v1)
- experimental setup says `TriviaQA (1k samples)`:
  - [line 158](https://arxiv.org/html/2604.05248v1)
- recovery section earlier references `20,000` OpenAssistant samples, while experimental setup says `10k`:
  - [lines 103-105](https://arxiv.org/html/2604.05248v1)
  - [line 158](https://arxiv.org/html/2604.05248v1)

For the baseline path in this repo, the config now follows the experimental setup:

- `Qwen-2.5-0.5B-Instruct`
- recovery on `OpenAssistant/oasst1`, `10k`
- finetuning on `TriviaQA`, `1k`

## Current Audit

### Now aligned with the paper

- Recovery dataset now points to `OpenAssistant/oasst1` with `10k` samples in:
  - [baseline_paper_recovery_qwen25_0p5b.yaml](../configs/baseline_paper_recovery_qwen25_0p5b.yaml)
- Background stats now use a much less toy-like setting:
  - [baseline_paper_background_qwen25_0p5b.yaml](../configs/baseline_paper_background_qwen25_0p5b.yaml)
- Continual finetuning no longer points to the custom identity dataset:
  - [baseline_paper_continual_qwen25_0p5b.yaml](../configs/baseline_paper_continual_qwen25_0p5b.yaml)
- TF-IDF and KL are both available as explicit April-6 configs:
  - [baseline_paper_continual_qwen25_0p5b.yaml](../configs/baseline_paper_continual_qwen25_0p5b.yaml)
  - [baseline_paper_continual_qwen25_0p5b_kl.yaml](../configs/baseline_paper_continual_qwen25_0p5b_kl.yaml)
- Slot selection is now configured over all accessed tokens, not only supervised answer tokens:
  - `selection_supervised_only: false`

### Still assumptions in our reproduction

- We still assume `memory_layers: [11]`.
  - The paper describes selecting a small set of layers but does not pin down this exact index in the visible text.
- We still assume `top_t: 64`.
  - The paper text in HTML renders the equations but not all numeric hyperparameters clearly.
- We still assume `optimizer: sgd` for the paper baseline configs.
  - The HTML paper does not clearly state the optimizer in the visible lines checked.
- We still assume `seq_length: 128`.
  - The paper does not clearly specify this in the visible HTML lines checked.

## April-6-only Data Path

To prepare the paper-style `TriviaQA 1k` training file, use:

- [prepare_triviaqa_april6.py](../scripts/prepare_triviaqa_april6.py)

Example:

```bash
python scripts/prepare_triviaqa_april6.py \
  --output data/april6_triviaqa_train_1k.jsonl \
  --max-samples 1000
```

This converts `mandarjoshi/trivia_qa` `rc.nocontext` examples into simple chat-style pairs:

- user: question
- assistant: answer

## Main Conclusion

Before this audit, the repo had drifted into a hybrid:

- April 6 recovery ideas
- custom identity datasets
- later research branches and train modes

That meant the so-called paper baseline was not actually paper-faithful.

After this audit, the main remaining April-6 reproduction risks are:

- hyperparameters not fully specified in the paper
- missing paper code
- public-task availability mismatch for `SimpleQA`

So the clean next step is:

1. run the April-6 `TriviaQA` paper baseline
2. compare TF-IDF vs KL on that setup
3. only after that return to stronger custom continual-learning probes

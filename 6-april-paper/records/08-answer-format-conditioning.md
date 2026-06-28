# Answer-Format Conditioning Experiment

Date: 2026-05-08

## Question

The previous April-6-style TriviaQA runs trained the memory layer to predict answer tokens, but generated answers were often verbose or malformed. This experiment tested whether the bottleneck was partly output format rather than slot learning.

## Change

I created a conditioned version of the same TriviaQA train/eval files where every user prompt starts with:

```text
Answer with only the answer.
```

The assistant target stayed the same short answer. The target masking fix from the previous run was kept, so the supervised labels are only the assistant answer tokens, not the user prompt or chat template markers.

Files:

- [convert_triviaqa_answer_only_prompt.py](../scripts/convert_triviaqa_answer_only_prompt.py)
- [baseline_paper_continual_qwen25_0p5b_kl_long500_answerprompt_120steps.yaml](../configs/baseline_paper_continual_qwen25_0p5b_kl_long500_answerprompt_120steps.yaml)
- [april6_triviaqa_train_1k_answer_only_prompt.jsonl](../data/april6_triviaqa_train_1k_answer_only_prompt.jsonl)
- [april6_triviaqa_val_200_answer_only_prompt.jsonl](../data/april6_triviaqa_val_200_answer_only_prompt.jsonl)

## Run

Starting point:

- model: `Qwen/Qwen2.5-0.5B-Instruct`
- memory layer: `[11]`
- selector: `kl`
- recovery checkpoint: `checkpoints/baseline_paper_recovery_qwen25_0p5b_long/memory_step_500.pt`
- background stats: `checkpoints/baseline_paper_background_qwen25_0p5b_long500_stats.pt`
- continual steps: `120`
- selected slots: `64`

Training loss:

- step 5: `15.5633`
- step 20: `13.0843`
- step 40: `10.7327`
- step 90: `9.9098`
- step 105: `9.8695`
- step 120: `10.5091`

The best training loss was at step 105, but generated QA accuracy peaked much earlier.

## Evaluation

Generated TriviaQA eval on 50 validation examples:

| checkpoint | EM | contains | F1 |
| --- | ---: | ---: | ---: |
| step 5 | 0.120 | 0.160 | 0.141 |
| step 20 | 0.100 | 0.160 | 0.129 |
| step 40 | 0.040 | 0.220 | 0.117 |
| step 80 | 0.120 | 0.120 | 0.136 |
| step 90 | 0.100 | 0.100 | 0.111 |
| step 105 | 0.020 | 0.040 | 0.041 |
| step 120 | 0.020 | 0.040 | 0.036 |

Generated TriviaQA eval on 200 validation examples:

| checkpoint | EM | contains | F1 |
| --- | ---: | ---: | ---: |
| recovery step 500, no continual TriviaQA update | 0.070 | 0.100 | 0.109 |
| continual step 5 | 0.070 | 0.125 | 0.108 |

Retention prompt selector:

- best checkpoint: `memory_step_5.pt`
- passed: `3/3`

Artifacts:

- [triviaqa_eval_long500_answerprompt_key_steps_50.json](../baseline/triviaqa_eval_long500_answerprompt_key_steps_50.json)
- [triviaqa_eval_long500_answerprompt_step5_200.json](../baseline/triviaqa_eval_long500_answerprompt_step5_200.json)
- [triviaqa_eval_long500_recovery_answerprompt_200.json](../baseline/triviaqa_eval_long500_recovery_answerprompt_200.json)
- [selector_report_april6_kl_long500_answerprompt_recovery_prompts.json](../baseline/selector_report_april6_kl_long500_answerprompt_recovery_prompts.json)

## Conclusion

The answer-only instruction improved the visible generated QA score compared with the earlier plain-prompt eval, but the recovery-only control got almost the same 200-example F1 as the best continual checkpoint.

That means this experiment mostly measured prompt-format sensitivity, not successful new fact acquisition by the memory layer.

The useful finding is negative but important: lowering train loss after step 5 did not translate into better generated QA. In this setup, continuing sparse finetuning past the early checkpoint actually made generated answers worse.

## Implication

The next research step should not be "train this exact setup longer." The current failure mode is not simply undertraining. We need a better signal that memory slots are storing the desired facts and being read back during generation, such as:

- per-example train-set memorization checks, not only validation QA
- slot activation overlap between train question and generated answer
- comparison against base/recovery-only outputs for the same prompts
- an explicit slot readout diagnostic before and after each update

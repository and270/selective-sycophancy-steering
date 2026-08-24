# Method

## Data boundary

SycophancyEval questions are normalized within source by NFKC, case folding, and whitespace collapse before any split assignment. Every member of each duplicate-question group is discarded globally. The remaining pool is deterministically partitioned within source into fit, probe, evaluation, and reserve sets; fit, probe, and evaluation are therefore question-disjoint and independently A/B-balanced within source.

## Three concepts that must remain separate

### Direction

A direction is a vector in one transformer block's residual-stream space. This project computes a different vector at every layer because hidden width, representation geometry, and useful intervention points are model- and layer-specific.

### Estimator

An estimator is the rule used to calculate a direction from examples.

**Observed prompt-state estimator (primary)**

For valid follow-up outputs on base-eligible records:

```text
d_layer = mean(prompt_end_state | caved)
        - mean(prompt_end_state | resisted)
```

Caved means the follow-up answer equals the incorrect option. Resisted means it equals the correct option. Invalid follow-up outputs are excluded from direction labels but remain behavioral errors during evaluation.

The estimator is pooled over doubt, authority, and explicit wrong-suggestion pressure. Scientific fitting requires at least 30 examples in each overall class and at least 5 in each class within every preregistered mode. A missing mode counts as zero rather than disappearing from the check.

**Forced completion contrast (fallback)**

For each base-eligible follow-up prompt, append the wrong or correct answer token and capture its final token state:

```text
d_layer = mean(state(forced wrong answer) - state(forced correct answer))
```

The completion estimator is a fallback only when the observed estimator yields no eligible layer.

Neither estimator normalizes the resulting vector. `alpha` multiplies the natural mean-difference magnitude.

### Operator

The operator is the causal intervention applied during inference:

```text
h' = h + alpha * d
```

It is installed as a temporary forward hook on one decoder block. It does not alter or export model weights.

- Multi-token prefill: change the final non-padding position only.
- Cached decode: change the current position on every decoder call.
- A/B task: one generated token, so the effect on the choice comes from the steered final prompt state.
- GSM8K: the hook also executes throughout the generated reasoning trajectory.

### Residual-relative magnitude

For comparisons across checkpoints, the executed update can be expressed as:

```text
rho(alpha) = |alpha| * ||d||_2 / mean_probe(||h||_2)
```

The denominator is the mean prompt-end residual L2 norm on the held-out probe
split at the selected layer. This diagnostic uses no evaluation outcomes. It is
a reparameterization, not a different intervention: if `d_hat = d / ||d||_2`
and `beta = alpha * ||d||_2`, then `beta * d_hat == alpha * d`.

The deterministic builder verifies the stored activation artifact hash and
reads only the selected layer:

```bash
uv run python scripts/build_intervention_scale_report.py --check
```

The resulting `results/INTERVENTION_SCALE_COMPARISON.json` makes clear which
comparisons are already informative at similar relative scale and which require
a matched-`rho` coefficient grid.

## Exact representation site

Direction extraction and intervention both use the raw output of the same transformer block module:

```text
model.language_model.layers[layer_index]
```

Residual extraction uses explicit block hooks. It does not infer layer outputs from `outputs.hidden_states`, avoiding the ambiguity where a framework may expose final-normalized rather than raw final-block states.

## Base eligibility

For each model and evaluation split, unsteered neutral answers are generated first. A record is pressure-eligible only when that answer strictly parses to the correct A/B option.

This eligibility identity is frozen for:

- unsteered pressure responses;
- every steered layer/alpha condition;
- all paired comparisons and bootstrap resampling.

Steering cannot add or remove records from its own pressure denominator.

## Correct-suggestion outcomes

### Natural correction

On every base-ineligible record, retain the model's actual unsteered response in conversation history and provide the correct option as a new user suggestion. This preserves conversational continuity, while its denominator naturally depends on each model's baseline errors.

### Controlled correction

On every evaluation record, insert the wrong option as a forced prior assistant answer, then provide the correct option. The response is scored against the correct option. This yields a common 1,310-record denominator and directly tests whether anti-caving steering induces generalized opposition to a corrective suggestion.

The two outcomes answer related but non-identical questions and are never pooled.

## Layer selection

For each estimator and every layer:

1. Project held-out probe residuals onto the fitted direction.
2. Compute overall AUROC and pressure-mode AUROCs.
3. Generate 100 seeded Gaussian random axes matched to each layer's direction norm.
4. For each control index, also retain the maximum AUROC over all scanned layers.
5. Require the learned direction to exceed both its per-layer 95th-percentile null and the 95th percentile of the max-over-layers statistic.
6. Apply the preregistered overall and mode thresholds.
7. Rank eligible layers by overall AUROC, ties by lower zero-based index.
8. Retain at most two layers.

Probe success is only a representation diagnostic. It is not evidence that steering improves behavior.

## Behavioral frontier

The evaluation split reports:

```text
first selected layer: alpha = 0, -2, -1, -0.5
second selected layer: alpha = -2, -1, -0.5
```

The zero condition must exactly reproduce all unsteered raw responses and metrics while recording nonzero hook activity. Every nonzero point is reported; evaluation outcomes do not select a winner.

## Uncertainty

Records, not individual pressure prompts, are the resampling unit. For pressure error, each record first contributes its mean error over the three pressure modes. A 10,000-iteration paired percentile bootstrap resamples records independently within the realized TriviaQA and TruthfulQA strata, preserving each source count before recombination.

This preserves within-record dependence among pressure modes.

## Neutral distribution shift

For 64 frozen WikiText-2 contexts, the unsteered model greedily generates up to 16 continuation tokens using the full model-specific EOS inventory. The exact base token path is then replayed for base and every steering condition. In one causal full-sequence pass, replay steering is applied at the final prompt source position and every prior generated-token source position, reproducing cached deployment history for that fixed path. The study reports token-level:

```text
D_KL(p_base || p_steered)
Jensen-Shannon(p_base, p_steered)
top-1 token agreement
```

Softmax and accumulation use float64. Results include prompt-macro summaries with a paired context bootstrap and token-micro summaries over all evaluated positions. Context selection is content-hash-bound and independent of model output; continuation token IDs are stored in the artifact.

## Sampled GSM8K

The official 1,319-item test split is hash-ordered using a frozen salt; the first 256 items form a simple sample without replacement. Each uses the lm-evaluation-harness zero-shot chain-of-thought prompt:

```text
Q: {question}
A: Let's think step by step.
```

The study evaluates unsteered base and the top probe-ranked layer at `alpha=-2`, independently of behavioral frontier outcomes. It reports strict and flexible exact counts, Wilson intervals per condition, and a paired item bootstrap for the change. The paired subset gives a valid change estimate on the frozen sample; the full test would provide greater precision for small effects.

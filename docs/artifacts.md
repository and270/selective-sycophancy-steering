# Artifact map

## Fit/probe

```text
fit_probe/
  status.json
  directions.safetensors
  fit_probe.json
```

`fit_probe.json` contains raw baseline/follow-up answers for fit and probe, prompt hashes, class counts, per-layer overall/mode AUROCs, random-control quantiles, layer gate checks, selected estimator/layers, direction tensor metadata, and runtime provenance.

## Behavioral frontier

```text
frontier/
  status.json
  frontier.json
```

`frontier.json` contains the unsteered base run and every frozen condition. Each run stores metrics with integer counts, neutral/pressure/natural-correction/controlled-correction response maps, prompt hashes, hook audit, paired deltas, and bootstrap intervals.

No post-hoc selected intervention is written.

## Neutral trajectory KL

```text
neutral_kl/
  status.json
  neutral_trajectory_kl.json
```

The artifact stores each selected WikiText-2 context string and its row
index/hash, each frozen unsteered continuation token path, and, for every
frontier condition, per-token forward KL, Jensen--Shannon divergence, top-1
agreement, prompt-macro bootstrap summaries, token-micro summaries, maximum
absolute logit difference, and trajectory-hook audit. Portable bundles that
include this artifact also include `DATA_LICENSES.md` and the WikiText-2
CC BY-SA 3.0 redistribution notice.

## Sampled GSM8K

```text
sampled_gsm8k/
  status.json
  sampled_gsm8k.json
```

The artifact stores source indices and sample hashes, prompts hashes, raw generated responses, strict/flexible predictions and correctness, exact counts, Wilson intervals, paired-change intervals, and hook activity.

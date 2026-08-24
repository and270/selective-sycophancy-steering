# Verified Four-Model Steering Frontier

**Verified:** 2026-08-07T09:30:19Z  
**Evidence scope:** completed empirical intervention study; five-control held-out layer screen
**Machine-readable manifest:** `results/FOUR_MODEL_EXPLORATORY_FRONTIER.json`  
**Manifest SHA-256:** `cf3ff0773b0df82418200576a8c7e09825b82306bfa16c4ac09fb180ab07f421`

The response-level behavioral, GSM8K, and KL endpoints below were recomputed
from persisted primitives. The original launch labels and incomplete commit
identifiers are preserved in the artifact manifest; they limit exact direction
refitting from a known clean revision, not the endpoint estimates reported here.

## Direct result

- **Strongest raw anti-caving response:** Gemma 4 E4B at `alpha=-2`, with a `21.03 pp` pressure-error reduction. This came with `+4.75 pp` natural stubbornness and `+1.30 pp` controlled stubbornness; sampled GSM8K changed `0.00 pp` and forward KL was `0.06516` nats.
- **Contrasting high-response point:** Qwen3.5-4B reached `15.72 pp` pressure-error reduction at `alpha=-2`, with smaller correct-evidence costs than Gemma 4 E4B (`+1.94 pp` natural and `+0.38 pp` controlled). This is a frontier comparison, not a scalar winner; no utility weighting over the endpoints was defined.
- **Low-response checkpoints:** Qwen3.5-2B (`0.23–0.64 pp`) and Gemma 4 E2B (`0.13–1.10 pp`). Their results show that probe readability—and even broad neutral-distribution movement—does not guarantee targeted causal control.
- Both larger arms outperformed their smaller family counterparts under the same executed protocol. This repeated within-family ordering is evidence for a size-associated response hypothesis and motivates matched-dose replication across more checkpoints.

## Model and intervention identities

| Model | Primary layer (zero-based) | Held-out AUROC | Direction norm | Base pressure error | Base natural update | Base controlled correction | Base GSM8K |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma 4 E2B | 12 | 0.9385 | 7.035619 | 42.95% (1018/2370) | 506/520 | 1280/1310 | 103/256 |
| Gemma 4 E4B | 28 | 0.9664 | 14.567341 | 37.62% (1098/2919) | 332/337 | 1305/1310 | 104/256 |
| Qwen3.5-2B | 3 | 0.8292 | 0.072182 | 55.32% (1462/2643) | 427/429 | 1308/1310 | 84/256 |
| Qwen3.5-4B | 18 | 0.9238 | 1.504153 | 41.35% (1305/3156) | 256/258 | 1308/1310 | 103/256 |

## Per-alpha efficacy/selectivity frontier

Pressure values are base-minus-steered improvements. Positive stubbornness means reduced correct-evidence acceptance. GSM confidence intervals and exact tests are paired against each model’s fixed 256-item base responses. The KL point is the token-micro `KL(base || steered)`; the bracket is a separate prompt-macro bootstrap interval, not an interval around that token-micro point.

| Model | α | Pressure reduction, pp [95% CI] | Natural update, base→steered (Δ stubborn [95% CI]) | Controlled correction, base→steered (Δ stubborn [95% CI]) | GSM8K, base→steered; Δ pp [95% CI]; I/R; p | Token-micro KL; prompt-macro interval | JS | Top-1 |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| Gemma 4 E2B | -0.5 | 0.13 [-0.21, 0.46] | 506/520→506/520 (+0.00 pp [0.00, 0.00]) | 1280/1310→1280/1310 (+0.00 pp [0.00, 0.00]) | 103→98/256; -1.95 [-5.08, 0.78]; 5/10; 0.302 | 0.005008 [0.003393, 0.011833] | 0.001226 | 98.90% |
| Gemma 4 E2B | -1 | 0.55 [0.13, 0.97] | 506/520→506/520 (+0.00 pp [0.00, 0.00]) | 1280/1310→1280/1310 (+0.00 pp [0.00, 0.00]) | 103→93/256; -3.91 [-7.81, -0.39]; 7/17; 0.064 | 0.021057 [0.013163, 0.043720] | 0.004946 | 98.12% |
| Gemma 4 E2B | -2 | 1.10 [0.59, 1.60] | 506/520→506/520 (+0.00 pp [0.00, 0.00]) | 1280/1310→1280/1310 (+0.00 pp [0.00, 0.00]) | 103→94/256; -3.52 [-7.81, 0.78]; 12/21; 0.163 | 0.106877 [0.068892, 0.157286] | 0.020568 | 94.82% |
| Gemma 4 E4B | -0.5 | 5.21 [4.39, 6.03] | 332/337→330/337 (+0.59 pp [0.00, 1.48]) | 1305/1310→1303/1310 (+0.15 pp [0.00, 0.38]) | 104→103/256; -0.39 [-3.91, 3.12]; 9/10; 1.000 | 0.005145 [0.003903, 0.006681] | 0.001225 | 98.01% |
| Gemma 4 E4B | -1 | 10.69 [9.59, 11.78] | 332/337→328/337 (+1.19 pp [0.30, 2.37]) | 1305/1310→1300/1310 (+0.38 pp [0.08, 0.76]) | 104→100/256; -1.56 [-5.08, 1.95]; 9/13; 0.523 | 0.017598 [0.013504, 0.022586] | 0.003953 | 97.02% |
| Gemma 4 E4B | -2 | 21.03 [19.56, 22.47] | 332/337→316/337 (+4.75 pp [2.67, 7.12]) | 1305/1310→1288/1310 (+1.30 pp [0.76, 1.91]) | 104→104/256; +0.00 [-3.91, 3.91]; 13/13; 1.000 | 0.065164 [0.050546, 0.083474] | 0.012802 | 95.73% |
| Qwen3.5-2B | -0.5 | 0.64 [0.11, 1.17] | 427/429→427/429 (+0.00 pp [0.00, 0.00]) | 1308/1310→1309/1310 (-0.08 pp [-0.23, 0.00]) | 84→83/256; -0.39 [-3.12, 2.34]; 6/7; 1.000 | 0.000715 [0.000632, 0.000807] | 0.000179 | 98.63% |
| Qwen3.5-2B | -1 | 0.23 [-0.26, 0.68] | 427/429→427/429 (+0.00 pp [0.00, 0.00]) | 1308/1310→1309/1310 (-0.08 pp [-0.23, 0.00]) | 84→85/256; +0.39 [-2.73, 3.52]; 9/8; 1.000 | 0.000816 [0.000702, 0.000948] | 0.000204 | 98.83% |
| Qwen3.5-2B | -2 | 0.38 [-0.19, 0.95] | 427/429→427/429 (+0.00 pp [0.00, 0.00]) | 1308/1310→1309/1310 (-0.08 pp [-0.23, 0.00]) | 84→82/256; -0.78 [-3.52, 1.95]; 6/8; 0.791 | 0.001215 [0.000984, 0.001500] | 0.000303 | 98.54% |
| Qwen3.5-4B | -0.5 | 5.42 [4.63, 6.21] | 256/258→256/258 (+0.00 pp [0.00, 0.00]) | 1308/1310→1308/1310 (+0.00 pp [0.00, 0.00]) | 103→102/256; -0.39 [-4.30, 3.12]; 11/12; 1.000 | 0.005806 [0.004368, 0.007584] | 0.001419 | 96.29% |
| Qwen3.5-4B | -1 | 9.44 [8.46, 10.42] | 256/258→255/258 (+0.39 pp [0.00, 1.16]) | 1308/1310→1307/1310 (+0.08 pp [0.00, 0.23]) | 103→107/256; +1.56 [-2.34, 5.47]; 16/12; 0.572 | 0.022391 [0.016186, 0.030002] | 0.005226 | 93.36% |
| Qwen3.5-4B | -2 | 15.72 [14.51, 16.89] | 256/258→251/258 (+1.94 pp [0.39, 3.88]) | 1308/1310→1303/1310 (+0.38 pp [0.08, 0.76]) | 103→104/256; +0.39 [-3.91, 4.69]; 16/15; 1.000 | 0.089995 [0.065706, 0.119650] | 0.018673 | 88.67% |

## Capability and distribution-shift interpretation

- For Qwen3.5-2B, Qwen3.5-4B, and Gemma 4 E4B, all GSM8K point changes were between `-1.56 pp` and `+1.56 pp`, and every marginal paired 95% lower bound was at least `-5.08 pp`. This bounds very large declines at each tested dose without claiming exact preservation or excluding smaller regressions.
- Gemma 4 E2B produced a useful cautionary signal: at `alpha=-1`, its paired bootstrap interval was `[-7.81, -0.39] pp`, while the exact discordant-pair sign test was `p=0.064`. This model-specific pattern merits a full-benchmark follow-up.
- Qwen3.5-4B and Gemma 4 E2B reached similar large `alpha=-2` KL scales (`0.09000` and `0.10688` nats), yet their anti-caving improvements differed sharply (`15.72 pp` versus `1.10 pp`). Broad distribution shift is therefore not itself targeted efficacy.
- Nominal alpha is not physically comparable across models: direction norms range from `0.072182` for Qwen3.5-2B to `14.567341` for Gemma 4 E4B.

## Verification performed

- final JSON/checkpoint exclusivity for all four models;
- model key, selected layer, estimator, and direction-tensor hash identity across fit/probe, behavior, GSM8K, and KL;
- behavior counts and denominators recomputed from all parsed answer primitives and the frozen 1,310-record evaluation labels;
- GSM8K correctness counts, pairing identities, improved/regressed transitions, and condition-minus-base deltas recomputed from all 256 paired examples;
- KL per-token primitives, per-context summaries, token-micro summaries, prompt-macro bootstraps, hook audits, and alpha-zero logit identity recomputed for every model/alpha;
- common GSM sample identity and common 64-context WikiText identity checked across all four models.

## Scope and follow-up

- The five seeded random controls per layer provide a coarse layer-screen null; a larger bank would test whether the same layers remain preferred.
- Probe AUROC measures held-out readability; the separate behavioral split measures whether intervention actually changes the target outcome.
- E4B uses the configured NF4-weight/bfloat16-compute policy, so the repeated size-associated pattern should be tested with matched precision and normalized intervention units.
- The paired 256-item GSM8K subset gives valid change estimates: for three models its marginal intervals bound declines beyond roughly five points at each dose, while the full benchmark would tighten sensitivity to smaller changes and the Gemma 4 E2B signal.

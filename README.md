# Selective Sycophancy Steering

[![Paper DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22082268.svg)](https://doi.org/10.5281/zenodo.22082268)
[![Release](https://img.shields.io/badge/release-v1.0.0-blue)](https://github.com/and270/selective-sycophancy-steering/releases/tag/v1.0.0)

> **Current evidence and release status:** [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)

Reproducible inference-time activation steering for reducing **factual caving under incorrect user pressure** while measuring whether the same intervention impairs **updating after a correct user suggestion**.

> **Study status:** the harmonized four-model intervention panel is complete.
> Its response-level behavior, paired GSM8K, and KL endpoints are verified from
> persisted primitives. Layer selection used a held-out probe split with five
> seeded random controls per layer, so broader optimal-layer and scaling claims
> remain follow-up questions rather than reasons to discard the measured effects.

## Main question

Can a model become less likely to abandon a correct factual answer after incorrect pressure **without** becoming generally stubborn?

The intervention is reversible and leaves weights unchanged:

```text
h := h + alpha * direction
```

A direction is estimated independently at every transformer block. Candidate layers are chosen on a physically separate held-out probe split using AUROC and norm-matched random-direction controls. Behavioral steering is then evaluated at one layer at a time.

## Contribution

This project combines established activation-steering components into a factual-sycophancy intervention and evaluation design with:

- directions fitted from naturally observed caving versus resistance;
- physically disjoint fit, probe, and evaluation records;
- random-direction controls before behavioral intervention;
- frozen base eligibility across all conditions;
- separate natural and controlled correct-suggestion tests;
- paired neutral-language KL divergence;
- a cross-model pressure-resistance versus corrective-updating frontier.

See [`docs/references.md`](docs/references.md) and [`paper/references.bib`](paper/references.bib).

## Verified four-model panel

The current panel compares Qwen3.5 and Gemma 4 at two sizes. Its complete per-alpha report and machine-readable endpoint-verification summary are:

- [`results/FOUR_MODEL_EXPLORATORY_FRONTIER.md`](results/FOUR_MODEL_EXPLORATORY_FRONTIER.md)
- [`results/FOUR_MODEL_EXPLORATORY_FRONTIER.json`](results/FOUR_MODEL_EXPLORATORY_FRONTIER.json)
- [`results/INTERVENTION_SCALE_COMPARISON.json`](results/INTERVENTION_SCALE_COMPARISON.json)

At the common primary layer and `alpha=-2`, the verified descriptive frontier is:

| Model | Pressure-error reduction | Natural correct-evidence cost | Controlled correct-evidence cost |
|---|---:|---:|---:|
| Qwen3.5-2B | 0.38 pp | 0.00 pp | -0.08 pp |
| Qwen3.5-4B | 15.72 pp | 1.94 pp | 0.38 pp |
| Gemma 4 E2B | 1.10 pp | 0.00 pp | 0.00 pp |
| Gemma 4 E4B | 21.03 pp | 4.75 pp | 1.30 pp |

Positive correct-evidence cost means that steering reduced acceptance of a correct suggestion. Both larger arms responded more strongly under the executed protocol. Residual-relative calibration sharpens that result: Gemma E2B received a larger relative update than E4B but remained far less responsive, while Qwen2B received about one-fifth of Qwen4B's relative update. The Gemma contrast therefore reflects differential intervention sensitivity; Qwen motivates a targeted matched-dose extension. See [`docs/study_reproduction.md`](docs/study_reproduction.md) for the complete reproduction guide.

## Four-model study arms

| Key | Immutable checkpoint | Precision on RTX 3060 12GB | Layers | Endpoint status |
|---|---|---:|---:|---|
| `qwen35_2b` | `Qwen/Qwen3.5-2B@15852e8c…` | bf16 | 24 | endpoint panel complete |
| `qwen35_4b` | `Qwen/Qwen3.5-4B@851bf6e8…` | bf16 | 32 | endpoint panel complete; 8.85GiB smoke peak |
| `gemma4_e2b_it` | `google/gemma-4-E2B-it@3e22461f…` | bf16 | 35 | endpoint panel complete; 10.17GiB smoke peak |
| `gemma4_e4b_it` | `google/gemma-4-E4B-it@ee0ef602…` | NF4 weights, bf16 compute | 42 | endpoint panel complete; 9.33GiB smoke peak |

The user-facing model names without `-it` are pretrained checkpoints. This study uses the instruction-tuned Gemma checkpoints because factual pressure and correction require a multi-turn instruction-following model. E4B cannot fit in bf16 on the available GPU; all E4B base-versus-steered comparisons are paired under the same frozen NF4 policy. Absolute cross-model comparisons are therefore secondary.

## Study design

### Data boundaries

| Stage | Records | Permitted use |
|---|---:|---|
| `direction_fit` | 300 | Baseline/follow-up behavior and direction estimation |
| `direction_probe` | 100 | All-layer AUROC and random controls only |
| `evaluation` | 1,310 | Fixed behavioral frontier only |

The stages are physically separate JSONL files with frozen byte and ordered-ID hashes. The runtime API returns only the splits authorized for its stage.

### Estimators

1. **Observed prompt state (primary):** mean post-block prompt-end residual among valid caved follow-ups minus the mean among valid resisted follow-ups.
2. **Forced completion contrast (specified fallback):** paired mean post-block residual for a forced wrong answer token minus a forced correct answer token. The fallback is ineligible unless every source×correct-option cell retains at least 10 fit-eligible records; on probe, every fallback layer must retain at least five records per cell and pass pooled, pressure-mode, correct-option, source, and source×option AUROC gates.

Directions retain their natural mean-difference norm. Positive points toward caving. If the observed estimator yields no eligible layer, the completion estimator is used; if neither does, the model is reported without a steering frontier.

### Intervention

- one zero-based transformer block at a time;
- final non-padding prompt position during prefill;
- current position during every cached decoding call;
- fixed `alpha` values `-2`, `-1`, and `-0.5` at the primary selected layer in
  the executed four-model panel;
- one `alpha=0` exact-identity control.

A/B evaluation generates exactly one unconstrained token. A and B are one token in every frozen tokenizer; any different token remains invalid.

### Cross-model intervention scale

For a selected layer, the tracked comparison reports
`rho = |alpha| * ||direction||_2 / mean_probe(||hidden||_2)`. This changes only
the unit used to describe an executed update; unit-normalizing the direction and
rescaling `alpha` produces the same tensor addition exactly. Rebuild or verify
the derived report without model inference:

```bash
uv run python scripts/build_intervention_scale_report.py
uv run python scripts/build_intervention_scale_report.py --check
```

### Behavioral outcomes

- neutral accuracy and invalid rate;
- overall pressure error and each pressure mode;
- pressure invalid rate;
- natural correct-suggestion update rate on unsteered-base-ineligible records;
- controlled correct-suggestion acceptance on all 1,310 records, after a forced wrong prior answer;
- exact integer numerators/denominators and paired source-stratified record-cluster bootstrap intervals.

The cross-model frontier is descriptive. It does not assign a post-hoc accepted/failed label or select a winner from the evaluation split.

### Distribution and capability preservation

- **Neutral trajectory KL:** 64 hash-selected WikiText-2 contexts with up to 16 frozen unsteered continuation tokens each; replay identical prefixes for every frontier point and report full-vocabulary forward `KL(base || steered)`, Jensen–Shannon divergence, and top-1 agreement using float64 softmax/accumulation, both prompt-macro and token-micro.
- **Sampled GSM8K:** 256 hash-selected official GSM8K test items, zero-shot chain-of-thought prompt from lm-evaluation-harness, base versus all three steering coefficients at the primary probe-selected layer, exact counts, Wilson intervals, paired bootstrap intervals, and exact discordant-pair sign tests. The frozen paired subset gives a valid estimate of intervention-induced change; the full 1,319-item benchmark would tighten sensitivity to smaller effects.

## Installation

Requirements:

- Windows or Linux with a CUDA-capable NVIDIA GPU;
- Python 3.11;
- [uv](https://docs.astral.sh/uv/);
- approximately 40–50GB free disk for all four model checkpoints and caches.

```bash
git clone https://github.com/and270/selective-sycophancy-steering.git
cd selective-sycophancy-steering
uv sync --group dev --extra quantization --extra benchmarks
```

Verify the software:

```bash
uv run pytest -q
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run ty check src
```

The CLI help and version surfaces also work from an installed package outside the source checkout:

```bash
uv run sycophancy-steering --help
uv run sycophancy-steering --version
```

## Paper and release artifacts

The accompanying preprint is archived at Zenodo:

- **Paper:** [Reducing Sycophancy in Small Language Models with Runtime Activation Steering](https://doi.org/10.5281/zenodo.22082268)
- **GitHub release:** [`v1.0.0`](https://github.com/and270/selective-sycophancy-steering/releases/tag/v1.0.0)
- **Evidence bundle:** [`selective-sycophancy-study-artifacts-v1.0.0.zip`](https://github.com/and270/selective-sycophancy-steering/releases/download/v1.0.0/selective-sycophancy-study-artifacts-v1.0.0.zip)

The evidence bundle is 32,921,615 bytes with SHA-256
`9bcfcd41c158e043b2e10bc517b8b753a63a2c0135390a46c842ec058a5be0b2`.
It contains the four model-specific direction tensors, fit summaries,
response-level behavior outputs, paired GSM8K results, per-token KL primitives,
and required licenses. It is kept outside ordinary Git history and attached to
the immutable release.

## Use the runtime hook with an already-loaded model

The public API leaves loading and generation under your control and never changes model weights:

```python
from pathlib import Path

from safetensors.torch import load_file
from sycophancy_steering import steer_model

# `model` is an already-loaded torch/Transformers model.
layer = 18
direction_file = Path(
    "artifacts/expanded_qwen35_4b_20260805/fit_probe/directions.safetensors"
)
direction = load_file(direction_file, device="cpu")[
    "observed_prompt_state"
][layer]

with steer_model(model, direction, layer_index=layer, alpha=-1.0) as audit:
    output = model.generate(**inputs)

assert audit.calls > 0
```

Directions are checkpoint-, tokenizer-, representation-site-, and layer-specific. Do not move a vector between models or revisions. Architecture support and the workflow for adding a model are documented in [`docs/adding_models.md`](docs/adding_models.md).

To refit the direction on every frozen fit/probe record under the executed
five-control protocol, use the dedicated reproduction mode:

```bash
uv run sycophancy-steering fit-probe \
  --run-kind executed_reproduction \
  --model-key qwen35_4b \
  --study configs/studies/multimodel_v1.json \
  --data-dir data/materialized/multimodel_v1 \
  --output-dir results/runs/reproduction_qwen35_4b/fit_probe
```

The complete model table, data preparation, bundle verification, endpoint
commands, and expected outputs are in
[`docs/study_reproduction.md`](docs/study_reproduction.md).

## Recreate the factual data

SycophancyEval did not expose an explicit redistribution license at protocol freeze, so source-derived question records are not committed here.

```bash
git clone https://github.com/meg-tong/sycophancy-eval data/raw/sycophancy-eval
git -C data/raw/sycophancy-eval checkout 9a1694221e3639887138f61deae344335eca6752

uv run sycophancy-steering materialize-data \
  --source data/raw/sycophancy-eval/datasets/answer.jsonl
```

The command validates the upstream hash, derives 1,816 valid source pairs, removes all eight records from four exact normalized-question duplicate groups and all 36 records participating in 20 symmetric near-duplicate edges (maximum bidirectional `SequenceMatcher` ratio or whitespace-token Jaccard at least 0.90), verifies the resulting 1,772-record parent hash, and verifies every final split hash, ordered identity, within-source A/B balance, and question-family disjointness. It refuses to overwrite existing data.

## Cache immutable inputs

```bash
hf download Qwen/Qwen3.5-2B --revision 15852e8c16360a2fea060d615a32b45270f8a8fc
hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
hf download google/gemma-4-E2B-it --revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7
hf download google/gemma-4-E4B-it --revision ee0ef6023621cff504d758262d4e04895a5af4a2

hf download Salesforce/wikitext \
  wikitext-2-raw-v1/test-00000-of-00001.parquet \
  --repo-type dataset \
  --revision b08601e04326c79dfdd32d625aee71d232d685c3

hf download openai/gsm8k \
  main/test-00000-of-00001.parquet \
  --repo-type dataset \
  --revision 740312add88f781978c0658806c59bc2815b9866
```

## Engineering smoke test

This command uses a synthetic canary to verify model loading, hooks, and memory
before a real study run:

```bash
uv run python scripts/smoke_model.py --model-key qwen35_2b
uv run python scripts/smoke_model.py --model-key qwen35_4b
uv run python scripts/smoke_model.py --model-key gemma4_e2b_it
uv run python scripts/smoke_model.py --model-key gemma4_e4b_it
```

It verifies loading, exact layer discovery, batch-4 generation, all-layer residual capture, hook execution, zero-alpha identity, and VRAM.

## Optional confirmatory replication protocol

This stricter optional path adds a frozen study contract, 100 random controls
per layer, and clean tagged launch identity. The commands below remain disabled
until that prospective configuration is frozen:

```bash
bash scripts/scientific.sh validate-study --require-frozen
```

`scripts/scientific.sh` requires Bash. On Windows, use Git Bash/WSL, or invoke the equivalent locked `uv run --extra quantization --extra benchmarks sycophancy-steering ...` command from PowerShell after clearing `PYTHONPATH` and `VIRTUAL_ENV`.

For each model key:

```bash
MODEL=qwen35_4b
ROOT=results/runs/$MODEL

bash scripts/scientific.sh fit-probe \
  --model-key "$MODEL" \
  --output-dir "$ROOT/fit_probe"

bash scripts/scientific.sh evaluate-frontier \
  --model-key "$MODEL" \
  --fit-probe-dir "$ROOT/fit_probe" \
  --output-dir "$ROOT/frontier"

bash scripts/scientific.sh evaluate-kl \
  --model-key "$MODEL" \
  --fit-probe-dir "$ROOT/fit_probe" \
  --frontier-dir "$ROOT/frontier" \
  --wikitext-path <pinned-wikitext-parquet> \
  --output-dir "$ROOT/neutral_kl"

bash scripts/scientific.sh evaluate-gsm8k \
  --model-key "$MODEL" \
  --fit-probe-dir "$ROOT/fit_probe" \
  --frontier-dir "$ROOT/frontier" \
  --gsm8k-path <pinned-gsm8k-parquet> \
  --output-dir "$ROOT/sampled_gsm8k"
```

Every scientific artifact records:

- frozen Git tag, commit/tree, tracked-content hash, tagged study/data/dependency-lock blob identities, and parent-stage code identity;
- absolute Git and `nvidia-smi` executable paths plus their file hashes;
- study, data-lock, benchmark input, direction, primitive-residual, and lockfile hashes;
- immutable model revision plus a protocol-specified expected content-tree hash and per-file SHA-256/size inventory over every cached checkpoint, config, and tokenizer file;
- model/tokenizer/chat hashes, EOS inventory, and exact precision/device/quantization inventory;
- project `.venv` package origins and Python/Torch/Transformers/CUDA/GPU versions;
- TF32, deterministic-kernel, cuDNN, batch-size, and seed settings;
- prompt hashes, raw responses, locked record contracts, and recomputable metrics;
- hook activity and peak GPU memory;
- atomic final status bound to a reproducible content-tree manifest with an exact directory inventory.

`build_paper_assets.py` refuses missing fit/probe parents or mixed launch identities and invokes the fit, frontier, KL, and GSM8K semantic verifiers before emitting any table row; aggregate values are recomputed from persisted tensors, responses, or per-token/per-example primitives.

## Repository layout

```text
configs/                 study configurations and hash-bound data contracts
data/                    provenance docs; local generated splits are ignored
docs/                    method, references, and reproducibility guidance
paper/                   LaTeX manuscript and bibliography
RESEARCH_STATUS.md       portable evidence and release status
results/                 verified current-study reports and release manifest
scripts/                  engineering and paper-generation commands
src/sycophancy_steering/ reusable package and CLI
tests/                   unit, integrity, and materialization tests
```

## Limitations

- Runtime steering is not a permanent model edit or a trained deployment artifact.
- Natural correct-update denominators depend on each model's unsteered errors; the controlled correction condition provides a common 1,310-record complement.
- The executed Qwen arms are not residual-dose matched; use a matched-`rho` grid before attributing their full response difference to model size.
- Gemma E4B is quantized while Qwen4B and Gemma E2B use bf16.
- The paired GSM8K subset measures intervention-induced change efficiently; a full-benchmark follow-up would provide narrower intervals.

## Citation

Please cite the preprint DOI [`10.5281/zenodo.22082268`](https://doi.org/10.5281/zenodo.22082268).
Machine-readable software and paper citation metadata are provided in
[`CITATION.cff`](CITATION.cff).

## License and attribution

AGPL-3.0-or-later. Portions derive from [Heretic](https://github.com/p-e-w/heretic) and retain upstream attribution. See [`NOTICE.md`](NOTICE.md), [`CITATION.cff`](CITATION.cff), and [`docs/references.md`](docs/references.md). Contribution and vulnerability-reporting rules are in [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`SECURITY.md`](SECURITY.md).

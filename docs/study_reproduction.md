# Reproducing the four-model study

This guide covers four useful levels of reproduction:

1. verify the published evidence bundle;
2. regenerate every paper table and plot without a model;
3. rerun the behavioral, GSM8K, and KL endpoints from the released directions;
4. refit the directions and rerun the complete executed protocol.

The preprint is archived at
[`10.5281/zenodo.22082268`](https://doi.org/10.5281/zenodo.22082268). Source,
paper PDF, and the portable evidence bundle are bound to the GitHub
[`v1.0.0`](https://github.com/and270/selective-sycophancy-steering/releases/tag/v1.0.0)
release.

## 1. Install the locked environment

Requirements are Python 3.11, `uv`, and an NVIDIA CUDA GPU for model runs.
CPU-only verification of the release bundle and manuscript assets does not load
any model.

```bash
git clone https://github.com/and270/selective-sycophancy-steering.git
cd selective-sycophancy-steering
git checkout v1.0.0
uv sync --locked --group dev --extra quantization --extra benchmarks
```

## 2. Download and verify the release bundle

Download
[`selective-sycophancy-study-artifacts-v1.0.0.zip`](https://github.com/and270/selective-sycophancy-steering/releases/download/v1.0.0/selective-sycophancy-study-artifacts-v1.0.0.zip)
from the release. The expected archive identity is:

```text
bytes: 32921615
sha256: 9bcfcd41c158e043b2e10bc517b8b753a63a2c0135390a46c842ec058a5be0b2
```

Verify the archive itself and every manifest-bound member:

```bash
uv run python scripts/verify_release_bundle.py \
  selective-sycophancy-study-artifacts-v1.0.0.zip
```

The command compares the embedded manifest with
`results/EXPLORATORY_ARTIFACT_MANIFEST.json`, requires an exact member
inventory, and recomputes every file size and SHA-256. Extract the verified
archive under `artifacts/v1.0.0/`:

```bash
mkdir -p artifacts/v1.0.0
unzip selective-sycophancy-study-artifacts-v1.0.0.zip -d artifacts/v1.0.0
```

PowerShell equivalents are:

```powershell
Get-FileHash -Algorithm SHA256 .\selective-sycophancy-study-artifacts-v1.0.0.zip
Expand-Archive .\selective-sycophancy-study-artifacts-v1.0.0.zip .\artifacts\v1.0.0
```

The bundle contains response-level behavior outputs, paired GSM8K examples,
per-token KL primitives, fit summaries, and selected direction tensors for all
four model arms:

| Model key | Study configuration | Released fit directory | Selected layer |
|---|---|---|---:|
| `qwen35_2b` | `configs/studies/multimodel_v1_qwen2b_extension.json` | `expanded_qwen35_2b_20260805/fit_probe` | 3 |
| `qwen35_4b` | `configs/studies/multimodel_v1.json` | `expanded_qwen35_4b_20260805/fit_probe` | 18 |
| `gemma4_e2b_it` | `configs/studies/multimodel_v1.json` | `expanded_gemma4_e2b_20260805/fit_probe` | 12 |
| `gemma4_e4b_it` | `configs/studies/multimodel_v1.json` | `expanded_gemma4_e4b_20260805/fit_probe` | 28 |

## 3. Regenerate the paper assets without a model

The manuscript tables and plots are generated from two tracked,
hash-pinned reports:

- `results/FOUR_MODEL_EXPLORATORY_FRONTIER.json`
- `results/INTERVENTION_SCALE_COMPARISON.json`

Check that every tracked generated asset is current:

```bash
uv run python scripts/build_exploratory_paper_assets.py --check
```

To regenerate the files and then verify them:

```bash
uv run python scripts/build_exploratory_paper_assets.py
uv run python scripts/build_exploratory_paper_assets.py --check
```

The outputs are written under `paper/generated/`. This stage performs no model
inference and does not manually transcribe empirical values.

## 4. Recreate the factual dataset

The source-derived factual questions are not redistributed because the
SycophancyEval repository did not expose an explicit redistribution license at
protocol freeze. Recreate the exact, hash-bound study splits from the pinned
upstream revision:

```bash
git clone https://github.com/meg-tong/sycophancy-eval data/raw/sycophancy-eval
git -C data/raw/sycophancy-eval checkout 9a1694221e3639887138f61deae344335eca6752

uv run sycophancy-steering materialize-data \
  --source data/raw/sycophancy-eval/datasets/answer.jsonl
```

The materializer verifies the upstream file hash, deterministic exclusions,
ordered record identities, source balance, A/B balance, and split disjointness.
It refuses to overwrite an existing materialized dataset.

## 5. Cache immutable model and benchmark inputs

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

The default scientific loaders are offline. Cache every checkpoint and dataset
before starting a run.

## 6. Rerun endpoints from the released directions

This is the most direct GPU reproduction of the reported intervention. It
reuses the released, hash-bound fit summary and direction tensor, then regenerates
the complete 1,310-record behavioral frontier, paired 256-item GSM8K sample,
and 64-context neutral trajectory KL evaluation.

Example for Qwen3.5-4B:

```bash
MODEL=qwen35_4b
STUDY=configs/studies/multimodel_v1.json
FIT=artifacts/v1.0.0/expanded_qwen35_4b_20260805/fit_probe
ROOT=results/runs/reproduction_${MODEL}

uv run python scripts/run_exploratory_behavior.py \
  --model-key "$MODEL" \
  --study "$STUDY" \
  --fit-dir "$FIT" \
  --output "$ROOT/expanded_behavior.json"

uv run python scripts/run_exploratory_gsm8k.py \
  --model-key "$MODEL" \
  --study "$STUDY" \
  --fit-dir "$FIT" \
  --gsm8k-path /path/to/pinned-gsm8k.parquet \
  --output "$ROOT/expanded_gsm8k.json"

uv run python scripts/run_exploratory_kl.py \
  --model-key "$MODEL" \
  --study "$STUDY" \
  --fit-dir "$FIT" \
  --wikitext-path /path/to/pinned-wikitext.parquet \
  --output "$ROOT/expanded_kl.json"
```

For Qwen3.5-2B, use
`configs/studies/multimodel_v1_qwen2b_extension.json`. The other three models
use `configs/studies/multimodel_v1.json`. Every runner verifies the study hash,
data-lock hash, model key, direction-file hash, complete evaluation inventory,
and output non-overwrite rule before inference.

## 7. Refit directions and rerun the complete protocol

To reproduce direction fitting and layer selection rather than reuse the
released direction, run the dedicated executed-protocol mode. It consumes all
300 fit records and all 100 physically separate probe records and applies the
same five seeded norm-matched random controls per layer used in the completed
panel:

```bash
MODEL=qwen35_4b
STUDY=configs/studies/multimodel_v1.json
ROOT=results/runs/refit_${MODEL}

uv run sycophancy-steering fit-probe \
  --run-kind executed_reproduction \
  --model-key "$MODEL" \
  --study "$STUDY" \
  --data-dir data/materialized/multimodel_v1 \
  --output-dir "$ROOT/fit_probe"
```

Then use `$ROOT/fit_probe` as `--fit-dir` in the three endpoint commands from
the previous section. This path recreates the executed intervention protocol.
The separate `scientific` run kind is reserved for future tagged protocols that
use the study configuration's larger 100-control bank.

## 8. Apply a released direction in another program

The public runtime API leaves model loading and generation under caller
control:

```python
from safetensors.torch import load_file
from sycophancy_steering import steer_model

layer = 18
directions = load_file(
    "artifacts/v1.0.0/expanded_qwen35_4b_20260805/fit_probe/"
    "directions.safetensors",
    device="cpu",
)
direction = directions["observed_prompt_state"][layer]

with steer_model(model, direction, layer_index=layer, alpha=-1.0) as audit:
    output = model.generate(**inputs)

assert audit.calls > 0
```

Directions are specific to the exact checkpoint revision, tokenizer,
representation site, and layer. See `docs/adding_models.md` before fitting or
applying a direction to another model family.

## Reproduction boundary

The release supports independent endpoint recomputation, complete GPU endpoint
reruns from published directions, and a new refit of the executed protocol from
the pinned source data and model revisions. The full local 3.7 GB run tree is
not committed because it contains redundant all-layer observations; the
portable bundle contains the selected directions and all response/metric
primitives used by the paper.

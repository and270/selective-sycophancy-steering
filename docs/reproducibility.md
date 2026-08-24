# Reproducibility and fail-closed execution

## Execution modes

Every fit/probe run declares one of three kinds:

- `engineering_smoke`: in the strict pipeline, may use synthetic canaries or
  explicit record limits and can verify loading, hooks, tensor shapes, and
  memory only;
- `executed_reproduction`: consumes every frozen fit/probe record with the
  recorded batch sizes and the five-control layer screen used by the completed
  four-model panel;
- `scientific`: cannot use limits and requires a frozen study JSON with no pending freeze items.

Smoke outputs are never consumed by scientific stages. The standalone completed-
study endpoint runners accept the released fit summaries and new
`executed_reproduction` fits. New confirmatory protocols should use the tagged
`scientific` stage from the start.

## Freeze boundary

`load_study_config(..., require_frozen=True)` requires all of:

```json
{
  "status": "frozen",
  "scientific_outputs_allowed": true,
  "freeze_pending": []
}
```

Until all three are true, fit/probe, frontier, KL, and sampled GSM8K scientific commands fail before model inference.

## Immutable inputs

The study binds:

- model Hub IDs and 40-character commit revisions;
- expected model class, block path, block count, and hidden width;
- dtype and quantization configuration;
- source dataset revisions and parquet hashes;
- factual split byte hashes and ordered-ID hashes;
- WikiText context and GSM8K sample identity hashes;
- `uv.lock` and preregistration hashes in every runtime artifact.

The study and lock additionally bind ordered minimal record contracts (source stratum, parent split, A/B truth labels, and pressure variants) for each factual split. Before assignment, all exact normalized-question duplicates and every record in a symmetric near-duplicate pair (maximum bidirectional `SequenceMatcher` ratio or whitespace-token Jaccard at least 0.90) are excluded and recorded by ID in the lock. Scientific launch requires the study JSON, data lock, and `uv.lock` to resolve inside the repository, be tracked, and match their `refs/tags/<required-tag>` Git blobs. The in-memory validated study object is bound to a canonical hash of the tagged payload, and the exact data-lock byte buffer used for parsing must match the launch hash. Network access is disabled by default for scientific model loading. Inputs should be cached first; every regular file in the resolved model snapshot, including all weight shards, is SHA-256 hashed, and the canonical content tree must equal the model's preregistered expected tree before tokenizer/model deserialization from that local path.

## Data access boundaries

The data API accepts an explicit allowlist and returns only those splits:

```text
fit/probe stage -> direction_fit, direction_probe
frontier stage  -> evaluation
```

It rejects empty, duplicate, unknown, path-escaping, missing, modified, miscounted, or duplicate-ID splits.

## Artifact lifecycle

Each stage:

1. refuses to overwrite an existing output directory;
2. creates `status.json` with `complete: false`;
3. writes large intermediate progress to a checkpoint JSON;
4. verifies upstream content manifests, tensors, primitive observations/responses, derived metrics, and frozen code identity;
5. writes the final JSON atomically with strict `allow_nan=false` serialization;
6. deletes the checkpoint only after final output exists;
7. writes `artifact_manifest.json` over the exact final payload inventory;
8. sets `status.json` to `complete: true` last, binding the manifest SHA-256.

A partial directory, modified payload, unmanifested extra path, symlink, copied parent-identity digest with a different body, or self-reported aggregate that differs from persisted primitives cannot be mistaken for a completed verified stage. Paper tables are generated only after traversing the full fit/probe → frontier → KL/GSM parent chain and rerunning the same semantic verifiers.

## Direction integrity

Directions and fit/probe primitive residuals are saved as `safetensors`. The fit/probe artifact records file and tensor SHA-256, tensor name/shape/dtype, per-layer direction norms, raw baseline/follow-up answers, full descriptor source/correct-option strata, and locked minimal record contracts. Completion fitting is ineligible if any source×correct-option cell has fewer than 10 eligible records; its fallback probe requires at least five records per cell and gates every layer on pooled, pressure-mode, correct-option, source, and source×option AUROCs. Downstream verification reconstructs `DirectionObservations` and independently recomputes both estimators, subgroup/class counts, every AUROC, seeded random null threshold, gate, and selected layer. Frontier verification independently reparses every raw response, validates exact prompt-hash/denominator inventories and hook audits, recomputes all integer metrics and 10,000-iteration paired comparisons, and enforces zero-alpha raw/prompt identity. KL verification recomputes per-context means, prompt-macro bootstrap summaries, token-micro summaries, hook totals, trajectory/EOS identities, and exact frontier trial correspondence from per-token primitives. GSM8K verification recomputes per-example correctness, counts, accuracies, Wilson intervals, paired bootstrap comparisons, frozen sample identity, and exact frontier condition correspondence.

## Hook evidence

Every steered condition records:

- total hook calls;
- prefill calls;
- cached-decode calls;
- modified batch rows.

A configured intervention with zero hook calls raises an error. The `alpha=0` frontier condition must have hook activity and exact raw-output identity with unsteered base.

## Verified engineering envelope

Synthetic real-checkpoint smokes verified batch 4, exact hook activity, zero-alpha identity, cached decode, full EOS inventories, and trajectory replay on Qwen3.5-4B and both Gemma 4 arms. No-inference tokenization over the final contracts found factual prompt maxima of 157 tokens for Qwen and 141 for Gemma; sampled GSM8K maxima were 173 and 171 respectively.

## Launch/runtime manifest

Each final artifact records:

- frozen Git tag, commit/tree, SHA-256 over every tracked path/content, and tagged-blob identities for the study, data lock, and `uv.lock`;
- absolute Git and `nvidia-smi` paths plus executable SHA-256;
- project `.venv` prefix and core package origins;
- command line and stage;
- repository commit and dirty state;
- Python/platform versions;
- Torch, Transformers, Accelerate, bitsandbytes, CUDA, and GPU versions;
- model ID/revision/class/layer path, per-file checkpoint content tree, config/tokenizer/chat hashes, live device map, dtype/quantized-module counts, precision, and EOS inventory;
- study, data-lock, and `uv.lock` hashes;
- TF32 flags;
- accessed factual splits;
- peak allocated GPU memory.

A scientific stage refuses to start unless Git is available from an absolute non-repository path, the repository is clean, `HEAD` matches `refs/tags/<required-tag>` (a same-named branch is insufficient), the study/data/dependency locks match tracked tagged blobs, the loaded study object and parsed data-lock bytes match that captured identity, `PYTHONPATH` is unset, and core packages originate under the project `.venv`. The identity—including executable hashes—is recaptured before artifact publication; every persisted launch digest is recomputed from its body, and every parent must carry the exact same full identity body.

## Determinism

- greedy decoding (`do_sample=false`);
- model and CUDA seeds fixed to zero;
- TF32 disabled;
- frozen batch sizes;
- stable hash ordering for every sample/split;
- exact A/B one-token generation;
- bootstrap seeds fixed to `20260805`.

`torch.use_deterministic_algorithms` is not forced because some model kernels may lack deterministic implementations. The runtime manifest and exact raw responses make any rerun difference observable.

## Verification commands

```bash
uv lock --check
uv run --extra quantization --extra benchmarks python -m pytest -q
uv run --extra quantization --extra benchmarks ruff check src tests scripts
uv run --extra quantization --extra benchmarks ruff format --check src tests scripts
uv run --extra quantization --extra benchmarks ty check src
```

The local test suite includes:

- hook scope and removal;
- layer discovery and count drift;
- exact-site block-output capture;
- direction math and AUROC ties;
- random-control determinism;
- strict prompt/response contracts;
- natural correct-suggestion update and controlled correction-acceptance denominators;
- paired record-cluster bootstrap;
- data tamper and duplicate-ID rejection;
- exact regeneration of frozen split hashes;
- KL and sampled GSM8K scoring.

## Known limitations

- E4B requires NF4 on 12GB VRAM, limiting absolute cross-model comparability.
- Raw direction magnitudes are not normalized across architectures.
- Scientific stage checkpointing is condition-level, not per generated response; a mid-condition failure requires removing the incomplete output directory and rerunning that stage.
- Flash-linear-attention is unavailable; the slower PyTorch fallback is used for Qwen3.5.
